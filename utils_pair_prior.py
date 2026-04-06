import os
import torch
import torchvision
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from diffusers import (
    PriorTransformer,
    StableUnCLIPImg2ImgPipeline,
    DDPMScheduler,
    DDIMScheduler,
)


# =========================
# Logging / dataset helpers
# =========================

def setup_logging(run_name: str) -> str:
    base_dir = os.path.join("experiments", run_name)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    return base_dir


class EmbeddingDataset(Dataset):
    def __init__(self, data_dir, image_dir1, image_dir2, partition="train"):
        self.image_dir1 = image_dir1
        self.image_dir2 = image_dir2

        self.embeds1 = torch.load(
            os.path.join(data_dir, f"{partition}_dataset1_embeds.pt"),
            weights_only=True,
        ).to(torch.float32)

        self.embeds2 = torch.load(
            os.path.join(data_dir, f"{partition}_dataset2_embeds.pt"),
            weights_only=True,
        ).to(torch.float32)

        self.df = pd.read_csv(os.path.join(data_dir, "partition.csv"))
        self.df = self.df[self.df["Partition"] == partition].reset_index(drop=True)

    def __getitem__(self, index):
        name1 = self.df.iloc[index]["Image1_Path"]
        name2 = self.df.iloc[index]["Image2_Path"]

        path1 = os.path.join(self.image_dir1, name1)
        path2 = os.path.join(self.image_dir2, name2)

        return self.embeds1[index], self.embeds2[index], path1, path2

    def __len__(self):
        return len(self.embeds1)


# ======================================
# Pair-state helpers for the Gaussian prior
# ======================================

PAIR_EMBED_DIM = 768
PAIR_STATE_DIM = PAIR_EMBED_DIM * 2


def make_pair_state(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    """Concatenate the two clean embeddings into a single state vector."""
    return torch.cat([e1, e2], dim=-1)



def split_pair_state(pair_state: torch.Tensor):
    """Split a concatenated pair-state back into two embeddings."""
    if pair_state.shape[-1] != PAIR_STATE_DIM:
        raise ValueError(
            f"Expected pair_state last dimension to be {PAIR_STATE_DIM}, got {pair_state.shape[-1]}"
        )
    return pair_state[..., :PAIR_EMBED_DIM], pair_state[..., PAIR_EMBED_DIM:]



def make_average_condition(e1: torch.Tensor, e2: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Conditioning signal used by the prior: weighted average embedding."""
    return e1 * (1.0 - alpha) + e2 * alpha



def maybe_drop_condition(cond: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0.0:
        return cond

    keep_mask = (torch.rand(cond.shape[0], device=cond.device) >= drop_prob).to(cond.dtype)
    return cond * keep_mask.unsqueeze(-1)



def permutation_invariant_pair_loss(pred_pair: torch.Tensor, target_pair: torch.Tensor):
    """
    Computes a permutation-invariant MSE over the two predicted embeddings.

    Returns:
        loss: scalar loss
        swap_mask: boolean tensor of shape [B], True where swapping the target order is better.
    """
    p1, p2 = split_pair_state(pred_pair)
    t1, t2 = split_pair_state(target_pair)

    loss_12 = ((p1 - t1) ** 2).mean(dim=-1) + ((p2 - t2) ** 2).mean(dim=-1)
    loss_21 = ((p1 - t2) ** 2).mean(dim=-1) + ((p2 - t1) ** 2).mean(dim=-1)

    swap_mask = loss_21 < loss_12
    best = torch.where(swap_mask, loss_21, loss_12)
    return best.mean(), swap_mask



def align_prediction_to_target(pred_pair: torch.Tensor, target_pair: torch.Tensor) -> torch.Tensor:
    """Reorder the predicted pair so metric computation matches the ground-truth pair ordering."""
    _, swap_mask = permutation_invariant_pair_loss(pred_pair, target_pair)

    p1, p2 = split_pair_state(pred_pair)
    aligned_first = torch.where(swap_mask.unsqueeze(-1), p2, p1)
    aligned_second = torch.where(swap_mask.unsqueeze(-1), p1, p2)
    return make_pair_state(aligned_first, aligned_second)


# =========================
# Prior / schedulers
# =========================


def get_pair_prior_model(num_layers: int = 12) -> PriorTransformer:
    """
    PriorTransformer adapted to predict a concatenated pair state.

    Key idea:
      - hidden_states / target live in a 1536-d space (two concatenated 768-d CLIP image embeddings)
      - proj_embedding is still the 768-d average embedding
      - output is also 1536-d, so the model predicts both clean embeddings jointly
    """
    return PriorTransformer(
        embedding_dim=PAIR_STATE_DIM,
        num_attention_heads=24,
        num_layers=num_layers,
        embedding_proj_dim=PAIR_EMBED_DIM,
        clip_embed_dim=PAIR_STATE_DIM,
        num_embeddings=0,
        additional_embeddings=3,
        encoder_hid_proj_type=None,
        added_emb_type=None,
    )



def get_train_noise_scheduler(args) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type="sample",
        clip_sample=False,
    )



def get_inference_scheduler(args) -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type="sample",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


@torch.no_grad()
def sample_pair_embeddings(
    model,
    cond_avg: torch.Tensor,
    scheduler: DDIMScheduler,
    num_inference_steps: int,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    DDIM-style sampling in the same x0-prediction regime used by the diffusion prior paper.

    The model predicts x0 directly (prediction_type='sample'). The scheduler converts that prediction
    into x_{t-1} updates.
    """
    device = cond_avg.device
    dtype = cond_avg.dtype
    batch_size = cond_avg.shape[0]

    scheduler.set_timesteps(num_inference_steps, device=device)

    if generator is None:
        x_t = torch.randn((batch_size, PAIR_STATE_DIM), device=device, dtype=dtype)
    else:
        x_t = torch.randn((batch_size, PAIR_STATE_DIM), generator=generator, device=device, dtype=dtype)

    null_cond = torch.zeros_like(cond_avg)

    was_training = model.training
    model.eval()

    for t in scheduler.timesteps:
        if guidance_scale == 1.0:
            pred_x0 = model(
                hidden_states=x_t,
                timestep=t,
                proj_embedding=cond_avg,
            ).predicted_image_embedding
        else:
            pred_x0_uncond = model(
                hidden_states=x_t,
                timestep=t,
                proj_embedding=null_cond,
            ).predicted_image_embedding

            pred_x0_cond = model(
                hidden_states=x_t,
                timestep=t,
                proj_embedding=cond_avg,
            ).predicted_image_embedding

            pred_x0 = pred_x0_uncond + guidance_scale * (pred_x0_cond - pred_x0_uncond)

        step_out = scheduler.step(pred_x0, t, x_t, eta=eta)
        x_t = step_out.prev_sample

    if was_training:
        model.train()

    return x_t


# =========================
# Decoder helpers
# =========================


def get_decoder_pipeline(device):
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-1-unclip-small",
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline



def encode_pil_images_to_embeddings(pil_images, pipeline, device):
    if not isinstance(pil_images, list):
        pil_images = [pil_images]

    encoder_dtype = next(pipeline.image_encoder.parameters()).dtype

    inputs = pipeline.feature_extractor(images=pil_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=encoder_dtype)

    with torch.no_grad():
        image_embeds = pipeline.image_encoder(pixel_values=pixel_values).image_embeds

    return image_embeds.to(torch.float32)



def decode_embeddings_to_images(embeddings, pipeline, device, seed=0):
    embeddings = embeddings.to(device=device, dtype=pipeline.unet.dtype)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    images = pipeline(
        prompt=[""] * embeddings.shape[0],
        image_embeds=embeddings,
        noise_level=0,
        num_inference_steps=20,
        generator=generator,
    ).images

    transform = torchvision.transforms.ToTensor()
    image_tensors = torch.stack([transform(img) for img in images]).to(device)
    return image_tensors



def save_labeled_image_row(image_tensors, labels, save_path, tile_size=(64, 64), pad=6, label_pad=4, label_height=18):
    assert len(image_tensors) == len(labels), "image_tensors and labels must have the same length"

    pil_images = []
    to_pil = torchvision.transforms.ToPILImage()

    for img in image_tensors:
        img = img.detach().cpu().clamp(0, 1)
        pil = to_pil(img)
        pil = pil.resize(tile_size, Image.Resampling.BILINEAR)
        pil_images.append(pil)

    n = len(pil_images)
    tile_w, tile_h = tile_size
    canvas_w = n * tile_w + (n + 1) * pad
    canvas_h = pad + tile_h + label_pad + label_height + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (pil, label) in enumerate(zip(pil_images, labels)):
        x = pad + i * (tile_w + pad)
        y = pad
        canvas.paste(pil, (x, y))

        if font is not None:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
        else:
            text_w = len(label) * 6

        text_x = x + max((tile_w - text_w) // 2, 0)
        text_y = y + tile_h + label_pad
        draw.text((text_x, text_y), label, fill="black", font=font)

    canvas.save(save_path)


# =========================
# Evaluation helpers
# =========================


def compute_pair_embedding_metrics_over_testset(args, test_dataloader, model, scheduler):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_vectors = 0
    total_elements = 0

    mse_1_sum = 0.0
    mse_2_sum = 0.0
    cos_1_sum = 0.0
    cos_2_sum = 0.0

    generator = None
    if args.eval_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        for e1, e2, _, _ in test_dataloader:
            e1 = e1.to(device)
            e2 = e2.to(device)

            target_pair = make_pair_state(e1, e2)
            cond_avg = make_average_condition(e1, e2, alpha=args.cond_alpha)

            pred_pair = sample_pair_embeddings(
                model=model,
                cond_avg=cond_avg,
                scheduler=scheduler,
                num_inference_steps=args.num_inference_steps,
                eta=args.eta,
                guidance_scale=args.cfg_scale,
                generator=generator,
            )

            pred_pair = align_prediction_to_target(pred_pair, target_pair)
            pred_e1, pred_e2 = split_pair_state(pred_pair)

            mse_1_sum += ((pred_e1 - e1) ** 2).sum().item()
            mse_2_sum += ((pred_e2 - e2) ** 2).sum().item()

            cos_1_sum += F.cosine_similarity(pred_e1, e1, dim=-1).sum().item()
            cos_2_sum += F.cosine_similarity(pred_e2, e2, dim=-1).sum().item()

            total_vectors += e1.shape[0]
            total_elements += e1.numel()

    return {
        "mse_target": mse_1_sum / total_elements,
        "mse_deduced": mse_2_sum / total_elements,
        "mse_mean": (mse_1_sum + mse_2_sum) / (2 * total_elements),
        "cos_target": cos_1_sum / total_vectors,
        "cos_deduced": cos_2_sum / total_vectors,
        "cos_mean": (cos_1_sum + cos_2_sum) / (2 * total_vectors),
    }



def run_pair_image_evaluation(args, base_dir, test_dataloader, model, scheduler):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder_pipeline = get_decoder_pipeline(device)

    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    def load_real_image_64(path):
        img = Image.open(path).convert("RGB")
        img = TF.to_tensor(TF.resize(img, (64, 64), antialias=True))
        return img.unsqueeze(0).to(device)

    def resize_decoded(img_tensor):
        return F.interpolate(img_tensor, size=(64, 64), mode="area")

    saved_path = None

    generator = None
    if args.eval_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.eval_seed)

    model.eval()
    with torch.no_grad():
        for e1, e2, p1, p2 in test_dataloader:
            e1 = e1.to(device)
            e2 = e2.to(device)

            target_pair = make_pair_state(e1, e2)
            cond_avg = make_average_condition(e1, e2, alpha=args.cond_alpha)
            pred_pair = sample_pair_embeddings(
                model=model,
                cond_avg=cond_avg,
                scheduler=scheduler,
                num_inference_steps=args.num_inference_steps,
                eta=args.eta,
                guidance_scale=args.cfg_scale,
                generator=generator,
            )
            pred_pair = align_prediction_to_target(pred_pair, target_pair)
            pred_e1, pred_e2 = split_pair_state(pred_pair)

            emb_e1 = e1[0:1]
            emb_e2 = e2[0:1]
            emb_avg = cond_avg[0:1]
            emb_pred1 = pred_e1[0:1]
            emb_pred2 = pred_e2[0:1]

            real_img1 = load_real_image_64(p1[0])
            real_img2 = load_real_image_64(p2[0])
            real_mix = real_img1 * (1.0 - args.cond_alpha) + real_img2 * args.cond_alpha

            dec_e1 = resize_decoded(decode_embeddings_to_images(emb_e1, decoder_pipeline, device, seed=100))
            dec_e2 = resize_decoded(decode_embeddings_to_images(emb_e2, decoder_pipeline, device, seed=101))
            dec_avg = resize_decoded(decode_embeddings_to_images(emb_avg, decoder_pipeline, device, seed=102))
            dec_pred1 = resize_decoded(decode_embeddings_to_images(emb_pred1, decoder_pipeline, device, seed=103))
            dec_pred2 = resize_decoded(decode_embeddings_to_images(emb_pred2, decoder_pipeline, device, seed=104))

            tiles = [
                real_img1.squeeze(0),
                real_img2.squeeze(0),
                real_mix.squeeze(0),
                dec_e1.squeeze(0),
                dec_e2.squeeze(0),
                dec_avg.squeeze(0),
                dec_pred1.squeeze(0),
                dec_pred2.squeeze(0),
            ]

            labels = [
                "Real 1",
                "Real 2",
                "Real Avg",
                "Dec E1",
                "Dec E2",
                "Dec Avg",
                "Pred Dec 1",
                "Pred Dec 2",
            ]

            saved_path = os.path.join(base_dir, "results", "gaussian_pair_prior_8_tile_row_labeled.png")
            save_labeled_image_row(tiles, labels, saved_path, tile_size=(64, 64))
            break

    del decoder_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return f"Saved labeled image row to: {saved_path}\n"

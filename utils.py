import os
import torch
import math
import torchvision
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from diffusers import PriorTransformer, StableUnCLIPImg2ImgPipeline

def setup_logging(run_name):
    base_dir = os.path.join("experiments", run_name)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    return base_dir

class EmbeddingDataset(Dataset):
    def __init__(self, data_dir, image_dir1, image_dir2, partition="train", align_to_mean=True):
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

        # --- NEW SORTING LOGIC ---
        if align_to_mean:
            print(f"Aligning {partition} dataset pairs to the global mean...")
            
            # 1. Calculate global mean across all embeddings
            global_mean = torch.cat([self.embeds1, self.embeds2], dim=0).mean(dim=0)
            
            # 2. Compute MSE distance to the mean for both arrays
            dist1 = F.mse_loss(self.embeds1, global_mean.unsqueeze(0).expand_as(self.embeds1), reduction="none").mean(dim=1)
            dist2 = F.mse_loss(self.embeds2, global_mean.unsqueeze(0).expand_as(self.embeds2), reduction="none").mean(dim=1)
            
            # 3. Create mask where e2 is closer to the mean than e1
            swap_mask = dist2 < dist1
            
            # 4. Swap embeddings using the mask
            e1_aligned = torch.where(swap_mask.unsqueeze(1), self.embeds2, self.embeds1)
            e2_aligned = torch.where(swap_mask.unsqueeze(1), self.embeds1, self.embeds2)
            self.embeds1 = e1_aligned
            self.embeds2 = e2_aligned
            
            # 5. Swap dataframe paths to keep image paths synced with the swapped embeddings
            swap_idx = swap_mask.nonzero(as_tuple=True)[0].tolist()
            temp_paths = self.df.loc[swap_idx, "Image1_Path"].copy()
            self.df.loc[swap_idx, "Image1_Path"] = self.df.loc[swap_idx, "Image2_Path"]
            self.df.loc[swap_idx, "Image2_Path"] = temp_paths

    def __getitem__(self, index):
        name1 = self.df.iloc[index]["Image1_Path"]
        name2 = self.df.iloc[index]["Image2_Path"]

        path1 = os.path.join(self.image_dir1, name1)
        path2 = os.path.join(self.image_dir2, name2)

        return self.embeds1[index], self.embeds2[index], path1, path2

    def __len__(self):
        return len(self.embeds1)

class ColdDiffusionEmbeds:
    def __init__(self, max_timesteps=10, alpha_start=0.0, alpha_max=0.5, device="cuda"):
        self.max_timesteps = max_timesteps
        self.device = device
        self.alpha_start = alpha_start
        self.alpha_max = alpha_max
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps

    def mix_embeds(self, emb_1, emb_2, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=emb_1.device)

        if t.dim() == 0:
            t = t.expand(emb_1.shape[0])

        alpha_t = self.alteration_per_t * t.float()
        alpha_t = alpha_t.unsqueeze(1)

        return emb_1 * (1.0 - alpha_t) + emb_2 * alpha_t

    def sample_timesteps(self, n, max_t=None):
        if max_t is None:
            max_t = self.max_timesteps
        return torch.randint(low=1, high=max_t + 1, size=(n,), device=self.device)

    def sample(self, model, superimposed_emb, alpha_init=0.5):
        n = len(superimposed_emb)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)

        model.eval()
        with torch.no_grad():
            x_t = superimposed_emb.clone()

            for i in reversed(range(1, init_timestep + 1)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)
                null_proj = torch.zeros_like(x_t)

                predicted_emb = model(
                    hidden_states=x_t,
                    timestep=t,
                    proj_embedding=null_proj,
                ).predicted_image_embedding.squeeze(1)

                other_emb = (superimposed_emb - (1.0 - alpha_init) * predicted_emb) / alpha_init

                x_t = self.mix_embeds(predicted_emb, other_emb, t - 1) #(
                    #x_t
                    #- self.mix_embeds(predicted_emb, other_emb, t)
                    #+ self.mix_embeds(predicted_emb, other_emb, t - 1)
                #)

        model.train()
        other_emb = (superimposed_emb - (1.0 - alpha_init) * x_t) / alpha_init
        return x_t, other_emb

def get_prior_model():
    return PriorTransformer(
        embedding_dim=768,
        num_attention_heads=12,
        num_layers=12,
        embedding_proj_dim=768,
        clip_embed_dim=768,
        num_embeddings=0,
        additional_embeddings=3,
        encoder_hid_proj_type=None,
        added_emb_type=None,
    )

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

def compute_embedding_metrics_over_testset(args, test_dataloader, model, diffusion, mode="Regular"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_vectors = 0
    total_elements = 0

    mse_t_sum = 0.0
    mse_d_sum = 0.0
    cos_t_sum = 0.0
    cos_d_sum = 0.0

    model.eval()
    with torch.no_grad():
        for e1, e2, _, _ in test_dataloader:
            e1 = e1.to(device)
            e2 = e2.to(device)
            n = e1.shape[0]

            S = e1 * (1.0 - args.alpha_init) + e2 * args.alpha_init

            if mode == "Regular":
                pred_e1, pred_e2 = diffusion.sample(model, S, args.alpha_init)
            else:
                init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
                t = torch.full((n,), init_timestep, device=device, dtype=torch.long)
                null_proj = torch.zeros_like(S)

                pred_e1 = model(
                    hidden_states=S,
                    timestep=t,
                    proj_embedding=null_proj,
                ).predicted_image_embedding.squeeze(1)

                pred_e2 = (S - (1.0 - args.alpha_init) * pred_e1) / args.alpha_init

            mse_t_sum += ((pred_e1 - e1) ** 2).sum().item()
            mse_d_sum += ((pred_e2 - e2) ** 2).sum().item()

            cos_t_sum += F.cosine_similarity(pred_e1, e1, dim=-1).sum().item()
            cos_d_sum += F.cosine_similarity(pred_e2, e2, dim=-1).sum().item()

            total_vectors += n
            total_elements += e1.numel()

    metrics = {
        "mse_target": mse_t_sum / total_elements,
        "mse_deduced": mse_d_sum / total_elements,
        "mse_mean": (mse_t_sum + mse_d_sum) / (2 * total_elements),
        "cos_target": cos_t_sum / total_vectors,
        "cos_deduced": cos_d_sum / total_vectors,
        "cos_mean": (cos_t_sum + cos_d_sum) / (2 * total_vectors),
    }

    return metrics

def run_image_evaluation(args, base_dir, test_dataloader, model, diffusion, mode="Regular"):
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

    model.eval()
    with torch.no_grad():
        for e1, e2, p1, p2 in test_dataloader:
            e1 = e1.to(device)
            e2 = e2.to(device)
            n = len(e1)

            S = e1 * (1.0 - args.alpha_init) + e2 * args.alpha_init

            if mode == "Regular":
                pred_e1, pred_e2 = diffusion.sample(model, S, args.alpha_init)
            else:
                init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
                t = torch.full((n,), init_timestep, device=device, dtype=torch.long)
                null_proj = torch.zeros_like(S)

                pred_e1 = model(
                    hidden_states=S,
                    timestep=t,
                    proj_embedding=null_proj,
                ).predicted_image_embedding.squeeze(1)

                pred_e2 = (S - (1.0 - args.alpha_init) * pred_e1) / args.alpha_init

            emb_e1 = e1[0:1]
            emb_e2 = e2[0:1]
            emb_S = S[0:1]
            emb_pred1 = pred_e1[0:1]
            emb_pred2 = pred_e2[0:1]

            real_img1 = load_real_image_64(p1[0])
            real_img2 = load_real_image_64(p2[0])
            real_mix = real_img1 * (1.0 - args.alpha_init) + real_img2 * args.alpha_init

            dec_e1 = resize_decoded(decode_embeddings_to_images(emb_e1, decoder_pipeline, device, seed=100))
            dec_e2 = resize_decoded(decode_embeddings_to_images(emb_e2, decoder_pipeline, device, seed=101))
            dec_S = resize_decoded(decode_embeddings_to_images(emb_S, decoder_pipeline, device, seed=102))
            dec_pred1 = resize_decoded(decode_embeddings_to_images(emb_pred1, decoder_pipeline, device, seed=103))
            dec_pred2 = resize_decoded(decode_embeddings_to_images(emb_pred2, decoder_pipeline, device, seed=104))

            blank = torch.zeros_like(real_img1)

            tiles = [
                real_img1.squeeze(0),
                real_img2.squeeze(0),
                real_mix.squeeze(0),
                dec_e1.squeeze(0),
                dec_e2.squeeze(0),
                dec_S.squeeze(0),
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

            saved_path = os.path.join(base_dir, "results", f"{mode.lower()}_8_tile_row_labeled.png")
            save_labeled_image_row(tiles, labels, saved_path, tile_size=(64, 64))
            break

    del decoder_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return f"Saved {mode} labeled image row to: {saved_path}\n"

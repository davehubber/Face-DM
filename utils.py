import os
import torch
import math
import numpy as np
import torchvision
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from diffusers import PriorTransformer, StableUnCLIPImg2ImgPipeline
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def setup_logging(run_name):
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

class ColdDiffusionEmbeds:
    def __init__(self, max_timesteps=10, alpha_start=0., alpha_max=0.5, device="cuda"):
        self.max_timesteps = max_timesteps
        self.device = device
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps

    def mix_embeds(self, emb_1, emb_2, t):
        w1 = (1. - self.alteration_per_t * t).unsqueeze(1)
        w2 = (self.alteration_per_t * t).unsqueeze(1)
        mixed = emb_1 * w1 + emb_2 * w2
        
        scale = torch.norm(emb_1, p=2, dim=-1, keepdim=True) 
        mixed = torch.nn.functional.normalize(mixed, p=2, dim=-1) * scale
        
        return mixed

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def sample(self, model, superimposed_emb, alpha_init=0.5):
        n = len(superimposed_emb)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        model.eval()
        with torch.no_grad():
            x_t = superimposed_emb.clone()

            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_emb = model(hidden_states=x_t, timestep=t, proj_embedding=superimposed_emb).predicted_image_embedding.squeeze(1)
                other_emb = (superimposed_emb - (1. - alpha_init) * predicted_emb) / alpha_init
                x_t = x_t - self.mix_embeds(predicted_emb, other_emb, t) + self.mix_embeds(predicted_emb, other_emb, t-1)

        model.train()
        other_emb = (superimposed_emb - (1 - alpha_init) * x_t) / alpha_init
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

def decode_embeddings_to_images(embeddings, pipeline, device):
    embeddings = embeddings.to(device=device, dtype=pipeline.unet.dtype)

    images = pipeline(
        prompt=[""] * embeddings.shape[0],
        image_embeds=embeddings,
        noise_level=0,
        num_inference_steps=20,
    ).images

    transform = torchvision.transforms.ToTensor()
    image_tensors = torch.stack([transform(img) for img in images]).to(device)
    return image_tensors

def compute_image_metrics(gt, pred, avg, psnr_fn, ssim_fn, lpips_fn):
    psnr_val = psnr_fn(pred, gt).item()
    ssim_val = ssim_fn(pred, gt).item()

    gt_lpips = gt * 2.0 - 1.0
    pred_lpips = pred * 2.0 - 1.0
    avg_lpips = avg * 2.0 - 1.0

    lpips_val = lpips_fn(pred_lpips, gt_lpips).item()

    dist_to_gt = lpips_val
    dist_to_avg = lpips_fn(pred_lpips, avg_lpips).item()
    success = 1 if dist_to_gt < dist_to_avg else 0

    return psnr_val, ssim_val, lpips_val, success

def run_image_evaluation(args, base_dir, test_dataloader, model, diffusion, mode="Regular"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

    decoder_pipeline = get_decoder_pipeline(device)

    num_eval_images = 1
    evaluated_count = 0

    psnr_list, ssim_list, lpips_list = [], [], []
    success_count = 0
    grid_images = []

    for e1, e2, p1, p2 in test_dataloader:
        e1, e2 = e1.to(device), e2.to(device)
        n = len(e1)
        S = e1 * (1 - args.alpha_init) + e2 * args.alpha_init

        scale = torch.norm(e1, p=2, dim=-1, keepdim=True)
        S = torch.nn.functional.normalize(S, p=2, dim=-1) * scale

        with torch.no_grad():
            if mode == "Regular":
                pred_e1, pred_e2 = diffusion.sample(model, S, args.alpha_init)
            else:
                init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
                t = (torch.ones(n) * init_timestep).long().to(device)
                pred_e1 = model(
                    hidden_states=S,
                    timestep=t,
                    proj_embedding=S,
                ).predicted_image_embedding.squeeze(1)
                pred_e2 = (S - (1 - args.alpha_init) * pred_e1) / args.alpha_init

        for i in range(n):
            if evaluated_count >= num_eval_images:
                break

            emb_p1 = pred_e1[i : i + 1]
            emb_p2 = pred_e2[i : i + 1]
            emb_S = S[i : i + 1]

            real_img1 = Image.open(p1[i]).convert("RGB")
            real_img2 = Image.open(p2[i]).convert("RGB")

            gt1_img = TF.to_tensor(TF.resize(real_img1, (64, 64))).unsqueeze(0).to(device)
            gt2_img = TF.to_tensor(TF.resize(real_img2, (64, 64))).unsqueeze(0).to(device)

            avg_img_large = decode_embeddings_to_images(emb_S, decoder_pipeline, device)
            pred1_img_large = decode_embeddings_to_images(emb_p1, decoder_pipeline, device)
            pred2_img_large = decode_embeddings_to_images(emb_p2, decoder_pipeline, device)

            avg_img = F.interpolate(avg_img_large, size=(64, 64), mode="area")
            pred1_img = F.interpolate(pred1_img_large, size=(64, 64), mode="area")
            pred2_img = F.interpolate(pred2_img_large, size=(64, 64), mode="area")

            p, s, l, succ = compute_image_metrics(gt1_img, pred1_img, avg_img, psnr_metric, ssim_metric, lpips_metric)
            psnr_list.append(p)
            ssim_list.append(s)
            lpips_list.append(l)
            success_count += succ

            p, s, l, succ = compute_image_metrics(gt2_img, pred2_img, avg_img, psnr_metric, ssim_metric, lpips_metric)
            psnr_list.append(p)
            ssim_list.append(s)
            lpips_list.append(l)
            success_count += succ

            grid_images.extend(
                [
                    gt1_img.squeeze(0),
                    gt2_img.squeeze(0),
                    avg_img.squeeze(0),
                    pred1_img.squeeze(0),
                    pred2_img.squeeze(0),
                ]
            )
            evaluated_count += 1

        if evaluated_count >= num_eval_images:
            break

    del decoder_pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(grid_images) > 0:
        grid_tensor = torch.stack(grid_images)
        grid_path = os.path.join(base_dir, "results", f"{mode.lower()}_1_pairs_grid.png")
        torchvision.utils.save_image(grid_tensor, grid_path, nrow=5)

    success_rate = (success_count / (num_eval_images * 2)) * 100

    img_metrics_report = (
        f"\n--- {mode} Image Evaluation Metrics (Calculated on 1 Pairs at 64x64) ---\n"
        f"PSNR: {np.mean(psnr_list):.4f}\n"
        f"SSIM: {np.mean(ssim_list):.4f}\n"
        f"LPIPS: {np.mean(lpips_list):.4f}\n"
        f"Success Rate of Reversal (%S): {success_rate:.2f}%\n"
    )
    return img_metrics_report
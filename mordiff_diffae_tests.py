import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- Configuration & Paths ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_MORPHS = 100

PATH_TO_DIFFAE = "/nas-ctm01/homes/dacordeiro/diffae/"
CHECKPOINT_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt"

FFHQ_DIR = Path("/nas-ctm01/datasets/public/ffhq256/")
OUTPUT_MORPHS_DIR = Path("/nas-ctm01/homes/dacordeiro/controlled_ffhq_morphs")
OUTPUT_MORPHS_DIR.mkdir(parents=True, exist_ok=True)

# Z-score normalization files
MEAN_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_train_mean.npy"
STD_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_train_std.npy"
REPORT_PATH = "controlled_morph_interpolation_report.txt"

# Make DiffAE importable
sys.path.insert(0, str(PATH_TO_DIFFAE))
old_cwd = os.getcwd()
os.chdir(PATH_TO_DIFFAE)
from templates import ffhq256_autoenc
from experiment import LitModel
os.chdir(old_cwd)


def load_diffae_model():
    conf = ffhq256_autoenc()
    conf.latent_infer_path = None
    model = LitModel(conf)
    state = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def slerp(val, low, high):
    """Spherical linear interpolation for stochastic codes (from MorDIFF script)."""
    low_norm = low / torch.norm(low, dim=0, keepdim=True)
    high_norm = high / torch.norm(high, dim=0, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum())
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high


def get_random_ffhq_pairs(num_pairs):
    exts = {".png", ".jpg", ".jpeg"}
    all_paths = [p for p in FFHQ_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    random.seed(42)  # For reproducibility
    selected = random.sample(all_paths, num_pairs * 2)
    return [(selected[i], selected[i+1]) for i in range(0, len(selected), 2)]


def main():
    print(f"--- Loading Model & Data ---")
    model = load_diffae_model()
    
    train_mean = torch.from_numpy(np.load(MEAN_PATH).astype(np.float32)).to(DEVICE)
    train_std = torch.from_numpy(np.load(STD_PATH).astype(np.float32)).to(DEVICE)
    
    # NOTE: The *inputs* still require normalization to [-1, 1] because model.encode expects it.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    pairs = get_random_ffhq_pairs(NUM_MORPHS)
    
    results = {
        "raw_mse": [], "raw_cos": [],
        "z1_mse": [], "z1_cos": [],
        "z2_mse": [], "z2_cos": []
    }

    print(f"Generating {NUM_MORPHS} controlled morphs and evaluating...")
    
    for idx, (path1, path2) in enumerate(tqdm(pairs, desc="Processing Morphs")):
        
        # 1. Load and Transform Base Images (Maps [0, 255] -> [-1, 1])
        img1_pil = Image.open(path1).convert('RGB')
        img2_pil = Image.open(path2).convert('RGB')
        
        t_img1 = transform(img1_pil).unsqueeze(0).to(DEVICE)
        t_img2 = transform(img2_pil).unsqueeze(0).to(DEVICE)
        
        # 2. Extract Latents for Generation
        with torch.no_grad():
            z1 = model.encode(t_img1)
            z2 = model.encode(t_img2)
            
            xT1 = model.encode_stochastic(t_img1, z1, T=250)
            xT2 = model.encode_stochastic(t_img2, z2, T=250)
            
        # 3. Perform Mathematical Interpolation (50/50)
        z_intp = 0.5 * z1 + 0.5 * z2
        xT_intp = slerp(0.5, xT1.flatten(), xT2.flatten()).view(xT1.shape)
        
        # 4. Render Morph Image (Output is already in [0, 1])
        with torch.no_grad():
            pred = model.render(xT_intp, z_intp, T=20)
            
        # 5. Save Morph to Disk (Simulate real 8-bit image quantization)
        # FIXED: Removed the `(pred + 1) / 2` shift to prevent white-washing.
        pred_img_tensor = pred[0].cpu().clamp(0, 1) 
        morph_pil = transforms.ToPILImage()(pred_img_tensor)
        
        morph_filename = f"ffhq_morph_{idx:03d}_{path1.stem}_and_{path2.stem}.png"
        morph_save_path = OUTPUT_MORPHS_DIR / morph_filename
        morph_pil.save(morph_save_path)
        
        # 6. Re-load and Re-encode the Saved Morph
        morph_loaded = Image.open(morph_save_path).convert('RGB')
        t_morph = transform(morph_loaded).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            z_morph_reencoded = model.encode(t_morph).squeeze(0)
            
        z1_flat = z1.squeeze(0)
        z2_flat = z2.squeeze(0)
        z_intp_flat = z_intp.squeeze(0)
        
        # 7. Apply Z-Score Normalization
        z_morph_norm = (z_morph_reencoded - train_mean) / train_std
        z1_norm = (z1_flat - train_mean) / train_std
        z2_norm = (z2_flat - train_mean) / train_std
        
        expected_norm_linear = 0.5 * z1_norm + 0.5 * z2_norm
        expected_norm_spherical = np.sqrt(0.5) * z1_norm + np.sqrt(0.5) * z2_norm

        # 8. Compute Metrics
        results["raw_mse"].append(F.mse_loss(z_morph_reencoded, z_intp_flat).item())
        results["raw_cos"].append(F.cosine_similarity(z_morph_reencoded.unsqueeze(0), z_intp_flat.unsqueeze(0)).item())
        
        results["z1_mse"].append(F.mse_loss(z_morph_norm, expected_norm_linear).item())
        results["z1_cos"].append(F.cosine_similarity(z_morph_norm.unsqueeze(0), expected_norm_linear.unsqueeze(0)).item())
        
        results["z2_mse"].append(F.mse_loss(z_morph_norm, expected_norm_spherical).item())
        results["z2_cos"].append(F.cosine_similarity(z_morph_norm.unsqueeze(0), expected_norm_spherical.unsqueeze(0)).item())

    # --- Generate Text Report ---
    report = (
        "=========================================================\n"
        "       CONTROLLED FFHQ-256 MORDIFF LATENT REPORT         \n"
        "=========================================================\n\n"
        f"Total Morphs Generated & Analyzed: {NUM_MORPHS}\n"
        "Dataset Source: FFHQ-256 (Perfectly Aligned)\n\n"
        
        "--- 1. RAW SEMANTIC SPACE (No Normalization) ---\n"
        "Expected: 0.5 * z_1 + 0.5 * z_2\n"
        f"Average MSE:               {np.mean(results['raw_mse']):.6f}\n"
        f"Average Cosine Similarity: {np.mean(results['raw_cos']):.4f}\n\n"
        
        "--- 2. Z-SCORE NORMALIZED SPACE (Linear Average) ---\n"
        "Expected: 0.5 * z'_1 + 0.5 * z'_2\n"
        f"Average MSE:               {np.mean(results['z1_mse']):.6f}\n"
        f"Average Cosine Similarity: {np.mean(results['z1_cos']):.4f}\n\n"
        
        "--- 3. Z-SCORE NORMALIZED SPACE (Variance Preserving) ---\n"
        "Expected: sqrt(0.5) * z'_1 + sqrt(0.5) * z'_2\n"
        f"Average MSE:               {np.mean(results['z2_mse']):.6f}\n"
        f"Average Cosine Similarity: {np.mean(results['z2_cos']):.4f}\n"
        "=========================================================\n"
    )

    with open(REPORT_PATH, "w") as f:
        f.write(report)
        
    print(report)
    print(f"\n[SUCCESS] Report exported to: {REPORT_PATH}")
    print(f"[SUCCESS] {NUM_MORPHS} Morphs saved to: {OUTPUT_MORPHS_DIR}")


if __name__ == "__main__":
    main()
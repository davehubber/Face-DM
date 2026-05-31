import os
import re
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

PATH_TO_DIFFAE = "/nas-ctm01/homes/dacordeiro/diffae/"
CHECKPOINT_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt"

MORDIFF_DIR = Path("/nas-ctm01/homes/dacordeiro/Face-DM/MorDIFF_crop")
FRLL_DIR = Path("/nas-ctm01/homes/dacordeiro/Face-DM/FRLL")

# Z-score normalization files generated from the training split
MEAN_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_train_mean.npy"
STD_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_train_std.npy"
REPORT_PATH = "morph_interpolation_report.txt"

# Make DiffAE importable
sys.path.insert(0, str(PATH_TO_DIFFAE))
old_cwd = os.getcwd()
os.chdir(PATH_TO_DIFFAE)
from templates import ffhq256_autoenc
from experiment import LitModel
os.chdir(old_cwd)


def load_diffae_model():
    """Loads the DiffAE model using the exact logic from the MorDIFF generation script."""
    conf = ffhq256_autoenc()
    # Ensure it only encodes into z_sem, avoiding precomputed latent inferences
    conf.latent_infer_path = None
    
    model = LitModel(conf)
    state = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    
    model.ema_model.eval()
    model.ema_model.to(DEVICE)
    
    for p in model.parameters():
        p.requires_grad_(False)
        
    return model


def process_image(img_path, transform):
    """Loads, converts to RGB, validates size, and transforms the image."""
    img = Image.open(img_path).convert('RGB')
    if img.size[0] != 256 or img.size[1] != 256:
        img = img.resize((256, 256)) # Fallback if slightly off
    return transform(img).unsqueeze(0).to(DEVICE)


def find_base_image(base_name):
    """Searches the FRLL subdirectories for the base jpg image."""
    possible_paths = [
        FRLL_DIR / "neutral_front" / f"{base_name}.jpg",
        FRLL_DIR / "smiling_front" / f"{base_name}.jpg"
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def main():
    print(f"--- Loading Model & Data ---")
    model = load_diffae_model()
    
    # Load Z-score statistics and map to device
    train_mean = torch.from_numpy(np.load(MEAN_PATH).astype(np.float32)).to(DEVICE)
    train_std = torch.from_numpy(np.load(STD_PATH).astype(np.float32)).to(DEVICE)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Find all morph images matching pattern: morphed_XXX_and_YYY.png or morph_XXX_and_YYY.png
    morph_paths = list(MORDIFF_DIR.rglob("*.png"))
    filename_pattern = re.compile(r"(?:morphed?_)?(.*)_and_(.*)\.png")

    results = {
        "raw_mse": [], "raw_cos": [],
        "z1_mse": [], "z1_cos": [],
        "z2_mse": [], "z2_cos": []
    }
    
    missing_bases = 0
    valid_morphs = 0

    print(f"Found {len(morph_paths)} potential morphed images. Analyzing...")
    
    for morph_path in tqdm(morph_paths, desc="Evaluating Morphs"):
        match = filename_pattern.match(morph_path.name)
        if not match:
            continue
            
        base1_name, base2_name = match.groups()
        
        base1_path = find_base_image(base1_name)
        base2_path = find_base_image(base2_name)
        
        if not base1_path or not base2_path:
            missing_bases += 1
            continue

        valid_morphs += 1
        
        # 1. Load and Transform Images
        img_morph = process_image(morph_path, transform)
        img_1 = process_image(base1_path, transform)
        img_2 = process_image(base2_path, transform)
        
        # 2. Encode to get Raw Semantic Codes (z_sem)
        with torch.no_grad():
            z_morph = model.encode(img_morph).squeeze(0)  # [512]
            z_1 = model.encode(img_1).squeeze(0)          # [512]
            z_2 = model.encode(img_2).squeeze(0)          # [512]
            
        # 3. Apply Z-Score Normalization
        z_morph_norm = (z_morph - train_mean) / train_std
        z_1_norm = (z_1 - train_mean) / train_std
        z_2_norm = (z_2 - train_mean) / train_std

        # 4. Calculate Expected Interpolations
        # A) Raw Linear Interpolation
        expected_raw = 0.5 * z_1 + 0.5 * z_2
        
        # B) Normalized Linear Interpolation
        expected_norm_linear = 0.5 * z_1_norm + 0.5 * z_2_norm
        
        # C) Normalized Variance-Preserving (Spherical Approximation) Interpolation
        expected_norm_spherical = np.sqrt(0.5) * z_1_norm + np.sqrt(0.5) * z_2_norm

        # 5. Compute Metrics
        # Raw Space
        results["raw_mse"].append(F.mse_loss(z_morph, expected_raw).item())
        results["raw_cos"].append(F.cosine_similarity(z_morph.unsqueeze(0), expected_raw.unsqueeze(0)).item())
        
        # Z-Scored (Linear)
        results["z1_mse"].append(F.mse_loss(z_morph_norm, expected_norm_linear).item())
        results["z1_cos"].append(F.cosine_similarity(z_morph_norm.unsqueeze(0), expected_norm_linear.unsqueeze(0)).item())
        
        # Z-Scored (Spherical/Variance Preserving)
        results["z2_mse"].append(F.mse_loss(z_morph_norm, expected_norm_spherical).item())
        results["z2_cos"].append(F.cosine_similarity(z_morph_norm.unsqueeze(0), expected_norm_spherical.unsqueeze(0)).item())

    # --- Generate Text Report ---
    if valid_morphs == 0:
        print("\nERROR: No valid morph/base pairs processed. Check paths and regex.")
        return

    report = (
        "=========================================================\n"
        "             MORDIFF LATENT INTERPOLATION REPORT         \n"
        "=========================================================\n\n"
        f"Total Morphs Analyzed: {valid_morphs}\n"
        f"Missing Base Images:   {missing_bases}\n\n"
        
        "--- 1. RAW SEMANTIC SPACE (No Normalization) ---\n"
        "Expected: 0.5 * z_1 + 0.5 * z_2\n"
        f"Average MSE:               {np.mean(results['raw_mse']):.6f}\n"
        f"Average Cosine Similarity: {np.mean(results['raw_cos']):.4f}\n\n"
        
        "--- 2. Z-SCORE NORMALIZED SPACE (Linear Average) ---\n"
        "Expected: 0.5 * z'_1 + 0.5 * z'_2\n"
        f"Average MSE:               {np.mean(results['z1_mse']):.6f}\n"
        f"Average Cosine Similarity: {np.mean(results['z1_cos']):.4f}\n"
        "> THEORETICAL NOTE: Is this directly comparable to Raw Space?\n"
        "> Yes. Because standardizing a linear combination mathematically distributes, the difference\n"
        "> vector in this space is simply the raw difference vector divided feature-wise by the\n"
        "> standard deviation. Consequently, this MSE is a *variance-weighted* version of the raw MSE.\n\n"
        
        "--- 3. Z-SCORE NORMALIZED SPACE (Variance Preserving) ---\n"
        "Expected: sqrt(0.5) * z'_1 + sqrt(0.5) * z'_2\n"
        f"Average MSE:               {np.mean(results['z2_mse']):.6f}\n"
        f"Average Cosine Similarity: {np.mean(results['z2_cos']):.4f}\n"
        "> THEORETICAL NOTE: Because z-scored embeddings have unit variance, interpolating with\n"
        "> 0.5 weights causes the expected vector's variance to collapse to 0.5. Using sqrt(0.5)\n"
        "> preserves the unit variance, representing a more accurate geometric midpoint in the\n"
        "> normalized high-dimensional space.\n"
        "=========================================================\n"
    )

    with open(REPORT_PATH, "w") as f:
        f.write(report)
        
    print(report)
    print(f"\n[SUCCESS] Report exported to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
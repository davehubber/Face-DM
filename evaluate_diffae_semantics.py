import sys
import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import csv
from tqdm import tqdm

def load_diffae_model(diffae_root: Path, device: torch.device):
    """Loads the DiffAE model checkpoint following the official autoencoder configuration."""
    diffae_root = Path(diffae_root).resolve()
    sys.path.insert(0, str(diffae_root))
    
    old_cwd = os.getcwd()
    os.chdir(diffae_root)
    try:
        from templates import ffhq256_autoenc
        from experiment import LitModel

        conf = ffhq256_autoenc()
        model = LitModel(conf)
        checkpoint_path = conf.name / "last.ckpt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")
            
        state = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)
        model.ema_model.eval()
        model.ema_model.to(device)
    finally:
        os.chdir(old_cwd)
        
    return model, conf.img_size

def build_frll_index(frll_root: Path):
    """Indexes all source images inside FRLL subdirectories by their stems for instant lookup."""
    index = {}
    exts = {".png", ".jpg", ".jpeg"}
    subdirs = ["neutral_front", "smiling_front"]
    
    for subdir in subdirs:
        dir_path = frll_root / subdir
        if not dir_path.exists():
            continue
        for p in dir_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                index[p.stem] = p
    return index

def parse_parents(filename: str):
    """Extracts parent image IDs from morph filenames (e.g., morph_034_08_and_126_08 -> 034_08, 126_08)."""
    stem = Path(filename).stem
    if "_and_" not in stem:
        return None, None
        
    parts = stem.split("_and_")
    part1, part2 = parts[0], parts[1]
    
    # Clean out common script prefixes
    for prefix in ["morph_", "morphed_", "comparison_"]:
        if part1.startswith(prefix):
            part1 = part1[len(prefix):]
            
    return part1, part2

def main():
    parser = argparse.ArgumentParser(description="Evaluate Morph Embedding Linearity via Euclidean Metrics")
    parser.add_argument("--mordiff-root", type=str, default="./MorDIFF_crop", help="Path to MorDIFF_crop dataset root")
    parser.add_argument("--frll-root", type=str, default="./FRLL", help="Path to FRLL source folder")
    parser.add_argument("--diffae-root", type=str, default="/nas-ctm01/homes/dacordeiro/diffae/", help="Path to your cloned DiffAE repository")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    mordiff_root = Path(args.mordiff_root)
    frll_root = Path(args.frll_root)
    
    # 1. Load Model & Setup Exact Image Transformation Pipeline
    print("Loading DiffAE model...")
    model, image_size = load_diffae_model(args.diffae_root, device)
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def load_and_prep(path):
        img = Image.open(path).convert('RGB')
        return transform(img).to(device)

    # 2. Map Original Identities
    print("Indexing FRLL reference images...")
    frll_index = build_frll_index(frll_root)
    print(f"Indexed {len(frll_index)} unique reference faces.")

    # 3. Harvest All Morph Images
    print("Locating morph folders inside dataset partitions...")
    morph_paths = sorted(list(mordiff_root.rglob("morphed/*.png")) + list(mordiff_root.rglob("morphed/*.jpg")))
    print(f"Found {len(morph_paths)} morph targets to analyze.")

    # Storage arrays for metric tracking
    l1_distances = []
    mse_distances = []
    cosine_similarities = []
    skipped_count = 0

    # 4. Processing Loop
    for m_path in tqdm(morph_paths, desc="Evaluating Semantic Code Linearity"):
        p1_id, p2_id = parse_parents(m_path.name)
        
        if p1_id not in frll_index or p2_id not in frll_index:
            skipped_count += 1
            continue
            
        p1_path = frll_index[p1_id]
        p2_path = frll_index[p2_id]
        
        with torch.no_grad():
            # Load and process triplets
            img_morph = load_and_prep(m_path)
            img_p1 = load_and_prep(p1_path)
            img_p2 = load_and_prep(p2_path)
            
            # Extract direct semantic embeddings [1, 512]
            z_morph_direct = model.encode(img_morph.unsqueeze(0))
            z_p1 = model.encode(img_p1.unsqueeze(0))
            z_p2 = model.encode(img_p2.unsqueeze(0))
            
            # Generate the true linear midpoint interpolation used by the generator (alpha = 0.5)
            z_interpolated = 0.5 * z_p1 + 0.5 * z_p2
            
            # Compute Euclidean Deviation metrics (Primary)
            l1_dist = F.l1_loss(z_morph_direct, z_interpolated).item()
            mse_dist = F.mse_loss(z_morph_direct, z_interpolated).item()
            
            # Compute Angular Metric (Secondary check)
            cos_sim = F.cosine_similarity(z_morph_direct.flatten(), z_interpolated.flatten(), dim=0).item()
            
            l1_distances.append(l1_dist)
            mse_distances.append(mse_dist)
            cosine_similarities.append(cos_sim)

    if not l1_distances:
        print("Error: No valid pairs processed. Check your directory roots and naming conventions.")
        return

    # 5. Compile Statistical Summary Focused on Euclidean Properties
    report_path = mordiff_root / "morph_semantic_linearity_report.txt"
    
    report = f"""===================================================================
DIFFAE MORPH SEMANTIC CODE LINEARITY REPORT (EUCLIDEAN ALIGNED)
===================================================================
Evaluated Dataset Partition: {mordiff_root.name}
Total Morph Images Located:  {len(morph_paths)}
Successfully Evaluated:      {len(l1_distances)}
Skipped (Missing Parents):   {skipped_count}

This test measures the geometric drift and coordinate deviation between:
  A. The direct semantic encoding of the generated morph image: model.encode(M)
  B. The mathematical midpoint of the parent codes: 0.5 * model.encode(P1) + 0.5 * model.encode(P2)

1. PRIMARY METRICS: EUCLIDEAN COORDINATE DEVIATION
-------------------------------------------------------------------
MEAN ABSOLUTE ERROR (L1 Distance):
  - Mean L1 Distance:       {np.mean(l1_distances):.6f}
  - Standard Deviation L1:   {np.std(l1_distances):.6f}
  - Minimum Observed L1:     {np.min(l1_distances):.6f}
  - Maximum Observed L1:     {np.max(l1_distances):.6f}

L1 Distance Percentiles:
  - 25th Percentile:        {np.percentile(l1_distances, 25):.6f}
  - 50th Percentile (Med):  {np.percentile(l1_distances, 50):.6f}
  - 75th Percentile:        {np.percentile(l1_distances, 75):.6f}

MEAN SQUARED ERROR (MSE / L2 squared Distance):
  - Mean MSE Distance:      {np.mean(mse_distances):.6f}
  - Standard Deviation MSE:  {np.std(mse_distances):.6f}
  - Minimum Observed MSE:    {np.min(mse_distances):.6f}
  - Maximum Observed MSE:    {np.max(mse_distances):.6f}

MSE Distance Percentiles:
  - 25th Percentile:        {np.percentile(mse_distances, 25):.6f}
  - 50th Percentile (Med):  {np.percentile(mse_distances, 50):.6f}
  - 75th Percentile:        {np.percentile(mse_distances, 75):.6f}

2. SECONDARY DIAGNOSTIC: ANGULAR ALIGNMENT (Cosine Similarity)
-------------------------------------------------------------------
  - Mean Cosine Similarity: {np.mean(cosine_similarities):.4f}
  - Standard Deviation:     {np.std(cosine_similarities):.4f}
  - 50th Percentile (Med):  {np.percentile(cosine_similarities, 50):.4f}

3. GEOMETRIC & MATHEMATICAL INTERPRETATION
-------------------------------------------------------------------
- Why Euclidean Metrics Dominate Here: 
  Unlike facial identity vectors (which reside on a hypersphere crust),
  DiffAE semantic codes ($z_sem$) represent a continuous density cloud 
  where vector magnitude directly determines feature scaling and structural 
  intensity (e.g., degree of expression, lighting context). Linear blending 
  travels through the interior of the cluster ellipsoid. Tracking L1 and MSE
  is the only way to accurately evaluate absolute coordinate position preservation.

- Evaluating the Distances:
  A low Mean L1/MSE confirms that the autoencoder can translate pixel-level 
  morph images back into the precise coordinate positions dictated by the linear 
  averaging operation, validating the semantic consistency of the image space.

- Interpreting Cosine Stability: 
  Because linear mixtures cut across the interior of the density cloud rather 
  than tracing a spherical arc, cosine velocity is naturally non-linear here. 
  Use Section 2 purely to monitor if the global orientation experiences severe 
  warping during the T=20 diffusion rendering process.
===================================================================
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\nEvaluation successful. Detailed breakdown written to:\n{report_path}\n")
    print(report)

if __name__ == "__main__":
    main()

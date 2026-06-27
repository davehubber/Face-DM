import csv
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Attempt to import lpips cleanly
try:
    import lpips
except ImportError:
    print("Error: 'lpips' package not found. Please install it via: pip install lpips")
    sys.exit(1)

def run_similarity_test(
    diffae_npy_path: str,
    diffae_metadata_path: str,
    mean_path: str,
    std_path: str,
    out_dir: str,
    total_pairs: int = 100_000_000,
    chunk_size: int = 5_000_000,
    similarity_threshold: float = 0.8
):
    diffae_path = Path(diffae_npy_path).resolve()
    metadata_path = Path(diffae_metadata_path).resolve()
    mean_p = Path(mean_path).resolve()
    std_p = Path(std_path).resolve()
    output_path = Path(out_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for LPIPS: {device}")

    # 1. Load Embeddings and Metadata
    print("Loading master embeddings and normalization statistics...")
    embeddings = np.load(diffae_path).astype(np.float32)
    train_mean = np.load(mean_p).astype(np.float32)
    train_std = np.load(std_p).astype(np.float32)

    print("Reading metadata mapping...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    num_samples = len(embeddings)
    if num_samples != len(metadata_rows):
        raise ValueError("Mismatch between embeddings count and metadata rows!")

    # 2. Z-Score Normalization & Unit Length Scaling
    print("Applying Z-score normalization using training statistics...")
    norm_embeddings = (embeddings - train_mean) / train_std

    print("Pre-computing L2 unit vectors for ultra-fast cosine similarity...")
    row_norms = np.linalg.norm(norm_embeddings, axis=1, keepdims=True)
    row_norms = np.where(row_norms == 0, 1.0, row_norms)  # Zero division guard
    unit_embeddings = norm_embeddings / row_norms

    # 3. Process 100 Million Pairs in Vectorized Chunks
    print(f"\nSampling {total_pairs:,} random pairs...")
    np.random.seed(42)  # For reproducibility
    
    high_sim_pairs = []
    sampled_count = 0
    num_chunks = total_pairs // chunk_size

    for chunk_idx in tqdm(range(num_chunks), desc="Processing Chunks"):
        idx_A = np.random.randint(0, num_samples, size=chunk_size)
        idx_B = np.random.randint(0, num_samples, size=chunk_size)

        # Filter out self-pairs
        valid_mask = idx_A != idx_B
        idx_A = idx_A[valid_mask]
        idx_B = idx_B[valid_mask]
        sampled_count += len(idx_A)

        # Vectorized dot product on unit vectors yields exact cosine similarity
        sims = np.sum(unit_embeddings[idx_A] * unit_embeddings[idx_B], axis=1)

        # Check threshold
        match_mask = sims > similarity_threshold
        if np.any(match_mask):
            matched_A = idx_A[match_mask]
            matched_B = idx_B[match_mask]
            matched_sims = sims[match_mask]

            for a, b, s in zip(matched_A, matched_B, matched_sims):
                high_sim_pairs.append((int(a), int(b), float(s)))

    pct_found = (len(high_sim_pairs) / sampled_count) * 100 if sampled_count > 0 else 0.0
    print(f"\nFound {len(high_sim_pairs)} pairs with cosine similarity > {similarity_threshold} ({pct_found:.6f}%)")

    # 4. Compute Image Metrics for Highly Similar Pairs
    avg_ssim, avg_psnr, avg_lpips = 0.0, 0.0, 0.0

    if len(high_sim_pairs) > 0:
        print("\nInitializing LPIPS model network (AlexNet)...")
        loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

        transform_tensor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        ssim_values, psnr_values, lpips_values = [], [], []

        print("Computing visual metric spaces (SSIM, PSNR, LPIPS) for matched pairs...")
        for idx_A, idx_B, _ in tqdm(high_sim_pairs, desc="Evaluating Images"):
            path_A = Path(metadata_rows[idx_A]['image_path'])
            path_B = Path(metadata_rows[idx_B]['image_path'])

            img_A_pil = Image.open(path_A).convert("RGB")
            img_B_pil = Image.open(path_B).convert("RGB")

            # Transforms for LPIPS [-1, 1]
            tensor_A = transform_tensor(img_A_pil).unsqueeze(0).to(device)
            tensor_B = transform_tensor(img_B_pil).unsqueeze(0).to(device)

            # Numpy conversion for SSIM / PSNR [0, 255]
            np_A = np.array(img_A_pil.resize((256, 256)))
            np_B = np.array(img_B_pil.resize((256, 256)))

            # Metrics calculation
            val_ssim = ssim(np_A, np_B, channel_axis=2)
            val_psnr = psnr(np_A, np_B, data_range=255)
            
            with torch.no_grad():
                val_lpips = loss_fn_lpips(tensor_A, tensor_B).item()

            ssim_values.append(val_ssim)
            psnr_values.append(val_psnr)
            lpips_values.append(val_lpips)

        avg_ssim = np.mean(ssim_values)
        avg_psnr = np.mean(psnr_values)
        avg_lpips = np.mean(lpips_values)

    # 5. Generate and Save TXT Report
    report_path = output_path / "similarity_test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("      DIFFAE Z-SCORE SEMANTIC SIMILARITY REPORT   \n")
        f.write("==================================================\n\n")
        f.write(f"Total Valid Pairs Sampled:       {sampled_count:,}\n")
        f.write(f"Pairs found with CosSim > {similarity_threshold}: {len(high_sim_pairs):,}\n")
        f.write(f"Percentage of total:             {pct_found:.6f} %\n\n")
        f.write("Ground-Truth Visual Metric Averages (Target Pairs Only):\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Average SSIM:                    {avg_ssim:.4f}\n")
        f.write(f"Average PSNR:                    {avg_psnr:.2f} dB\n")
        f.write(f"Average LPIPS (AlexNet):         {avg_lpips:.4f}\n")

    print(f"Report saved successfully to: {report_path}")

    # 6. Export Visualization Grid (Up to 5 Rows)
    num_visualized = min(5, len(high_sim_pairs))
    if num_visualized > 0:
        grid_w, grid_h = 256 * 2, 256 * num_visualized
        grid_img = Image.new('RGB', (grid_w, grid_h))

        for i in range(num_visualized):
            idx_A, idx_B, _ = high_sim_pairs[i]
            img_A = Image.open(metadata_rows[idx_A]['image_path']).convert("RGB").resize((256, 256))
            img_B = Image.open(metadata_rows[idx_B]['image_path']).convert("RGB").resize((256, 256))
            
            grid_img.paste(img_A, (0, i * 256))
            grid_img.paste(img_B, (256, i * 256))

        grid_out_path = output_path / "high_similarity_pairs_grid.png"
        grid_img.save(grid_out_path)
        print(f"Visual validation grid saved to: {grid_out_path}")
    else:
        print("No pairs found above 0.8 similarity; skipping visualization grid generation.")

if __name__ == "__main__":
    BASE_DIR = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings"
    
    run_similarity_test(
        diffae_npy_path=f"{BASE_DIR}/ffhq256_diffae_zsem.npy",
        diffae_metadata_path=f"{BASE_DIR}/ffhq256_diffae_zsem_metadata.csv",
        mean_path=f"{BASE_DIR}/ffhq256_diffae_zsem_train_mean.npy",
        std_path=f"{BASE_DIR}/ffhq256_diffae_zsem_train_std.npy",
        out_dir=f"{BASE_DIR}/similarity_evaluation_results",
        total_pairs=100_000_000,
        chunk_size=5_000_000,
        similarity_threshold=0.8
    )
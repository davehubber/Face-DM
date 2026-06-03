import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.decomposition import PCA
import joblib
from PIL import Image
from datetime import datetime

# Setup paths to your DiffAE repository
PATH_TO_DIFF_MODEL = "/nas-ctm01/homes/dacordeiro/diffae/"
sys.path.append(PATH_TO_DIFF_MODEL)

from templates import ffhq256_autoenc
from experiment import LitModel

def load_img_tensor(path: str, img_size: int = 256) -> torch.Tensor:
    """Loads and preprocesses an image for DiffAE."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0) # Output shape: [1, 3, 256, 256]

def main():
    parser = argparse.ArgumentParser(description="Fit PCA and visually traverse the top 3 components.")
    parser.add_argument("--data-dir", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings")
    parser.add_argument("--prefix", type=str, default="ffhq256_diffae_zsem")
    parser.add_argument("--diffae-ckpt", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt")
    parser.add_argument("--variance-target", type=float, default=0.80)
    parser.add_argument("--jump-factor", type=float, default=0.20, help="Fraction of the min-max range to jump per step.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    prefix = args.prefix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Define file paths
    master_path = data_dir / f"{prefix}.npy"
    train_path = data_dir / f"{prefix}_train.npy"
    mean_path = data_dir / f"{prefix}_train_mean.npy"
    std_path = data_dir / f"{prefix}_train_std.npy"
    meta_path = data_dir / f"{prefix}_metadata.csv"

    # 2. Reconstruct deterministic train split mapping to fetch original images
    print("Loading datasets and reconstructing split indices...")
    master_len = len(np.load(master_path, mmap_mode='r'))
    
    np.random.seed(42)
    split_indices = np.arange(master_len)
    np.random.shuffle(split_indices)
    train_indices = split_indices[:int(master_len * 0.8)]

    idx_to_path = {}
    with open(meta_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_path[int(row['embedding_index'])] = row['image_path']

    # 3. Load and Normalize Train Data
    print("Applying Z-score normalization...")
    X_train = np.load(train_path).astype(np.float32)
    train_mean = np.load(mean_path).astype(np.float32)
    train_std = np.load(std_path).astype(np.float32)
    X_train_norm = (X_train - train_mean) / train_std

    # 4. Perform PCA
    print("Fitting PCA model...")
    pca_full = PCA()
    X_train_pca = pca_full.fit_transform(X_train_norm)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= args.variance_target) + 1
    
    # Truncate PCA to optimal components
    pca_optimal = PCA(n_components=n_components)
    pca_optimal.components_ = pca_full.components_[:n_components]
    pca_optimal.explained_variance_ = pca_full.explained_variance_[:n_components]
    pca_optimal.explained_variance_ratio_ = pca_full.explained_variance_ratio_[:n_components]
    pca_optimal.mean_ = pca_full.mean_

    pca_model_path = data_dir / f"{prefix}_pca_model_{n_components}comp.joblib"
    joblib.dump(pca_optimal, pca_model_path)
    
    print(f"Target variance ({args.variance_target*100:.1f}%) reached at {n_components} components. Model saved.")

    # 5. Initialize DiffAE for Visualization
    print("\nInitializing DiffAE for latent traversal visualizations...")
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load(args.diffae_ckpt, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    # 6. Latent Traversal Loop
    def denormalize(z_norm):
        return z_norm * train_std + train_mean

    for pc_idx in range(3):
        print(f"Processing PC{pc_idx+1}...")
        scores = X_train_pca[:, pc_idx]
        pc_vector = pca_full.components_[pc_idx]
        
        idx_high = np.argmax(scores)
        idx_low = np.argmin(scores)
        
        # Calculate a visible jump size based on the distribution range
        jump = (scores.max() - scores.min()) * args.jump_factor

        # Grab baseline normalized embeddings
        z_norm_high = X_train_norm[idx_high].copy()
        z_norm_low = X_train_norm[idx_low].copy()

        # Sequence creation: High decreases, Low increases
        seq_norm_high = [
            z_norm_high,
            z_norm_high - 1 * jump * pc_vector,
            z_norm_high - 2 * jump * pc_vector
        ]
        seq_norm_low = [
            z_norm_low,
            z_norm_low + 1 * jump * pc_vector,
            z_norm_low + 2 * jump * pc_vector
        ]

        # De-normalize back to original z_sem space
        seq_zsem_high = [denormalize(z) for z in seq_norm_high]
        seq_zsem_low  = [denormalize(z) for z in seq_norm_low]

        # Get original image paths
        path_high = idx_to_path[train_indices[idx_high]]
        path_low = idx_to_path[train_indices[idx_low]]

        grid_images = []
        
        with torch.no_grad():
            # Process HIGH sequence
            img_high = load_img_tensor(path_high, conf.img_size).to(device)
            cond_high = model.encode(img_high)
            xT_high = model.encode_stochastic(img_high, cond_high, T=250)
            
            for z_sem in seq_zsem_high:
                z_tensor = torch.tensor(z_sem, dtype=torch.float32, device=device).unsqueeze(0)
                out = model.render(xT_high, z_tensor, T=20)
                grid_images.append(out.squeeze(0))

            # Process LOW sequence
            img_low = load_img_tensor(path_low, conf.img_size).to(device)
            cond_low = model.encode(img_low)
            xT_low = model.encode_stochastic(img_low, cond_low, T=250)
            
            for z_sem in seq_zsem_low:
                z_tensor = torch.tensor(z_sem, dtype=torch.float32, device=device).unsqueeze(0)
                out = model.render(xT_low, z_tensor, T=20)
                grid_images.append(out.squeeze(0))

        # Save Visual Grid
        grid_out_path = data_dir / f"PC{pc_idx+1}_traversal_grid.png"
        # The outputs are bounded [-1, 1], normalize=True remaps them to [0, 1] for saving
        save_image(grid_images, grid_out_path, nrow=3, normalize=True, value_range=(-1, 1))
        print(f"Saved PC{pc_idx+1} traversal grid to: {grid_out_path.name}")

    # Generate truncated report
    report_path = data_dir / f"{prefix}_pca_report.txt"
    report_lines = [
        "===========================================================",
        "               PCA ANALYSIS REPORT",
        "===========================================================",
        f"Date Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Components Kept   : {n_components} out of 512",
        f"Actual Variance   : {cumulative_variance[n_components-1] * 100:.4f}%",
        "Top 3 Components Variance Explained:"
    ]
    for i in range(3):
        report_lines.append(f"  - PC{i+1}: {pca_full.explained_variance_ratio_[i]*100:.2f}%")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print("\n[SUCCESS] Pipeline Complete.")

if __name__ == "__main__":
    main()

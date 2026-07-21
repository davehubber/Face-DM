import csv
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torchvision
from sklearn.decomposition import PCA

def load_diffae_ffhq256_autoencoder(diffae_root: Path, checkpoint_path: Path, device: torch.device):
    """Loads the official DiffAE FFHQ256 autoencoder model securely."""
    diffae_root = Path(diffae_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    sys.path.insert(0, str(diffae_root))
    old_cwd = os.getcwd()
    os.chdir(diffae_root)

    try:
        from templates import ffhq256_autoenc
        from config import PretrainConfig
        from experiment import LitModel

        conf = ffhq256_autoenc()
        conf.pretrain = PretrainConfig(name="ffhq256_autoenc", path=str(checkpoint_path))
        conf.latent_infer_path = None

        model = LitModel(conf)
        model = model.to(device)
        model.eval()

        if hasattr(model, "ema_model"):
            model.ema_model.eval()

        for p in model.parameters():
            p.requires_grad_(False)
    finally:
        os.chdir(old_cwd)

    return model

def run_pca_pipeline():
    # ----------------------------------------------------
    # 1. Setup Directories and Configurations
    # ----------------------------------------------------
    base_dir = Path("/nas-ctm01/homes/dacordeiro/Face-DM")
    diffae_embeddings_dir = base_dir / "diffae_embeddings"
    out_dir = base_dir / "pca_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    diffae_root = Path("/nas-ctm01/homes/dacordeiro/diffae/")
    checkpoint_path = base_dir / "ffhq256_autoenc/last.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"PCA output directory initialized at: {out_dir}")

    # ----------------------------------------------------
    # 2. Load and Normalize Training Split
    # ----------------------------------------------------
    print("Loading training embeddings and normalization parameters...")
    train_raw = np.load(diffae_embeddings_dir / "ffhq256_diffae_zsem_train.npy")
    train_mean = np.load(diffae_embeddings_dir / "ffhq256_diffae_zsem_train_mean.npy")
    train_std = np.load(diffae_embeddings_dir / "ffhq256_diffae_zsem_train_std.npy")
    
    print("Applying Z-score normalization to training data...")
    train_normalized = (train_raw - train_mean) / train_std

    # ----------------------------------------------------
    # 3. Compute PCA (First 3 Principal Components)
    # ----------------------------------------------------
    print("Fitting PCA on normalized training embeddings...")
    pca = PCA(n_components=3, random_state=42)
    pca.fit(train_normalized)
    
    # Save parameters for future projection functionality
    np.save(out_dir / "pca_components_v.npy", pca.components_)  # Shape: (3, 512)
    np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)
    print("Saved components and variance data successfully.")

    # ----------------------------------------------------
    # 4. Generate Text Report
    # ----------------------------------------------------
    report_path = out_dir / "pca_report.txt"
    total_variance_3pc = np.sum(pca.explained_variance_ratio_)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("      DIFFUSION AUTOENCODER PCA ANALYSIS REPORT   \n")
        f.write("==================================================\n\n")
        f.write(f"Training split sample count: {train_normalized.shape[0]}\n")
        f.write(f"Semantic embedding dimensionality: {train_normalized.shape[1]}\n\n")
        f.write("--- Variance Explained by Top 3 Principal Components ---\n")
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            f.write(f"  * Principal Component {i+1}: {ratio * 100:.4f}%\n")
        f.write(f"\nTotal cumulative variance explained by top 3 PCs: {total_variance_3pc * 100:.4f}%\n")
    print(f"Informative txt report saved to: {report_path}")

    # ----------------------------------------------------
    # 5. Load Unique Test Set and Project
    # ----------------------------------------------------
    print("Loading unique test split data...")
    test_pairs = np.load(diffae_embeddings_dir / "ffhq256_diffae_zsem_test_pairs.npy")
    xt_pairs = np.load(diffae_embeddings_dir / "ffhq256_diffae_xt_test_pairs.npy")
    
    # Extract unique test instances from "Side A" of your paired matrix
    test_raw = test_pairs[:, 0, :]           # Shape: (1000, 512)
    test_xt_raw = xt_pairs[:, 0, :, :, :]    # Shape: (1000, 3, 256, 256)
    
    # Normalize test embeddings using Train stats to avoid data leakage
    test_normalized = (test_raw - train_mean) / train_std
    
    # Project unique test set onto the 3 PCs to acquire individual scores
    test_scores = pca.transform(test_normalized)  # Shape: (1000, 3)

    # ----------------------------------------------------
    # 6. Initialize DiffAE for Latent Traversal Rendering
    # ----------------------------------------------------
    print("Loading DiffAE model for visualization synthesis...")
    model = load_diffae_ffhq256_autoencoder(diffae_root, checkpoint_path, device)

    # ----------------------------------------------------
    # 7. Generate Traversal Grids for Each PC Axis
    # ----------------------------------------------------
    V = pca.components_  # Shape: (3, 512)
    
    for pc_i in range(3):
        print(f"Processing structural traversal for PC {pc_i + 1}...")
        pc_scores = test_scores[:, pc_i]
        
        # Identify extreme embedding indices across the test set
        idx_min = np.argmin(pc_scores)
        idx_max = np.argmax(pc_scores)
        
        # Establish a statistically significant jump interval based on test split dispersion
        score_std = np.std(pc_scores)
        jump_step = 2.5 * score_std  # Clear visual distinction step
        
        # Top Row: Start at Lowest score, apply two consecutive positive shifts
        z_low_col1 = test_normalized[idx_min]
        z_low_col2 = test_normalized[idx_min] + jump_step * V[pc_i]
        z_low_col3 = test_normalized[idx_min] + (2 * jump_step) * V[pc_i]
        
        # Bottom Row: Start at Highest score, apply two consecutive negative shifts
        z_high_col1 = test_normalized[idx_max]
        z_high_col2 = test_normalized[idx_max] - jump_step * V[pc_i]
        z_high_col3 = test_normalized[idx_max] - (2 * jump_step) * V[pc_i]
        
        # Stack rows together
        stacked_z_normalized = np.stack([
            z_low_col1, z_low_col2, z_low_col3,
            z_high_col1, z_high_col2, z_high_col3
        ], axis=0)
        
        # De-normalize back to raw semantic space for DiffAE rendering input
        stacked_z_raw = (stacked_z_normalized * train_std) + train_mean
        
        # Map appropriate spatial structures (x_T) to their respective rows
        xt_low = test_xt_raw[idx_min]
        xt_high = test_xt_raw[idx_max]
        stacked_xt = np.stack([xt_low, xt_low, xt_low, xt_high, xt_high, xt_high], axis=0)
        
        # Transfer tensors to operational device execution context
        torch_z = torch.from_numpy(stacked_z_raw).to(device).float()
        torch_xt = torch.from_numpy(stacked_xt).to(device).float()
        
        # Render the full canvas via DiffAE fast reverse-diffusion sampling
        with torch.no_grad():
            rendered_images = model.render(torch_xt, torch_z, T=20)  # Shape: (6, 3, 256, 256)
            
        # Compile images into a scannable 2-row, 3-column structural layout grid
        grid_path = out_dir / f"pc{pc_i + 1}_traversal_grid.png"
        torchvision.utils.save_image(
            rendered_images, 
            grid_path, 
            nrow=3, 
            normalize=False,
        )
        print(f"Successfully saved rendering visualization grid to: {grid_path}")

    print("\n[SUCCESS] PCA pipeline execution finished entirely. All outputs are located in 'pca_analysis/'.")

if __name__ == "__main__":
    run_pca_pipeline()

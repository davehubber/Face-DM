import argparse
import csv
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.decomposition import PCA
import joblib

def main():
    parser = argparse.ArgumentParser(description="Fit PCA and calculate traversal vectors.")
    parser.add_argument("--data-dir", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings")
    parser.add_argument("--prefix", type=str, default="ffhq256_diffae_zsem")
    parser.add_argument("--variance-target", type=float, default=0.80)
    parser.add_argument("--jump-factor", type=float, default=0.20, help="Fraction of the min-max range to jump per step.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    prefix = args.prefix

    # 1. Define file paths
    master_path = data_dir / f"{prefix}.npy"
    train_path = data_dir / f"{prefix}_train.npy"
    mean_path = data_dir / f"{prefix}_train_mean.npy"
    std_path = data_dir / f"{prefix}_train_std.npy"
    meta_path = data_dir / f"{prefix}_metadata.csv"

    # 2. Reconstruct deterministic train split mapping
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
    
    # Save the PCA model (Optional, but good for record keeping)
    pca_optimal = PCA(n_components=n_components)
    pca_optimal.components_ = pca_full.components_[:n_components]
    pca_optimal.explained_variance_ = pca_full.explained_variance_[:n_components]
    pca_optimal.explained_variance_ratio_ = pca_full.explained_variance_ratio_[:n_components]
    pca_optimal.mean_ = pca_full.mean_

    pca_model_path = data_dir / f"{prefix}_pca_model_{n_components}comp.joblib"
    joblib.dump(pca_optimal, pca_model_path)
    print(f"Target variance ({args.variance_target*100:.1f}%) reached at {n_components} components. Model saved.")

    # 5. Calculate Latent Traversals
    def denormalize(z_norm):
        return z_norm * train_std + train_mean

    traversal_data = {}

    for pc_idx in range(3):
        print(f"Calculating sequences for PC{pc_idx+1}...")
        scores = X_train_pca[:, pc_idx]
        pc_vector = pca_full.components_[pc_idx]
        
        idx_high = np.argmax(scores)
        idx_low = np.argmin(scores)
        
        jump = (scores.max() - scores.min()) * args.jump_factor

        z_norm_high = X_train_norm[idx_high].copy()
        z_norm_low = X_train_norm[idx_low].copy()

        seq_norm_high = [z_norm_high, z_norm_high - 1 * jump * pc_vector, z_norm_high - 2 * jump * pc_vector]
        seq_norm_low = [z_norm_low, z_norm_low + 1 * jump * pc_vector, z_norm_low + 2 * jump * pc_vector]

        # Save data into a dictionary for handing off to Script 2
        traversal_data[f"PC{pc_idx+1}"] = {
            "path_high": idx_to_path[train_indices[idx_high]],
            "path_low": idx_to_path[train_indices[idx_low]],
            "seq_zsem_high": np.array([denormalize(z) for z in seq_norm_high]),
            "seq_zsem_low": np.array([denormalize(z) for z in seq_norm_low])
        }

    # Save Handoff File
    handoff_path = data_dir / f"{prefix}_traversal_data.pkl"
    with open(handoff_path, "wb") as f:
        pickle.dump(traversal_data, f)
    print(f"Traversal vectors saved to: {handoff_path.name}")

    # 6. Generate Report
    report_path = data_dir / f"{prefix}_pca_report.txt"
    report_lines = [
        "===========================================================",
        "               PCA ANALYSIS REPORT",
        "===========================================================",
        f"Date Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Components Kept   : {n_components} out of X_train.shape[1]",
        f"Actual Variance   : {cumulative_variance[n_components-1] * 100:.4f}%",
        "Top 3 Components Variance Explained:"
    ]
    for i in range(3):
        report_lines.append(f"  - PC{i+1}: {pca_full.explained_variance_ratio_[i]*100:.2f}%")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print("\n[SUCCESS] Phase 1: Math and PCA Complete.")

if __name__ == "__main__":
    main()

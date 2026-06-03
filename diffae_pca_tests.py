import argparse
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import joblib
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Perform PCA on DiffAE embeddings and generate a report.")
    parser.add_argument("--data-dir", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings", help="Directory containing the split embeddings and stats.")
    parser.add_argument("--prefix", type=str, default="ffhq256_diffae_zsem", help="Base prefix of the saved .npy files.")
    parser.add_argument("--variance-target", type=float, default=0.80, help="Target cumulative variance to explain (e.g., 0.80 for 80%).")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir).resolve()
    prefix = args.prefix
    target_var = args.variance_target
    
    # 1. Define file paths
    train_path = data_dir / f"{prefix}_train.npy"
    mean_path = data_dir / f"{prefix}_train_mean.npy"
    std_path = data_dir / f"{prefix}_train_std.npy"
    
    # 2. Load the data
    print(f"Loading training data from: {train_path}")
    X_train = np.load(train_path).astype(np.float32)
    train_mean = np.load(mean_path).astype(np.float32)
    train_std = np.load(std_path).astype(np.float32)
    
    # 3. Apply Z-Score Normalization
    print("Applying Z-score normalization...")
    # Using the precomputed stats. The compute_splits.py already safeguards against std=0
    X_train_norm = (X_train - train_mean) / train_std
    
    # 4. Perform PCA
    print("Fitting PCA model...")
    # We fit a full PCA first to analyze the full variance spectrum
    pca_full = PCA()
    pca_full.fit(X_train_norm)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Find the number of components needed to reach the target variance
    # np.argmax returns the first index where the condition is true
    n_components = np.argmax(cumulative_variance >= target_var) + 1
    actual_variance_explained = cumulative_variance[n_components - 1]
    
    print(f"Target variance ({target_var*100:.1f}%) reached at {n_components} components.")
    
    # 5. Transform the data using the optimal components
    # We slice the already fitted components to save computation time
    pca_optimal = PCA(n_components=n_components)
    pca_optimal.components_ = pca_full.components_[:n_components]
    pca_optimal.explained_variance_ = pca_full.explained_variance_[:n_components]
    pca_optimal.explained_variance_ratio_ = pca_full.explained_variance_ratio_[:n_components]
    pca_optimal.mean_ = pca_full.mean_
    
    print("Transforming training embeddings...")
    X_train_pca = pca_optimal.transform(X_train_norm)
    
    # 6. Save outputs
    report_path = data_dir / f"{prefix}_pca_report.txt"
    pca_model_path = data_dir / f"{prefix}_pca_model.joblib"
    pca_embs_path = data_dir / f"{prefix}_train_pca_{n_components}comp.npy"
    
    np.save(pca_embs_path, X_train_pca)
    joblib.dump(pca_optimal, pca_model_path)
    
    # 7. Generate Text Report
    report_lines = [
        "===========================================================",
        "               PCA ANALYSIS REPORT",
        "===========================================================",
        f"Date Generated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset Prefix    : {prefix}",
        f"Original Shape    : {X_train.shape} (Samples, Features)",
        "Normalization     : Z-Score (using training mean and std)",
        "-----------------------------------------------------------",
        f"Target Variance   : {target_var * 100:.2f}%",
        f"Components Kept   : {n_components} out of {X_train.shape[1]}",
        f"Actual Variance   : {actual_variance_explained * 100:.4f}%",
        f"New Data Shape    : {X_train_pca.shape}",
        "-----------------------------------------------------------",
        "Top 10 Components by Variance Explained:",
    ]
    
    for i in range(min(10, n_components)):
        var = pca_full.explained_variance_ratio_[i] * 100
        cum_var = cumulative_variance[i] * 100
        report_lines.append(f"  - PC{i+1:<3}: {var:05.2f}% (Cumulative: {cum_var:05.2f}%)")
        
    report_lines.extend([
        "-----------------------------------------------------------",
        "Saved Artifacts:",
        f"  - Reduced Embs  : {pca_embs_path.name}",
        f"  - PCA Model     : {pca_model_path.name}",
        "==========================================================="
    ])
    
    report_content = "\n".join(report_lines)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"\n[SUCCESS] PCA Pipeline complete.")
    print(f"Report saved to: {report_path}")
    print(report_content)

if __name__ == "__main__":
    main()
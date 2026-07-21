import csv
import os
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

def run_arcface_pca_pipeline():
    # ----------------------------------------------------
    # 1. Setup Directories and Configurations
    # ----------------------------------------------------
    base_dir = Path("/nas-ctm01/homes/dacordeiro/Face-DM")
    arcface_embeddings_dir = Path("/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM")
    out_dir = base_dir / "pca_analysis_arcface"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"ArcFace PCA output directory initialized at: {out_dir}")

    # ----------------------------------------------------
    # 2. Load Training Split
    # ----------------------------------------------------
    print("Loading ArcFace training embeddings...")
    train_raw = np.load(arcface_embeddings_dir / "ffhq256_deepface_arcface_retinaface_l2norm_train.npy")
    
    print(f"Loaded training shape: {train_raw.shape}")
    print("Note: Embeddings are already L2-normalized; fitting PCA directly on raw data.")

    # ----------------------------------------------------
    # 3. Compute PCA (First 3 Principal Components)
    # ----------------------------------------------------
    print("Fitting PCA on ArcFace training embeddings...")
    pca = PCA(n_components=3, random_state=42)
    pca.fit(train_raw)
    
    # Save parameters for future projection functionality
    np.save(out_dir / "arcface_pca_components_v.npy", pca.components_)  # Shape: (3, 512)
    np.save(out_dir / "arcface_pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)
    print("Saved components and variance data successfully.")

    # ----------------------------------------------------
    # 4. Generate Text Report
    # ----------------------------------------------------
    report_path = out_dir / "arcface_pca_report.txt"
    total_variance_3pc = np.sum(pca.explained_variance_ratio_)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("         ARCFACE EMBEDDINGS PCA ANALYSIS REPORT   \n")
        f.write("==================================================\n\n")
        f.write(f"Training split sample count: {train_raw.shape[0]}\n")
        f.write(f"ArcFace embedding dimensionality: {train_raw.shape[1]}\n\n")
        f.write("--- Variance Explained by Top 3 Principal Components ---\n")
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            f.write(f"  * Principal Component {i+1}: {ratio * 100:.4f}%\n")
        f.write(f"\nTotal cumulative variance explained by top 3 PCs: {total_variance_3pc * 100:.4f}%\n")
    print(f"Informative txt report saved to: {report_path}")

    # ----------------------------------------------------
    # 5. Load Unique Test Set and Project
    # ----------------------------------------------------
    print("Loading unique test split data...")
    test_pairs = np.load(arcface_embeddings_dir / "ffhq256_deepface_arcface_retinaface_l2norm_test_pairs.npy")
    
    # Extract unique test instances from "Side A" of your paired matrix
    test_raw = test_pairs[:, 0, :]           # Shape: (1000, 512)
    
    # Project unique test set onto the 3 PCs to acquire individual scores
    print("Projecting test embeddings onto ArcFace PC axes...")
    test_scores = pca.transform(test_raw)  # Shape: (1000, 3)
    
    np.save(out_dir / "arcface_test_scores.npy", test_scores)
    print(f"Saved projected test scores successfully to: {out_dir / 'arcface_test_scores.npy'}")

    print("\n[SUCCESS] ArcFace PCA pipeline execution finished entirely. All outputs are located in 'pca_analysis_arcface/'.")

if __name__ == "__main__":
    run_arcface_pca_pipeline()

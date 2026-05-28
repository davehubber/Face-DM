import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

def main():
    # Setup paths based on your environment
    out_dir = Path("/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings")
    emb_path = out_dir / "ffhq256_diffae_zsem.npy"
    report_path = out_dir / "pca_variance_report.txt"

    if not emb_path.exists():
        print(f"Error: Could not find embedding file at {emb_path}")
        return

    print(f"Loading raw embeddings from {emb_path}...")
    X = np.load(emb_path).astype(np.float32)
    n_samples, n_features = X.shape
    print(f"Loaded array: {n_samples} samples x {n_features} dimensions.")

    # We fit more components (e.g., 30) to show the context of the decay tail
    n_comps_to_fit = min(30, n_features)
    print(f"Fitting PCA with {n_comps_to_fit} components (Raw data, auto-centered)...")
    pca = PCA(n_components=n_comps_to_fit, random_state=42)
    pca.fit(X)

    # Extract variance metrics
    exp_var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(exp_var_ratio)
    
    v1, v2, v3 = exp_var_ratio[0], exp_var_ratio[1], exp_var_ratio[2]

    # Relative dominance comparisons
    pc1_vs_pc2 = v1 / v2 if v2 > 0 else 0
    pc2_vs_pc3 = v2 / v3 if v3 > 0 else 0
    pc1_vs_pc3 = v1 / v3 if v3 > 0 else 0

    # Build an informative, structured text report
    report = f"""==================================================
PRINCIPAL COMPONENT ANALYSIS (PCA) REPORT
==================================================
Dataset:          {emb_path.name}
Total Samples:    {n_samples} 
Total Dimensions: {n_features} (512-D Semantic Space)

1. THE TOP 3 PRINCIPAL COMPONENTS: INDIVIDUAL IMPACT
--------------------------------------------------
PC1 (Component 1):
  - Explained Variance Ratio: {v1 * 100:.2f}%
  - Singular Value (Energy):  {pca.singular_values_[0]:.2f}
  - Role: The absolute dominant axis of semantic variation in the dataset.

PC2 (Component 2):
  - Explained Variance Ratio: {v2 * 100:.2f}%
  - Singular Value (Energy):  {pca.singular_values_[1]:.2f}
  - Role: The second largest orthogonal axis of variation.

PC3 (Component 3):
  - Explained Variance Ratio: {v3 * 100:.2f}%
  - Singular Value (Energy):  {pca.singular_values_[2]:.2f}
  - Role: The third largest orthogonal axis of variation.

Combined Top 3 Impact: The first 3 components alone capture {(v1+v2+v3)*100:.2f}% 
of the total variance present across all {n_features} dimensions.

2. RELATIVE DOMINANCE COMPARISON
--------------------------------------------------
To understand how much more "impactful" the leading components are 
compared to each other:

  - PC1 is {pc1_vs_pc2:.2f}x more impactful than PC2.
  - PC2 is {pc2_vs_pc3:.2f}x more impactful than PC3.
  - PC1 is {pc1_vs_pc3:.2f}x more impactful than PC3.

A high multiplier between PC1 and PC2 indicates a highly directional 
"stretched" data distribution, whereas multipliers close to 1.0 indicate 
a more symmetrical, evenly distributed cluster.

3. VARIANCE DECAY PROFILE (TOP {n_comps_to_fit} COMPONENTS)
--------------------------------------------------
Rank   Individual Var %   Cumulative Var %
--------------------------------------------------
"""
    for i in range(n_comps_to_fit):
        report += f"PC{i+1:<4}   {exp_var_ratio[i]*100:<16.2f}   {cum_var_ratio[i]*100:.2f}%\n"

    report += f"""--------------------------------------------------
4. MATHEMATICAL & GEOMETRIC INTERPRETATION
--------------------------------------------------
- High Individual PC1 Variance: If PC1 explains a massive percentage 
  (e.g., >15-20%), the dataset's facial semantic attributes are heavily 
  polarized along a single macro feature (e.g., an underlying global trait 
  like lighting bias, gender, or head rotation).
  
- Sharp vs. Gradual Decay: A sharp drop-off from PC1 to PC3 followed by 
  a long, flat tail indicates that a tiny subset of dimensions carries 
  the core clean information, while the rest represent low-amplitude 
  structural variances or encoding noise.
==================================================
"""

    # Save to directory
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nPCA Analysis complete. Report successfully saved to:\n{report_path}\n")
    print(report)

if __name__ == "__main__":
    main()

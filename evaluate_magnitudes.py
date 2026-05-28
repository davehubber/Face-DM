import numpy as np
from pathlib import Path

def main():
    # Paths configured based on your encode_diffae.py defaults
    out_dir = Path("/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings")
    emb_path = out_dir / "ffhq256_diffae_zsem.npy"
    report_path = out_dir / "embedding_magnitude_report.txt"

    if not emb_path.exists():
        print(f"Error: Could not find embedding file at {emb_path}")
        return

    print(f"Loading embeddings from {emb_path}...")
    embeddings = np.load(emb_path)
    num_embeddings = len(embeddings)
    print(f"Loaded {num_embeddings} embeddings with shape {embeddings.shape}")

    # Configuration
    num_pairs = 1000
    np.random.seed(42)  # For reproducibility

    # Sample random indices for pairs
    idx1 = np.random.choice(num_embeddings, size=num_pairs, replace=True)
    idx2 = np.random.choice(num_embeddings, size=num_pairs, replace=True)

    # Prevent pairing an embedding with itself
    for i in range(num_pairs):
        while idx1[i] == idx2[i]:
            idx2[i] = np.random.choice(num_embeddings)

    # Metrics storage
    orig_norms = []
    avg_norms = []
    
    closer_to_z1 = 0
    closer_to_z2 = 0
    exact_ties = 0

    diffs_to_z1 = []
    diffs_to_z2 = []

    print(f"Evaluating {num_pairs} random pairs...")
    for i in range(num_pairs):
        z1 = embeddings[idx1[i]]
        z2 = embeddings[idx2[i]]

        # Compute the averaged embedding
        z_avg = (z1 + z2) / 2.0

        # Calculate L2 magnitudes
        m1 = np.linalg.norm(z1)
        m2 = np.linalg.norm(z2)
        m_avg = np.linalg.norm(z_avg)

        orig_norms.extend([m1, m2])
        avg_norms.append(m_avg)

        # Absolute difference in magnitude
        d1 = abs(m_avg - m1)
        d2 = abs(m_avg - m2)

        diffs_to_z1.append(d1)
        diffs_to_z2.append(d2)

        if d1 < d2:
            closer_to_z1 += 1
        elif d2 < d1:
            closer_to_z2 += 1
        else:
            exact_ties += 1

    # Statistical Aggregations
    mean_orig = np.mean(orig_norms)
    std_orig = np.std(orig_norms)
    mean_avg = np.mean(avg_norms)
    std_avg = np.std(avg_norms)
    
    norm_reduction_pct = ((mean_orig - mean_avg) / mean_orig) * 100

    # Build the structured TXT report
    report = f"""==================================================
DIFFAE EMBEDDING MAGNITUDE EVALUATION REPORT
==================================================
Source File:     {emb_path.name}
Total Dataset:   {num_embeddings} samples
Pairs Tested:    {num_pairs} (Randomly Sampled)

1. MAGNITUDE DISTRIBUTION STATISTICS (L2 Norm)
--------------------------------------------------
Original Embeddings (z1, z2):
  - Mean Magnitude:       {mean_orig:.4f}
  - Standard Deviation:   {std_orig:.4f}

Averaged Embeddings (z_avg = (z1 + z2) / 2):
  - Mean Magnitude:       {mean_avg:.4f}
  - Standard Deviation:   {std_avg:.4f}
  - Avg. Norm Reduction:  {norm_reduction_pct:.2f}%

2. PROXIMITY ANALYSIS
--------------------------------------------------
Which original magnitude is the average closer to?
  - Closer to z1:         {closer_to_z1} pairs ({(closer_to_z1/num_pairs)*100:.2f}%)
  - Closer to z2:         {closer_to_z2} pairs ({(closer_to_z2/num_pairs)*100:.2f}%)
  - Exact Ties:           {exact_ties} pairs ({(exact_ties/num_pairs)*100:.2f}%)

Average Absolute Distance to Original Magnitudes:
  - Mean | |z_avg| - |z1| |: {np.mean(diffs_to_z1):.4f}
  - Mean | |z_avg| - |z2| |: {np.mean(diffs_to_z2):.4f}

3. GEOMETRIC INSIGHT
--------------------------------------------------
Because these 512-D embeddings are z-score normalized, their 
expected baseline L2 norm hovers around sqrt(512) (~22.62). 
When averaging two random vectors in high-dimensional spaces, 
orthogonality tendencies usually cause the resulting vector's 
magnitude to shrink towards the origin. This report details 
whether that shrinkage favors one target's baseline scale over 
the other due to sampling variance.
==================================================
"""

    # Save to directory
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport successfully generated and saved to:\n{report_path}\n")
    print(report)

if __name__ == "__main__":
    main()
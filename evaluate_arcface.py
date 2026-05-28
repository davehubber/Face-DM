import numpy as np
from pathlib import Path

def main():
    # Paths based on your encode_arcface.py setup
    out_dir = Path("/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/")
    emb_path = out_dir / "ffhq256_deepface_arcface_retinaface_l2norm.npy"
    report_path = out_dir / "arcface_similarity_report.txt"

    if not emb_path.exists():
        print(f"Error: Could not find embedding file at {emb_path}")
        return

    print(f"Loading L2-normalized ArcFace embeddings from {emb_path}...")
    embeddings = np.load(emb_path)
    num_embeddings = len(embeddings)
    print(f"Loaded {num_embeddings} embeddings with shape {embeddings.shape}")

    # Configuration
    num_pairs = 1000
    np.random.seed(42)  # For reproducibility

    # Sample random indices for pairs
    idx1 = np.random.choice(num_embeddings, size=num_pairs, replace=True)
    idx2 = np.random.choice(num_embeddings, size=num_pairs, replace=True)

    # Prevent pairing an image with itself
    for i in range(num_pairs):
        while idx1[i] == idx2[i]:
            idx2[i] = np.random.choice(num_embeddings)

    similarities = []

    print(f"Evaluating cosine similarity for {num_pairs} random pairs...")
    for i in range(num_pairs):
        z1 = embeddings[idx1[i]]
        z2 = embeddings[idx2[i]]

        # Cosine similarity of L2-normalized vectors is just the dot product
        cos_sim = np.dot(z1, z2)
        similarities.append(cos_sim)

    similarities = np.array(similarities)

    # Statistical Calculations
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    
    # Quantiles
    q25 = np.percentile(similarities, 25)
    median = np.percentile(similarities, 50)
    q75 = np.percentile(similarities, 75)

    # Count pairs that cross typical identity verification thresholds
    # (ArcFace verification thresholds usually sit between 0.35 and 0.45)
    threshold_030 = np.sum(similarities > 0.30)
    threshold_040 = np.sum(similarities > 0.40)
    threshold_050 = np.sum(similarities > 0.50)

    # Build the structured TXT report
    report = f"""==================================================
ARCFACE EMBEDDING COSINE SIMILARITY REPORT
==================================================
Source File:     {emb_path.name}
Total Dataset:   {num_embeddings} samples
Pairs Tested:    {num_pairs} (Randomly Sampled)
Metric Used:     Cosine Similarity (Dot Product on L2-Norm)

1. COSINE SIMILARITY STATISTICS
--------------------------------------------------
  - Mean Similarity:     {mean_sim:.4f}
  - Standard Deviation:  {std_sim:.4f}
  - Minimum Similarity:  {min_sim:.4f}
  - Maximum Similarity:  {max_sim:.4f}

Distribution Percentiles:
  - 25th Percentile:     {q25:.4f}
  - 50th (Median):       {median:.4f}
  - 75th Percentile:     {q75:.4f}

2. FALSE POSITIVE / SIMILARITY BOUNDS ANALYSIS
--------------------------------------------------
ArcFace groups identical individuals at high similarities and separates
different identities near or below a baseline zero-bias. 

Pairs exceeding target similarity thresholds:
  - Pairs > 0.30:        {threshold_030} / {num_pairs} ({(threshold_030/num_pairs)*100:.2f}%)
  - Pairs > 0.40:        {threshold_040} / {num_pairs} ({(threshold_040/num_pairs)*100:.2f}%)
  - Pairs > 0.50:        {threshold_050} / {num_pairs} ({(threshold_050/num_pairs)*100:.2f}%)

3. INTERPRETATION GUIDE
--------------------------------------------------
- Expected Baseline: Because FFHQ consists of different individuals, the
  mean cosine similarity should hover relatively low (often between 0.00 
  and 0.20 depending on global face alignments and background leakage).
  
- Lookalike Rate: Pairs crossing > 0.40 represent random combinations 
  that the model considers strongly lookalike or potentially the same 
  identity under poor environmental conditions. 

- Comparison to DiffAE: Unlike your DiffAE semantic space (which measures 
  overall scene and pixel-structural variance), ArcFace discards non-identity
  factors (lighting, pose, background) to focus purely on facial geometry.
==================================================
"""

    # Save to directory
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nEvaluation complete. Report successfully saved to:\n{report_path}\n")
    print(report)

if __name__ == "__main__":
    main()

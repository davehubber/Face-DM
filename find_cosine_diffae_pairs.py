import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

def analyze_cosine_extremes(base_path_str: str, split: str = "train", num_pairs: int = 1_000_000, batch_size: int = 100_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Resolve paths for data and precomputed train split metrics
    base_path = Path(base_path_str).resolve()
    parent = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_val", "").replace("_test", "")
    
    split_path = parent / f"{stem}_{split}.npy"
    mean_path = parent / f"{stem}_train_mean.npy"
    std_path = parent / f"{stem}_train_std.npy"
    
    # 2. Load and Z-score normalize the selected split
    print(f"Loading {split} split from: {split_path}")
    data = np.load(split_path).astype(np.float32)
    
    if mean_path.exists() and std_path.exists():
        print("Applying training z-score normalization...")
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        data = (data - mean) / std
    else:
        raise FileNotFoundError("Could not find train mean or std files to normalize the vectors.")

    # Move full array to torch tensor
    embeddings = torch.from_numpy(data).to(device)
    num_samples = embeddings.shape[0]
    print(f"Loaded dataset pool size: {num_samples} embeddings")

    # 3. Generate random pairs tracking non-zero offsets (idx1 != idx2 guaranteed)
    print(f"Generating {num_pairs:,} random valid pairs...")
    np.random.seed(42)  # Replicable generation
    idx1_all = np.random.randint(0, num_samples, size=num_pairs)
    offsets = np.random.randint(1, num_samples, size=num_pairs)
    idx2_all = (idx1_all + offsets) % num_samples

    # 4. Process in batches to evaluate metrics safely
    high_sim_count = 0
    low_sim_count = 0

    print("Analyzing similarity scores...")
    for i in range(0, num_pairs, batch_size):
        end_idx = min(i + batch_size, num_pairs)
        
        # Slices for the current batch
        batch_idx1 = idx1_all[i:end_idx]
        batch_idx2 = idx2_all[i:end_idx]
        
        # Gather vectors onto device
        z1 = embeddings[batch_idx1]
        z2 = embeddings[batch_idx2]
        
        # Calculate row-wise cosine similarity
        sims = F.cosine_similarity(z1, z2, dim=-1)
        
        # Threshold checks
        high_sim_count += (sims >= 0.8).sum().item()
        low_sim_count += (sims <= -0.35).sum().item()

    # 5. Report final metrics
    print("\n" + "="*45)
    print(f"--- Analysis Summary ({split.upper()} split) ---")
    print(f"Total random pairs evaluated: {num_pairs:,}")
    print(f"Pairs with Cosine Similarity >=  0.8: {high_sim_count:,} ({high_sim_count / num_pairs * 100:.4f}%)")
    print(f"Pairs with Cosine Similarity <= -0.35: {low_sim_count:,} ({low_sim_count / num_pairs * 100:.4f}%)")
    print("="*45)

if __name__ == "__main__":
    # Point this to your master/split folder location
    BASE_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    
    # Analyze the training split distribution
    analyze_cosine_extremes(base_path_str=BASE_PATH, split="train", num_pairs=100_000_000)

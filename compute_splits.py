import numpy as np
from pathlib import Path

def joint_split_and_extract_stats(diffae_npy_path: str, arcface_npy_path: str):
    """
    Jointly splits DiffAE and ArcFace master embedding .npy files into 80/10/10 Train/Val/Test files
    using the exact same randomized indices to preserve perfect alignment.
    Saves the splits in their respective parent directories.
    Calculates the mean and standard deviation STRICTLY from the DiffAE training split.
    """
    # 1. Resolve paths and directories
    diffae_path = Path(diffae_npy_path).resolve()
    arcface_path = Path(arcface_npy_path).resolve()
    
    if not diffae_path.exists():
        raise FileNotFoundError(f"DiffAE master file not found at: {diffae_path}")
    if not arcface_path.exists():
        raise FileNotFoundError(f"ArcFace master file not found at: {arcface_path}")
        
    diffae_out_dir = diffae_path.parent
    arcface_out_dir = arcface_path.parent
    
    # 2. Load Master Embeddings
    print(f"Loading DiffAE embeddings from: {diffae_path}")
    diffae_embs = np.load(diffae_path).astype(np.float32)
    
    print(f"Loading ArcFace embeddings from: {arcface_path}")
    arcface_embs = np.load(arcface_path).astype(np.float32)
    
    # 3. Critical Alignment Check
    num_samples = len(diffae_embs)
    if num_samples != len(arcface_embs):
        raise ValueError(
            f"DATASET MISMATCH FAULT: DiffAE has {num_samples} samples, "
            f"but ArcFace has {len(arcface_embs)} samples. They must be identical."
        )
    print(f"\nAlignment verified. Total paired samples: {num_samples}")
    
    # 4. Generate Single Shuffled Index
    np.random.seed(42)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_end = int(num_samples * 0.8)
    val_end = int(num_samples * 0.9)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    print(f"\nSplit distribution metrics:")
    print(f"  - Train Split (80%): {len(train_idx)} vectors")
    print(f"  - Val Split   (10%): {len(val_idx)} vectors")
    print(f"  - Test Split  (10%): {len(test_idx)} vectors")
    
    # 5. Apply Indices to Both Datasets and Export
    
    # --- DiffAE Processing ---
    print(f"\nSaving DiffAE splits to: {diffae_out_dir}")
    diffae_train = diffae_embs[train_idx]
    
    np.save(diffae_out_dir / f"{diffae_path.stem}_train.npy", diffae_train)
    np.save(diffae_out_dir / f"{diffae_path.stem}_val.npy", diffae_embs[val_idx])
    np.save(diffae_out_dir / f"{diffae_path.stem}_test.npy", diffae_embs[test_idx])
    
    # --- ArcFace Processing ---
    print(f"Saving ArcFace splits to: {arcface_out_dir}")
    np.save(arcface_out_dir / f"{arcface_path.stem}_train.npy", arcface_embs[train_idx])
    np.save(arcface_out_dir / f"{arcface_path.stem}_val.npy", arcface_embs[val_idx])
    np.save(arcface_out_dir / f"{arcface_path.stem}_test.npy", arcface_embs[test_idx])
    
    # 6. Compute DiffAE Z-Score Statistics (Train Only)
    print("\nExtracting feature-wise channel metrics from DiffAE Train Split...")
    train_mean = np.mean(diffae_train, axis=0)
    train_std = np.std(diffae_train, axis=0)
    
    # Prevent division by zero during normalization
    train_std = np.where(train_std == 0, 1.0, train_std)
    
    mean_path = diffae_out_dir / f"{diffae_path.stem}_train_mean.npy"
    std_path = diffae_out_dir / f"{diffae_path.stem}_train_std.npy"
    
    np.save(mean_path, train_mean)
    np.save(std_path, train_std)
    print(f"[SUCCESS] Saved normalization statistics files:\n  -> {mean_path}\n  -> {std_path}")
    print("\nPipeline complete. Datasets are perfectly synchronized.")

if __name__ == "__main__":
    DIFFAE_FILE = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    ARCFACE_FILE = "/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy"
    
    joint_split_and_extract_stats(
        diffae_npy_path=DIFFAE_FILE,
        arcface_npy_path=ARCFACE_FILE
    )
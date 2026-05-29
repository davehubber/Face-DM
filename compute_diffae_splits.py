import numpy as np
from pathlib import Path

def split_embeddings_and_extract_stats(master_npy_path: str, output_directory: str = None):
    """
    Splits a master embedding .npy file into 80/10/10 Train/Val/Test files.
    Calculates the mean and standard deviation STRICTLY from the training split
    to preserve downstream validation integrity.
    """
    master_path = Path(master_npy_path).resolve()
    if not master_path.exists():
        raise FileNotFoundError(f"Master embedding file not found at: {master_path}")
        
    out_dir = Path(output_directory).resolve() if output_directory else master_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading master embeddings from: {master_path}")
    embeddings = np.load(master_path).astype(np.float32)
    num_samples = len(embeddings)
    print(f"Total samples found: {num_samples}")
    
    np.random.seed(42)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    train_end = int(num_samples * 0.8)
    val_end = int(num_samples * 0.9)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_embs = embeddings[train_idx]
    val_embs = embeddings[val_idx]
    test_embs = embeddings[test_idx]
    
    print(f"\nSplit distribution metrics:")
    print(f"  - Train Split (80%): {len(train_embs)} vectors")
    print(f"  - Val Split   (10%): {len(val_embs)} vectors")
    print(f"  - Test Split  (10%): {len(test_embs)} vectors")
    
    base_stem = master_path.stem
    
    train_path = out_dir / f"{base_stem}_train.npy"
    val_path = out_dir / f"{base_stem}_val.npy"
    test_path = out_dir / f"{base_stem}_test.npy"
    
    np.save(train_path, train_embs)
    np.save(val_path, val_embs)
    np.save(test_path, test_embs)
    print(f"\n[SUCCESS] Split partitions exported safely to directory:\n  -> {out_dir}")
    
    print("\nExtracting feature-wise channel metrics from Train Split...")
    train_mean = np.mean(train_embs, axis=0)
    train_std = np.std(train_embs, axis=0)
    
    train_std = np.where(train_std == 0, 1.0, train_std)
    
    mean_path = out_dir / f"{base_stem}_train_mean.npy"
    std_path = out_dir / f"{base_stem}_train_std.npy"
    
    np.save(mean_path, train_mean)
    np.save(std_path, train_std)
    print(f"[SUCCESS] Saved normalization statistics files:\n  -> {mean_path}\n  -> {std_path}")

if __name__ == "__main__":
    TARGET_FILE = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    
    split_embeddings_and_extract_stats(master_npy_path=TARGET_FILE)
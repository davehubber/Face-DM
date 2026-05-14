import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Compute Z-score mean and std vectors for DiffAE embeddings."
    )
    
    parser.add_argument(
        "--embeddings",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy",
        help="Path to the saved raw DiffAE embeddings .npy file.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/",
        help="Directory to save the computed mean and std .npy files.",
    )

    args = parser.parse_args()

    embeddings_path = Path(args.embeddings).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    print(f"Loading embeddings from: {embeddings_path}")
    # Load as float32 to save memory and match standard precision
    embeddings = np.load(embeddings_path).astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape [N, D], but got {embeddings.shape}")

    num_samples, dim = embeddings.shape
    print(f"Loaded {num_samples} embeddings of dimension {dim}.")

    # Calculate mean and standard deviation across the dataset (axis=0)
    print("Computing mean...")
    zscore_mean = np.mean(embeddings, axis=0).astype(np.float32)
    
    print("Computing standard deviation...")
    # Add a tiny epsilon to std to prevent division by zero during normalization later
    zscore_std = np.std(embeddings, axis=0).astype(np.float32)
    zscore_std = np.maximum(zscore_std, 1e-8)

    # Define output paths
    mean_path = out_dir / f"{embeddings_path.stem}_mean.npy"
    std_path = out_dir / f"{embeddings_path.stem}_std.npy"

    # Save the files
    np.save(mean_path, zscore_mean)
    np.save(std_path, zscore_std)

    print("\nSuccess!")
    print(f"Saved mean vector shape {zscore_mean.shape} to: {mean_path}")
    print(f"Saved std vector shape {zscore_std.shape} to: {std_path}")

if __name__ == "__main__":
    main()
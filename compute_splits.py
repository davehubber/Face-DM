import csv
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class TestImageDataset(Dataset):
    """Lightweight dataset to load the 1000 selected test images for stochastic encoding."""
    def __init__(self, metadata_rows, image_size=256):
        self.rows = metadata_rows
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = Path(row['image_path'])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return {
            "img": img,
            "embedding_index": int(row['embedding_index']),
            "image_path": str(path),
            "filename": row['filename']
        }

def load_diffae_ffhq256_autoencoder(diffae_root: Path, checkpoint_path: Path, device: torch.device):
    """Loads the official DiffAE FFHQ256 autoencoder model."""
    diffae_root = Path(diffae_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    if not diffae_root.exists():
        raise FileNotFoundError(f"DiffAE repo not found: {diffae_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(diffae_root))
    old_cwd = os.getcwd()
    os.chdir(diffae_root)

    try:
        from templates import ffhq256_autoenc
        from config import PretrainConfig
        from experiment import LitModel

        conf = ffhq256_autoenc()
        conf.pretrain = PretrainConfig(name="ffhq256_autoenc", path=str(checkpoint_path))
        conf.latent_infer_path = None

        model = LitModel(conf)
        model = model.to(device)
        model.eval()

        if hasattr(model, "ema_model"):
            model.ema_model.eval()

        for p in model.parameters():
            p.requires_grad_(False)
    finally:
        os.chdir(old_cwd)

    return model

def joint_split_and_extract_stats(
    diffae_npy_path: str, 
    diffae_metadata_path: str,
    arcface_npy_path: str,
    diffae_root: str,
    checkpoint_path: str,
    device_str: str,
    batch_size: int,
    num_workers: int
):
    # 1. Resolve Paths
    diffae_path = Path(diffae_npy_path).resolve()
    metadata_path = Path(diffae_metadata_path).resolve()
    arcface_path = Path(arcface_npy_path).resolve()
    
    diffae_out_dir = diffae_path.parent
    arcface_out_dir = arcface_path.parent
    device = torch.device(device_str)

    # 2. Load Master Arrays and Metadata
    print("Loading master embedding files...")
    diffae_embs = np.load(diffae_path).astype(np.float32)
    arcface_embs = np.load(arcface_path).astype(np.float32)
    
    print(f"Reading metadata from: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))
        
    num_samples = len(diffae_embs)
    if num_samples != len(arcface_embs) or num_samples != len(metadata_rows):
        raise ValueError("DATASET MISMATCH: Total rows in embeddings and metadata do not match!")

    print(f"Alignment verified. Total paired master dataset samples: {num_samples}")

    # 3. Generate Splits (Deterministic Seed)
    np.random.seed(42)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Select exactly 1000 images for the Test Split, remainder to Train
    test_idx_A = indices[:1000]
    train_idx = indices[1000:]

    # Generate a random derangement for 'Side B' of the test pairs (no image paired with itself)
    rng = np.random.default_rng(42)
    test_idx_B = test_idx_A.copy()
    while True:
        rng.shuffle(test_idx_B)
        if np.all(test_idx_A != test_idx_B):
            break

    print(f"\nSplit distribution configured:")
    print(f"  - Train Split: {len(train_idx)} individual images")
    print(f"  - Test Split:  1000 unique images organized into 1000 distinct pairs")

    # 4. Save Train Splits & Normalization Metrics
    print("\nProcessing and exporting Train Split...")
    diffae_train = diffae_embs[train_idx]
    arcface_train = arcface_embs[train_idx]

    np.save(diffae_out_dir / f"{diffae_path.stem}_train.npy", diffae_train)
    np.save(arcface_out_dir / f"{arcface_path.stem}_train.npy", arcface_train)

    # Compute Z-Score Statistics (Train Only)
    train_mean = np.mean(diffae_train, axis=0)
    train_std = np.std(diffae_train, axis=0)
    train_std = np.where(train_std == 0, 1.0, train_std) # Division by zero guard
    np.save(diffae_out_dir / f"{diffae_path.stem}_train_mean.npy", train_mean)
    np.save(diffae_out_dir / f"{diffae_path.stem}_train_std.npy", train_std)

    # Save Train Metadata
    train_metadata_path = diffae_out_dir / f"{diffae_path.stem}_train_metadata.csv"
    with open(train_metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename"])
        writer.writeheader()
        for idx in train_idx:
            writer.writerow(metadata_rows[idx])

    # 5. Save Paired Test Semantic Embeddings
    print("Processing and exporting Paired Test Splits...")
    diffae_test_pairs = np.stack([diffae_embs[test_idx_A], diffae_embs[test_idx_B]], axis=1)
    arcface_test_pairs = np.stack([arcface_embs[test_idx_A], arcface_embs[test_idx_B]], axis=1)

    np.save(diffae_out_dir / f"{diffae_path.stem}_test_pairs.npy", diffae_test_pairs)
    np.save(arcface_out_dir / f"{arcface_path.stem}_test_pairs.npy", arcface_test_pairs)

    # Save Test Pairs Metadata mapping
    test_metadata_path = diffae_out_dir / f"{diffae_path.stem}_test_pairs_metadata.csv"
    with open(test_metadata_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "pair_index", "embedding_index_A", "image_path_A", "filename_A",
            "embedding_index_B", "image_path_B", "filename_B"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair_i, (idx_A, idx_B) in enumerate(zip(test_idx_A, test_idx_B)):
            writer.writerow({
                "pair_index": pair_i,
                "embedding_index_A": metadata_rows[idx_A]["embedding_index"],
                "image_path_A": metadata_rows[idx_A]["image_path"],
                "filename_A": metadata_rows[idx_A]["filename"],
                "embedding_index_B": metadata_rows[idx_B]["embedding_index"],
                "image_path_B": metadata_rows[idx_B]["image_path"],
                "filename_B": metadata_rows[idx_B]["filename"],
            })

    # 6. Extract and Store DiffAE Stochastic Codes (x_T) for Test Set
    print("\nInitializing DiffAE model to compute stochastic codes for the test set...")
    model = load_diffae_ffhq256_autoencoder(diffae_root, Path(checkpoint_path), device)
    
    # Isolate unique rows belonging to the test dataset (ordered by test_idx_A)
    test_metadata_subset = [metadata_rows[idx] for idx in test_idx_A]
    test_dataset = TestImageDataset(test_metadata_subset, image_size=256)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    xt_accumulated = []
    print("Extracting stochastic codes (x_T) for unique test images...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Encoding Stochastic Codes (T=250)"):
            imgs = batch["img"].to(device, non_blocking=True)
            cond = model.encode(imgs)
            # Match target time-horizon sequence logic from template
            xT = model.encode_stochastic(imgs, cond, T=250)
            xt_accumulated.append(xT.detach().cpu().numpy())

    # Shape: (1000, 3, 256, 256)
    xt_A_all = np.concatenate(xt_accumulated, axis=0).astype(np.float32)

    # Map Side B using fast indexing lookup vectors instead of running inference again
    lookup_map = {orig_idx: array_pos for array_pos, orig_idx in enumerate(test_idx_A)}
    b_positions_in_a = [lookup_map[orig_idx] for orig_idx in test_idx_B]
    xt_B_all = xt_A_all[b_positions_in_a]

    # Stack into perfectly matched paired tensors -> Shape: (1000, 2, 3, 256, 256)
    xt_test_pairs = np.stack([xt_A_all, xt_B_all], axis=1)

    xt_out_path = diffae_out_dir / "ffhq256_diffae_xt_test_pairs.npy"
    print(f"Saving noise-like stochastic code pairs tensor [Shape: {xt_test_pairs.shape}]...")
    np.save(xt_out_path, xt_test_pairs)

    print(f"\n[SUCCESS] Custom pipeline run finished.")
    print(f"  -> DiffAE Semantic Train: {diffae_out_dir / f'{diffae_path.stem}_train.npy'}")
    print(f"  -> DiffAE Semantic Pairs: {diffae_out_dir / f'{diffae_path.stem}_test_pairs.npy'}")
    print(f"  -> DiffAE Stochastic Pairs: {xt_out_path}")
    print(f"  -> ArcFace Semantic Pairs: {arcface_out_dir / f'{arcface_path.stem}_test_pairs.npy'}")

if __name__ == "__main__":
    DIFFAE_FILE = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    DIFFAE_METADATA = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_metadata.csv"
    ARCFACE_FILE = "/nas-ctm01/homes/dacordeiro/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy"
    
    DIFFAE_ROOT = "/nas-ctm01/homes/dacordeiro/diffae/"
    CHECKPOINT = "/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt"

    joint_split_and_extract_stats(
        diffae_npy_path=DIFFAE_FILE,
        diffae_metadata_path=DIFFAE_METADATA,
        arcface_npy_path=ARCFACE_FILE,
        diffae_root=DIFFAE_ROOT,
        checkpoint_path=CHECKPOINT,
        device_str="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        num_workers=4
    )
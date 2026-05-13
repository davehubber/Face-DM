import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def read_identity_file(identity_file: Path) -> Dict[str, str]:
    """
    Reads identity_CelebA.txt.

    Expected format per line:
        000001.jpg 2880

    Returns:
        filename -> original CelebA identity id as string
    """
    filename_to_identity = {}

    with open(identity_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            filename, identity_id = line.split()
            filename_to_identity[filename] = identity_id

    return filename_to_identity


def find_images_by_filename(image_root: Path) -> Dict[str, Path]:
    """
    Finds images recursively under image_root.

    The key is the basename, e.g. 000001.jpg, because identity_CelebA.txt
    is filename-based.

    If duplicate basenames exist, this raises an error to avoid silently
    assigning the wrong identity.
    """
    filename_to_path = {}

    for path in sorted(image_root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        filename = path.name

        if filename in filename_to_path:
            raise RuntimeError(
                f"Duplicate image filename found: {filename}\n"
                f"First path: {filename_to_path[filename]}\n"
                f"Second path: {path}\n"
                "CelebA identity annotations are filename-based, so duplicate "
                "basenames would make identity assignment ambiguous."
            )

        filename_to_path[filename] = path

    return filename_to_path


def build_rows(
    image_root: Path,
    identity_file: Path,
    min_images_per_identity: int,
    limit: Optional[int] = None,
) -> List[dict]:
    filename_to_identity = read_identity_file(identity_file)
    filename_to_path = find_images_by_filename(image_root)

    annotated_entries = []

    missing_identity = []

    for filename, path in filename_to_path.items():
        if filename not in filename_to_identity:
            missing_identity.append(filename)
            continue

        identity_id = filename_to_identity[filename]

        annotated_entries.append({
            "filename": filename,
            "image_path": str(path),
            "identity_id": identity_id,
        })

    if not annotated_entries:
        raise RuntimeError(
            "No images in image_root matched filenames in identity_CelebA.txt. "
            "Check that your image folder contains CelebA filenames such as 000001.jpg."
        )

    if missing_identity:
        print(
            f"Warning: {len(missing_identity)} images had no identity annotation "
            "and will be ignored."
        )

    identity_counts = Counter(row["identity_id"] for row in annotated_entries)

    kept_identity_ids = {
        identity_id
        for identity_id, count in identity_counts.items()
        if count >= min_images_per_identity
    }

    filtered_rows = [
        row for row in annotated_entries
        if row["identity_id"] in kept_identity_ids
    ]

    if limit is not None:
        filtered_rows = filtered_rows[:limit]

    if not filtered_rows:
        raise RuntimeError(
            f"No identities with >= {min_images_per_identity} images were found "
            "among the images present in image_root."
        )

    # Stable contiguous labels for classification.
    # Original CelebA identity IDs are not guaranteed to be contiguous after filtering.
    sorted_identity_ids = sorted(kept_identity_ids, key=lambda x: int(x))
    identity_to_label = {
        identity_id: label
        for label, identity_id in enumerate(sorted_identity_ids)
    }

    final_rows = []

    for i, row in enumerate(sorted(filtered_rows, key=lambda r: r["filename"])):
        identity_id = row["identity_id"]
        label = identity_to_label[identity_id]

        final_rows.append({
            "embedding_index": i,
            "filename": row["filename"],
            "image_path": row["image_path"],
            "identity_id": identity_id,
            "label": label,
            "images_per_identity": identity_counts[identity_id],
        })

    return final_rows


def assign_stratified_splits(
    rows: List[dict],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> List[dict]:
    """
    Assigns train/val/test split independently inside each identity.

    This guarantees every identity appears in train, val, and test,
    assuming min_images_per_identity is large enough.
    """
    test_frac = 1.0 - train_frac - val_frac

    if train_frac <= 0 or val_frac <= 0 or test_frac <= 0:
        raise ValueError("train_frac, val_frac, and test_frac must all be > 0.")

    rng = random.Random(seed)

    by_label = defaultdict(list)

    for row in rows:
        by_label[row["label"]].append(row)

    split_rows = []

    for label, group in by_label.items():
        group = list(group)
        rng.shuffle(group)

        n = len(group)

        n_val = max(1, round(n * val_frac))
        n_test = max(1, round(n * test_frac))
        n_train = n - n_val - n_test

        if n_train <= 0:
            raise RuntimeError(
                f"Identity label {label} has only {n} images, which is not enough "
                "for the requested split."
            )

        for row in group[:n_train]:
            row = dict(row)
            row["split"] = "train"
            split_rows.append(row)

        for row in group[n_train:n_train + n_val]:
            row = dict(row)
            row["split"] = "val"
            split_rows.append(row)

        for row in group[n_train + n_val:]:
            row = dict(row)
            row["split"] = "test"
            split_rows.append(row)

    # Restore embedding order.
    split_rows = sorted(split_rows, key=lambda r: r["embedding_index"])

    # Reassign embedding_index after split sorting, just to be absolutely clean.
    for i, row in enumerate(split_rows):
        row["embedding_index"] = i

    return split_rows


class CelebAIdentityDataset(Dataset):
    def __init__(self, rows: List[dict], image_size: int = 256):
        self.rows = rows

        # Deterministic transform. No random horizontal flip.
        # This should match your FFHQ DiffAE encoding setup.
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]

        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)

        return {
            "img": img,
            "embedding_index": row["embedding_index"],
            "filename": row["filename"],
            "image_path": row["image_path"],
            "identity_id": row["identity_id"],
            "label": row["label"],
            "split": row["split"],
        }


def load_diffae_ffhq256_autoencoder(
    diffae_root: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    """
    Loads the official DiffAE FFHQ256 autoencoder.

    The semantic embedding is obtained with:
        model.encode(x)

    For ffhq256_autoenc, the expected semantic embedding dimension is 512.
    """
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
        conf.pretrain = PretrainConfig(
            name="ffhq256_autoenc",
            path=str(checkpoint_path),
        )

        # We only need z_sem. No latent DPM inference path needed.
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


def save_metadata_csv(rows: List[dict], path: Path):
    fieldnames = [
        "embedding_index",
        "filename",
        "image_path",
        "identity_id",
        "label",
        "split",
        "images_per_identity",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row[key]
                for key in fieldnames
            })


def save_label_map_csv(rows: List[dict], path: Path):
    by_label = defaultdict(list)

    for row in rows:
        by_label[row["label"]].append(row)

    label_rows = []

    for label, group in sorted(by_label.items()):
        identity_id = group[0]["identity_id"]

        split_counts = Counter(row["split"] for row in group)

        label_rows.append({
            "label": label,
            "identity_id": identity_id,
            "count_total": len(group),
            "count_train": split_counts["train"],
            "count_val": split_counts["val"],
            "count_test": split_counts["test"],
        })

    fieldnames = [
        "label",
        "identity_id",
        "count_total",
        "count_train",
        "count_val",
        "count_test",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(label_rows)


def compute_and_save_zscore(
    embeddings: np.ndarray,
    rows: List[dict],
    out_dir: Path,
    eps: float = 1e-6,
):
    split_array = np.array([row["split"] for row in rows])
    train_mask = split_array == "train"

    if train_mask.sum() == 0:
        raise RuntimeError("No training samples found when computing z-score stats.")

    train_embeddings = embeddings[train_mask]

    mean = train_embeddings.mean(axis=0).astype(np.float32)
    std = train_embeddings.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)

    embeddings_zscore = ((embeddings - mean) / std).astype(np.float32)

    np.save(out_dir / "celeba_diffae_zsem_zscore.npy", embeddings_zscore)

    np.savez(
        out_dir / "celeba_diffae_zsem_train_stats.npz",
        mean=mean,
        std=std,
        eps=np.array(eps, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diffae-root",
        type=str,
        required=True,
        help="Path to cloned DiffAE repository.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ffhq256_autoenc/last.ckpt.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help=(
            "Folder containing your filtered CelebA aligned/cropped images. "
            "Filenames should still be CelebA names like 000001.jpg."
        ),
    )
    parser.add_argument(
        "--identity-file",
        type=str,
        required=True,
        help="Path to identity_CelebA.txt.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for embeddings and MLP-ready metadata.",
    )
    parser.add_argument(
        "--min-images-per-identity",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional small limit for a smoke test. Omit for the full dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    diffae_root = Path(args.diffae_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    image_root = Path(args.image_root).resolve()
    identity_file = Path(args.identity_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print(f"DiffAE root: {diffae_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"CelebA image root: {image_root}")
    print(f"Identity file: {identity_file}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")
    print(f"Min images per identity: {args.min_images_per_identity}")

    rows = build_rows(
        image_root=image_root,
        identity_file=identity_file,
        min_images_per_identity=args.min_images_per_identity,
        limit=args.limit,
    )

    rows = assign_stratified_splits(
        rows=rows,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    num_images = len(rows)
    num_identities = len(set(row["label"] for row in rows))
    split_counts = Counter(row["split"] for row in rows)

    print(f"Final image count: {num_images}")
    print(f"Final identity count: {num_identities}")
    print(f"Split counts: {dict(split_counts)}")

    dataset = CelebAIdentityDataset(
        rows=rows,
        image_size=args.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = load_diffae_ffhq256_autoencoder(
        diffae_root=diffae_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding CelebA z_sem"):
            imgs = batch["img"].to(device, non_blocking=True)

            z_sem = model.encode(imgs)
            z_sem = z_sem.detach().float().cpu().numpy()

            all_embeddings.append(z_sem)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    if embeddings.shape[0] != len(rows):
        raise RuntimeError(
            f"Embedding count mismatch: got {embeddings.shape[0]}, expected {len(rows)}."
        )

    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2D embedding array, got shape {embeddings.shape}.")

    if embeddings.shape[1] != 512:
        print(
            "Warning: expected 512-D semantic embeddings for ffhq256_autoenc, "
            f"but got dimension {embeddings.shape[1]}."
        )

    raw_emb_path = out_dir / "celeba_diffae_zsem.npy"
    metadata_path = out_dir / "celeba_diffae_zsem_metadata.csv"
    label_map_path = out_dir / "celeba_identity_label_map.csv"
    info_path = out_dir / "dataset_info.json"

    np.save(raw_emb_path, embeddings)
    save_metadata_csv(rows, metadata_path)
    save_label_map_csv(rows, label_map_path)

    compute_and_save_zscore(
        embeddings=embeddings,
        rows=rows,
        out_dir=out_dir,
    )

    info = {
        "num_images": num_images,
        "num_identities": num_identities,
        "embedding_shape": list(embeddings.shape),
        "min_images_per_identity": args.min_images_per_identity,
        "image_size": args.image_size,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": 1.0 - args.train_frac - args.val_frac,
        "seed": args.seed,
        "split_counts": dict(split_counts),
        "raw_embeddings": str(raw_emb_path),
        "zscore_embeddings": str(out_dir / "celeba_diffae_zsem_zscore.npy"),
        "zscore_stats": str(out_dir / "celeba_diffae_zsem_train_stats.npz"),
        "metadata_csv": str(metadata_path),
        "label_map_csv": str(label_map_path),
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("\nDone.")
    print(f"Saved raw embeddings: {raw_emb_path}")
    print(f"Saved z-scored embeddings: {out_dir / 'celeba_diffae_zsem_zscore.npy'}")
    print(f"Saved z-score train stats: {out_dir / 'celeba_diffae_zsem_train_stats.npz'}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved label map: {label_map_path}")
    print(f"Saved dataset info: {info_path}")
    print(f"Embedding array shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
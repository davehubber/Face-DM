import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class FFHQImageFolder(Dataset):
    def __init__(self, image_root: Path, image_size: int = 256, limit: Optional[int] = None):
        self.image_root = Path(image_root)

        exts = {".png", ".jpg", ".jpeg", ".webp"}
        self.paths = sorted(
            p for p in self.image_root.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )

        if limit is not None:
            self.paths = self.paths[:limit]

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
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return {
            "img": img,
            "index": index,
            "path": str(path),
            "filename": path.name,
        }


def load_diffae_ffhq256_autoencoder(diffae_root: Path, checkpoint_path: Path, device: torch.device):
    """
    Loads the official DiffAE FFHQ256 autoencoder and returns the LitModel.
    The semantic encoder used is model.encode(x), which internally uses the EMA encoder.
    """
    diffae_root = Path(diffae_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    if not diffae_root.exists():
        raise FileNotFoundError(f"DiffAE repo not found: {diffae_root}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # DiffAE uses many local imports, so make the repo importable.
    sys.path.insert(0, str(diffae_root))

    # Some DiffAE paths are relative to repo root.
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

        # Important: we only want to encode images into z_sem.
        # latent_infer_path is for precomputed latent statistics / latent DPM usage.
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diffae-root",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/diffae/",
        help="Path to your cloned DiffAE repository, e.g. ./diffae",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="/nas-ctm01/datasets/public/ffhq256/",
        help="Path to your FFHQ-256 image folder",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt",
        help="Path to DiffAE FFHQ256 autoencoder checkpoint, e.g. ./diffae/checkpoints/ffhq256_autoenc/last.ckpt",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings",
        help="Output directory for embeddings and metadata",
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
        help="Optional limit, e.g. 1000 for a test run. Omit for full dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    diffae_root = Path(args.diffae_root).resolve()
    image_root = Path(args.image_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print(f"DiffAE root: {diffae_root}")
    print(f"Image root: {image_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")

    dataset = FFHQImageFolder(
        image_root=image_root,
        image_size=256,
        limit=args.limit,
    )

    print(f"Found {len(dataset)} images")

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
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding z_sem"):
            imgs = batch["img"].to(device, non_blocking=True)

            # DiffAE semantic embedding.
            # For FFHQ256 autoenc, this should normally be [B, 512].
            z_sem = model.encode(imgs)
            z_sem = z_sem.detach().float().cpu().numpy()

            all_embeddings.append(z_sem)

            batch_indices = batch["index"].tolist()
            batch_paths = batch["path"]
            batch_filenames = batch["filename"]

            for local_i, emb_index in enumerate(batch_indices):
                rows.append({
                    "embedding_index": int(emb_index),
                    "image_path": batch_paths[local_i],
                    "filename": batch_filenames[local_i],
                })

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

    emb_path = out_dir / "ffhq256_diffae_zsem.npy"
    csv_path = out_dir / "ffhq256_diffae_zsem_metadata.csv"

    np.save(emb_path, embeddings)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["embedding_index", "image_path", "filename"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved embeddings to: {emb_path}")
    print(f"Saved metadata to: {csv_path}")
    print(f"Embedding array shape: {embeddings.shape}")

    if embeddings.shape[1] != 512:
        print(
            "Warning: expected 512-D semantic embeddings for FFHQ256 DiffAE, "
            f"but got dimension {embeddings.shape[1]}."
        )


if __name__ == "__main__":
    main()

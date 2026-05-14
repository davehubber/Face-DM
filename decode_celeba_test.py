import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def read_metadata(metadata_path: Path):
    rows = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {
            "embedding_index",
            "filename",
            "image_path",
            "identity_id",
            "label",
            "split",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

        for row in reader:
            row = dict(row)
            row["embedding_index"] = int(row["embedding_index"])
            row["label"] = int(row["label"])
            rows.append(row)

    rows = sorted(rows, key=lambda r: r["embedding_index"])

    expected = list(range(len(rows)))
    actual = [r["embedding_index"] for r in rows]

    if actual != expected:
        raise ValueError(
            "embedding_index values are not exactly 0..N-1. "
            "This script assumes metadata directly indexes the embedding array."
        )

    return rows


def select_row(rows, index: Optional[int], filename: Optional[str]):
    if index is not None and filename is not None:
        raise ValueError("Use either --index or --filename, not both.")

    if filename is not None:
        matches = [row for row in rows if row["filename"] == filename]

        if len(matches) == 0:
            raise ValueError(f"Filename not found in metadata: {filename}")

        if len(matches) > 1:
            raise ValueError(f"Multiple metadata rows found for filename: {filename}")

        return matches[0]

    if index is None:
        index = 0

    if index < 0 or index >= len(rows):
        raise IndexError(f"Index {index} is outside metadata range 0..{len(rows) - 1}")

    return rows[index]


def resolve_image_path(row, override_image_root: Optional[Path]):
    if override_image_root is not None:
        path = override_image_root / row["filename"]
    else:
        path = Path(row["image_path"])

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find original image: {path}\n"
            "If the metadata image_path is no longer valid, pass --override-image-root."
        )

    return path


def load_diffae_ffhq256_autoencoder(
    diffae_root: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    """
    Loads the same FFHQ256 DiffAE autoencoder style used in your encoding script.
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

        # We are not using the latent DPM. We only need autoencoding.
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


def make_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])


def input_tensor_to_vis(x: torch.Tensor) -> torch.Tensor:
    """
    Converts an input tensor from [-1, 1] to [0, 1].
    Expected shape: [C, H, W] or [B, C, H, W].
    """
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def render_tensor_to_vis(x: torch.Tensor) -> torch.Tensor:
    """
    DiffAE render output is usually already saveable in [0, 1] in the official
    prediction code. This function also handles the case where the output is
    in [-1, 1].
    """
    x = x.detach().float().cpu()

    min_val = float(x.min())
    max_val = float(x.max())

    if min_val < -0.05:
        x = (x + 1.0) / 2.0

    return x.clamp(0.0, 1.0)


def maybe_denormalize_zscore_embedding(
    z: np.ndarray,
    stats_path: Optional[Path],
) -> np.ndarray:
    if stats_path is None:
        return z

    if not stats_path.exists():
        raise FileNotFoundError(f"Z-score stats file not found: {stats_path}")

    stats = np.load(stats_path)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)

    return (z * std + mean).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diffae-root",
        type=str,
        required=True,
        help="Path to your cloned DiffAE repository.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to ffhq256_autoenc/last.ckpt.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing CelebA DiffAE embeddings and metadata.",
    )
    parser.add_argument(
        "--embeddings-file",
        type=str,
        default="celeba_diffae_zsem.npy",
        help=(
            "Raw semantic embedding file. Prefer celeba_diffae_zsem.npy. "
            "Do not use the z-scored file unless also passing --zscore-stats."
        ),
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="celeba_diffae_zsem_metadata.csv",
    )
    parser.add_argument(
        "--zscore-stats",
        type=str,
        default=None,
        help=(
            "Optional path to celeba_diffae_zsem_train_stats.npz. "
            "Only use this if --embeddings-file is celeba_diffae_zsem_zscore.npy."
        ),
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Embedding index to decode. Defaults to 0 if neither index nor filename is provided.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Decode the embedding corresponding to this CelebA filename, e.g. 000123.jpg.",
    )
    parser.add_argument(
        "--override-image-root",
        type=str,
        default=None,
        help=(
            "Optional folder containing the original aligned CelebA images. "
            "Useful if image_path inside the metadata CSV is no longer valid."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for original/reconstruction images.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--t-inv",
        type=int,
        default=200,
        help="Number of DDIM inversion steps used to obtain stochastic latent.",
    )
    parser.add_argument(
        "--t-step",
        type=int,
        default=200,
        help="Number of render/decoding steps.",
    )
    parser.add_argument(
        "--also-recompute-semantic",
        action="store_true",
        help=(
            "Also recompute z_sem from the image and save a second reconstruction. "
            "This helps verify whether the stored z_sem matches a fresh encode."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    diffae_root = Path(args.diffae_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    embeddings_path = dataset_dir / args.embeddings_file
    metadata_path = dataset_dir / args.metadata_file
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    zscore_stats_path = (
        Path(args.zscore_stats).resolve()
        if args.zscore_stats is not None
        else None
    )

    override_image_root = (
        Path(args.override_image_root).resolve()
        if args.override_image_root is not None
        else None
    )

    device = torch.device(args.device)

    print(f"DiffAE root: {diffae_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embeddings_path}")

    rows = read_metadata(metadata_path)
    row = select_row(rows, index=args.index, filename=args.filename)

    embedding_index = row["embedding_index"]
    image_path = resolve_image_path(row, override_image_root)

    print("\nSelected sample:")
    print(f"  embedding_index: {embedding_index}")
    print(f"  filename:        {row['filename']}")
    print(f"  identity_id:     {row['identity_id']}")
    print(f"  label:           {row['label']}")
    print(f"  split:           {row['split']}")
    print(f"  image_path:      {image_path}")

    embeddings = np.load(embeddings_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    if embedding_index < 0 or embedding_index >= embeddings.shape[0]:
        raise IndexError(
            f"embedding_index {embedding_index} outside embedding array range "
            f"0..{embeddings.shape[0] - 1}"
        )

    z_sem_np = embeddings[embedding_index].astype(np.float32)
    z_sem_np = maybe_denormalize_zscore_embedding(z_sem_np, zscore_stats_path)

    if z_sem_np.shape[0] != 512:
        print(
            f"Warning: expected a 512-D semantic embedding, got shape {z_sem_np.shape}"
        )

    transform = make_transform(args.image_size)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    z_sem_stored = torch.from_numpy(z_sem_np).float().unsqueeze(0).to(device)

    model = load_diffae_ffhq256_autoencoder(
        diffae_root=diffae_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    with torch.no_grad():
        print("\nObtaining stochastic latent from original image and stored z_sem...")
        stochastic_latent = model.encode_stochastic(
            img_tensor,
            z_sem_stored,
            T=args.t_inv,
        )

        print("Rendering reconstruction from stored z_sem + stochastic latent...")
        recon_stored = model.render(
            stochastic_latent,
            z_sem_stored,
            T=args.t_step,
        )[0]

        original_vis = input_tensor_to_vis(img_tensor[0].detach().cpu())
        recon_stored_vis = render_tensor_to_vis(recon_stored)

        save_image(original_vis, out_dir / "original_preprocessed.png")
        save_image(recon_stored_vis, out_dir / "reconstruction_from_stored_zsem.png")

        comparison_images = [original_vis, recon_stored_vis]
        comparison_names = ["original", "stored_zsem_reconstruction"]

        if args.also_recompute_semantic:
            print("\nAlso recomputing z_sem from the original image...")
            z_sem_recomputed = model.encode(img_tensor)

            cosine = F.cosine_similarity(z_sem_stored, z_sem_recomputed, dim=1).item()
            l1 = torch.abs(z_sem_stored - z_sem_recomputed).mean().item()
            mse = F.mse_loss(z_sem_stored, z_sem_recomputed).item()

            print("Stored vs recomputed z_sem:")
            print(f"  cosine similarity: {cosine:.8f}")
            print(f"  mean L1:           {l1:.8f}")
            print(f"  MSE:               {mse:.8f}")

            stochastic_recomputed = model.encode_stochastic(
                img_tensor,
                z_sem_recomputed,
                T=args.t_inv,
            )

            recon_recomputed = model.render(
                stochastic_recomputed,
                z_sem_recomputed,
                T=args.t_step,
            )[0]

            recon_recomputed_vis = render_tensor_to_vis(recon_recomputed)

            save_image(
                recon_recomputed_vis,
                out_dir / "reconstruction_from_recomputed_zsem.png",
            )

            comparison_images.append(recon_recomputed_vis)
            comparison_names.append("recomputed_zsem_reconstruction")

            with open(out_dir / "stored_vs_recomputed_zsem_metrics.txt", "w") as f:
                f.write(f"cosine_similarity: {cosine:.8f}\n")
                f.write(f"mean_l1: {l1:.8f}\n")
                f.write(f"mse: {mse:.8f}\n")

        comparison = torch.stack(comparison_images, dim=0)
        save_image(
            comparison,
            out_dir / "comparison.png",
            nrow=len(comparison_images),
        )

    print("\nSaved:")
    print(f"  {out_dir / 'original_preprocessed.png'}")
    print(f"  {out_dir / 'reconstruction_from_stored_zsem.png'}")
    print(f"  {out_dir / 'comparison.png'}")

    if args.also_recompute_semantic:
        print(f"  {out_dir / 'reconstruction_from_recomputed_zsem.png'}")
        print(f"  {out_dir / 'stored_vs_recomputed_zsem_metrics.txt'}")

    print("\nComparison order:")
    for i, name in enumerate(comparison_names):
        print(f"  column {i + 1}: {name}")


if __name__ == "__main__":
    main()
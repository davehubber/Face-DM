import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    return x / max(norm, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def average_two_images(path_a: str, path_b: str) -> Image.Image:
    img_a = Image.open(path_a).convert("RGB")
    img_b = Image.open(path_b).convert("RGB")

    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.BICUBIC)

    arr_a = np.asarray(img_a).astype(np.float32)
    arr_b = np.asarray(img_b).astype(np.float32)

    avg = ((arr_a + arr_b) / 2.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(avg, mode="RGB")


def load_diffae_ffhq256_autoencoder(
    diffae_root: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    """
    Loads the official DiffAE FFHQ256 autoencoder.

    The semantic embedding is obtained with:

        z_sem = model.encode(x)

    This does not compute or use the stochastic latent.
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


def build_diffae_transform(image_size: int = 256):
    """
    DiffAE FFHQ256-style preprocessing.

    No face detection.
    No ArcFace-style 112x112 crop.
    No random flip.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])


def encode_average_image_zsem(
    model,
    avg_img: Image.Image,
    transform,
    device: torch.device,
) -> np.ndarray:
    x = transform(avg_img).unsqueeze(0).to(device)

    with torch.no_grad():
        z_sem = model.encode(x)

    z_sem = z_sem.detach().float().cpu().numpy()[0]

    if z_sem.ndim != 1:
        raise RuntimeError(f"Unexpected z_sem shape: {z_sem.shape}")

    return z_sem.astype(np.float32)


def maybe_load_zscore(mean_path: Optional[str], std_path: Optional[str]):
    if mean_path is None and std_path is None:
        return None, None

    if mean_path is None or std_path is None:
        raise ValueError("Pass both --zscore-mean and --zscore-std, or neither.")

    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)

    std = np.maximum(std, 1e-8)

    return mean, std


def maybe_apply_zscore(x: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return x.astype(np.float32)

    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diffae-root",
        type=str,
        required=True,
        help="Path to your cloned DiffAE repository, e.g. ./diffae",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to FFHQ256 DiffAE autoencoder checkpoint, e.g. ./diffae/checkpoints/ffhq256_autoenc/last.ckpt",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to your saved FFHQ DiffAE semantic embeddings .npy file.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to matching metadata CSV with image_path column.",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        required=True,
        help="Path to save the txt report.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--keep-averaged-images",
        action="store_true",
        help="If set, saves the averaged images next to the report.",
    )
    parser.add_argument(
        "--zscore-mean",
        type=str,
        default=None,
        help="Optional .npy mean vector if your saved DiffAE embeddings are z-scored.",
    )
    parser.add_argument(
        "--zscore-std",
        type=str,
        default=None,
        help="Optional .npy std vector if your saved DiffAE embeddings are z-scored.",
    )

    args = parser.parse_args()

    diffae_root = Path(args.diffae_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    embeddings_path = Path(args.embeddings).resolve()
    metadata_path = Path(args.metadata).resolve()
    out_report = Path(args.out_report).resolve()
    out_report.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)

    if "image_path" not in metadata.columns:
        raise ValueError("Metadata CSV must contain an 'image_path' column.")

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Metadata rows ({len(metadata)}) and embeddings ({len(embeddings)}) do not match."
        )

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings shape [N, D], got {embeddings.shape}")

    zscore_mean, zscore_std = maybe_load_zscore(args.zscore_mean, args.zscore_std)

    if zscore_mean is not None:
        if zscore_mean.shape[-1] != embeddings.shape[-1]:
            raise ValueError(
                f"zscore mean dim {zscore_mean.shape} does not match embedding dim {embeddings.shape[-1]}"
            )
        if zscore_std.shape[-1] != embeddings.shape[-1]:
            raise ValueError(
                f"zscore std dim {zscore_std.shape} does not match embedding dim {embeddings.shape[-1]}"
            )

    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded metadata rows: {len(metadata)}")
    print(f"Device: {device}")

    model = load_diffae_ffhq256_autoencoder(
        diffae_root=diffae_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    transform = build_diffae_transform(image_size=256)

    n = len(metadata)
    if n < 2:
        raise ValueError("Need at least 2 images to form pairs.")

    rng = np.random.default_rng(args.seed)

    if n >= args.num_pairs * 2:
        selected = rng.choice(n, size=args.num_pairs * 2, replace=False)
        pairs = selected.reshape(args.num_pairs, 2)
    else:
        pairs = np.array([
            rng.choice(n, size=2, replace=False)
            for _ in range(args.num_pairs)
        ])

    if args.keep_averaged_images:
        avg_dir = out_report.parent / "averaged_images_diffae"
        avg_dir.mkdir(parents=True, exist_ok=True)
        tmp_context = None
    else:
        tmp_context = tempfile.TemporaryDirectory()
        avg_dir = Path(tmp_context.name)

    successes = []
    failures = []

    try:
        for pair_idx, (idx_a, idx_b) in enumerate(tqdm(pairs, desc="Testing averaged faces with DiffAE")):
            path_a = metadata.loc[idx_a, "image_path"]
            path_b = metadata.loc[idx_b, "image_path"]

            try:
                avg_img = average_two_images(path_a, path_b)

                avg_path = avg_dir / f"avg_pair_{pair_idx:04d}_idx_{idx_a}_{idx_b}.png"
                avg_img.save(avg_path)

                avg_zsem_raw = encode_average_image_zsem(
                    model=model,
                    avg_img=avg_img,
                    transform=transform,
                    device=device,
                )

                avg_zsem = maybe_apply_zscore(avg_zsem_raw, zscore_mean, zscore_std)

                zsem_a = embeddings[idx_a].astype(np.float32)
                zsem_b = embeddings[idx_b].astype(np.float32)

                sim_to_a = cosine_similarity(avg_zsem, zsem_a)
                sim_to_b = cosine_similarity(avg_zsem, zsem_b)

                zsem_pair_avg = l2_normalize((zsem_a + zsem_b) / 2.0)
                sim_to_embedding_average = cosine_similarity(avg_zsem, zsem_pair_avg)

                successes.append({
                    "pair_idx": pair_idx,
                    "idx_a": int(idx_a),
                    "idx_b": int(idx_b),
                    "image_a": path_a,
                    "image_b": path_b,
                    "avg_image": str(avg_path) if args.keep_averaged_images else "not_saved",
                    "cosine_to_a": sim_to_a,
                    "cosine_to_b": sim_to_b,
                    "cosine_to_embedding_average": sim_to_embedding_average,
                })

            except Exception as e:
                failures.append({
                    "pair_idx": pair_idx,
                    "idx_a": int(idx_a),
                    "idx_b": int(idx_b),
                    "image_a": path_a,
                    "image_b": path_b,
                    "error": repr(e),
                })

        with open(out_report, "w", encoding="utf-8") as f:
            f.write("DiffAE semantic embeddings on averaged FFHQ-256 image pairs\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"DiffAE root: {diffae_root}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Original semantic embeddings file: {embeddings_path}\n")
            f.write(f"Metadata file: {metadata_path}\n")
            f.write(f"Embedding shape: {embeddings.shape}\n")
            f.write(f"Number of requested pairs: {args.num_pairs}\n")
            f.write(f"Random seed: {args.seed}\n")
            f.write(f"Device: {device}\n\n")

            f.write("Preprocessing\n")
            f.write("-" * 80 + "\n")
            f.write("RGB\n")
            f.write("Resize(256)\n")
            f.write("CenterCrop(256)\n")
            f.write("ToTensor()\n")
            f.write("Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))\n")
            f.write("No face detector, no ArcFace crop, no stochastic latent.\n\n")

            f.write("Z-score handling\n")
            f.write("-" * 80 + "\n")
            if zscore_mean is None:
                f.write("No z-score applied to averaged-image z_sem.\n")
                f.write("Assumption: saved embeddings are raw DiffAE z_sem vectors.\n\n")
            else:
                f.write(f"Applied z-score to averaged-image z_sem using:\n")
                f.write(f"mean: {args.zscore_mean}\n")
                f.write(f"std: {args.zscore_std}\n")
                f.write("Assumption: saved embeddings are already z-scored with these same statistics.\n\n")

            f.write("Summary\n")
            f.write("-" * 80 + "\n")
            f.write(f"Succeeded: {len(successes)}\n")
            f.write(f"Failed: {len(failures)}\n\n")

            f.write("Successful pairs\n")
            f.write("-" * 80 + "\n")
            for item in successes:
                f.write(
                    f"pair {item['pair_idx']:04d} | "
                    f"idx_a={item['idx_a']} | "
                    f"idx_b={item['idx_b']} | "
                    f"cos(avg_img_zsem, zsem_a)={item['cosine_to_a']:.6f} | "
                    f"cos(avg_img_zsem, zsem_b)={item['cosine_to_b']:.6f} | "
                    f"cos(avg_img_zsem, avg_zsem_pair)={item['cosine_to_embedding_average']:.6f} | "
                    f"image_a={item['image_a']} | "
                    f"image_b={item['image_b']}\n"
                )

            f.write("\nFailed pairs\n")
            f.write("-" * 80 + "\n")
            if len(failures) == 0:
                f.write("None\n")
            else:
                for item in failures:
                    f.write(
                        f"pair {item['pair_idx']:04d} | "
                        f"idx_a={item['idx_a']} | "
                        f"idx_b={item['idx_b']} | "
                        f"image_a={item['image_a']} | "
                        f"image_b={item['image_b']} | "
                        f"error={item['error']}\n"
                    )

        print(f"Saved report to: {out_report}")
        print(f"Succeeded: {len(successes)}")
        print(f"Failed: {len(failures)}")

    finally:
        if tmp_context is not None:
            tmp_context.cleanup()


if __name__ == "__main__":
    main()

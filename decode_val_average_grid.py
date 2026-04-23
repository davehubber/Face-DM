import argparse
import csv
import os
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from utils_semantic import get_data, load_zscore_stats, setup_logging


@torch.no_grad()
def load_diffae_autoencoder(checkpoint_path: str, device: torch.device):
    from templates import ffhq256_autoenc
    from config import PretrainConfig
    from experiment import LitModel

    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(name="ffhq256_autoenc", path=checkpoint_path)
    conf.latent_infer_path = None

    model = LitModel(conf).to(device)
    model.eval()
    if hasattr(model, "ema_model") and model.ema_model is not None:
        model.ema_model.eval()
    return model


def unnormalize_semantic(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return z * std + mean


def _extract_image_tensor(decoded: Any) -> torch.Tensor:
    if isinstance(decoded, torch.Tensor):
        return decoded

    if isinstance(decoded, (list, tuple)) and len(decoded) > 0:
        for item in decoded:
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(decoded, dict):
        for key in ("pred", "sample", "samples", "image", "images", "img", "imgs", "x", "x_start"):
            value = decoded.get(key)
            if isinstance(value, torch.Tensor):
                return value

    raise TypeError(f"Could not extract decoded image tensor from object of type {type(decoded)!r}")


def _decode_attempts(target: Any, z_sem: torch.Tensor) -> Iterable[torch.Tensor]:
    if target is None:
        return

    if hasattr(target, "render") and callable(target.render):
        for fn in (
            lambda: target.render(z_sem=z_sem),
            lambda: target.render(z_sem),
            lambda: target.render(x=None, cond=None, z_sem=z_sem),
        ):
            try:
                yield _extract_image_tensor(fn())
            except Exception:
                pass

    if hasattr(target, "sample") and callable(target.sample):
        for fn in (
            lambda: target.sample(z_sem=z_sem),
            lambda: target.sample(z_sem),
        ):
            try:
                yield _extract_image_tensor(fn())
            except Exception:
                pass

    if callable(target):
        for fn in (
            lambda: target(z_sem=z_sem),
            lambda: target(z_sem),
        ):
            try:
                yield _extract_image_tensor(fn())
            except Exception:
                pass


@torch.no_grad()
def decode_semantic_batch(autoencoder, z_sem: torch.Tensor) -> torch.Tensor:
    decode_targets = []
    if hasattr(autoencoder, "ema_model"):
        decode_targets.append(autoencoder.ema_model)
    decode_targets.append(autoencoder)

    for target in decode_targets:
        for images in _decode_attempts(target, z_sem):
            if images.ndim == 4:
                return images

    raise RuntimeError(
        "Could not decode semantic latents with the available DiffAE API. "
        "Adjust decode_semantic_batch() to match your local LitModel/ema_model decode method."
    )


def tensor_to_display_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().float().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image.shape)}")

    if image.min().item() < 0.0:
        image = (image.clamp(-1.0, 1.0) + 1.0) / 2.0
    else:
        image = image.clamp(0.0, 1.0)

    return to_pil_image(image)


@torch.no_grad()
def collect_first_validation_batch(args) -> dict:
    data_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        train_samples_per_epoch=1,
        val_samples=args.num_pairs,
        batch_size=args.num_pairs,
        num_workers=args.num_workers,
    )
    val_loader = get_data(data_args, "val")
    return next(iter(val_loader))


def build_grid(
    dominant_images: torch.Tensor,
    average_images: torch.Tensor,
    recessive_images: torch.Tensor,
    rows_metadata: list[dict],
    out_path: str,
    tile_size: int = 256,
    margin: int = 16,
    column_gap: int = 12,
    row_gap: int = 18,
    header_h: int = 28,
    footer_h: int = 18,
    index_w: int = 44,
):
    num_rows = dominant_images.shape[0]
    col_titles = ["Dominant", "Average", "Recessive"]
    font = ImageFont.load_default()

    canvas_w = index_w + 3 * tile_size + 2 * column_gap + 2 * margin
    canvas_h = margin + header_h + num_rows * (tile_size + footer_h) + (num_rows - 1) * row_gap + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x0 = margin + index_w
    y0 = margin

    for col_idx, title in enumerate(col_titles):
        tx = x0 + col_idx * (tile_size + column_gap) + tile_size // 2
        bbox = draw.textbbox((0, 0), title, font=font)
        draw.text((tx - (bbox[2] - bbox[0]) / 2, y0), title, fill=(0, 0, 0), font=font)

    y = y0 + header_h
    for row_idx in range(num_rows):
        draw.text((margin, y + tile_size // 2 - 6), f"{row_idx:02d}", fill=(0, 0, 0), font=font)

        row_images = [
            tensor_to_display_image(dominant_images[row_idx]),
            tensor_to_display_image(average_images[row_idx]),
            tensor_to_display_image(recessive_images[row_idx]),
        ]

        for col_idx, image in enumerate(row_images):
            if image.size != (tile_size, tile_size):
                image = image.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
            x = x0 + col_idx * (tile_size + column_gap)
            canvas.paste(image, (x, y))

        footer = f"{rows_metadata[row_idx]['dominant_sample_id']} | {rows_metadata[row_idx]['recessive_sample_id']}"
        bbox = draw.textbbox((0, 0), footer, font=font)
        footer_x = x0 + (3 * tile_size + 2 * column_gap) / 2 - (bbox[2] - bbox[0]) / 2
        footer_y = y + tile_size + 2
        draw.text((footer_x, footer_y), footer, fill=(0, 0, 0), font=font)
        y += tile_size + footer_h + row_gap

    canvas.save(out_path)


def save_metadata(rows_metadata: list[dict], out_csv_path: str):
    fieldnames = [
        "row_idx",
        "dominant_sample_id",
        "recessive_sample_id",
        "dominant_source_path",
        "recessive_source_path",
        "dominant_relative_path",
        "recessive_relative_path",
        "dominant_idx",
        "recessive_idx",
        "dominant_norm",
        "recessive_norm",
    ]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_metadata:
            writer.writerow(row)


@torch.no_grad()
def main(args):
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    setup_logging(args.run_name)
    out_dir = os.path.join("experiments", args.run_name, "samples", "decode", f"val_average_grid_{args.num_pairs}")
    os.makedirs(out_dir, exist_ok=True)

    batch = collect_first_validation_batch(args)
    dominant = batch["dominant_embedding"].to(device)
    recessive = batch["recessive_embedding"].to(device)
    average = 0.5 * (dominant + recessive)

    mean, std = load_zscore_stats(args.dataset_root)
    mean = mean.to(device)
    std = std.to(device)

    dominant_unnorm = unnormalize_semantic(dominant, mean, std)
    average_unnorm = unnormalize_semantic(average, mean, std)
    recessive_unnorm = unnormalize_semantic(recessive, mean, std)

    autoencoder = load_diffae_autoencoder(args.diffae_checkpoint, device)
    dominant_images = decode_semantic_batch(autoencoder, dominant_unnorm)
    average_images = decode_semantic_batch(autoencoder, average_unnorm)
    recessive_images = decode_semantic_batch(autoencoder, recessive_unnorm)

    rows_metadata = []
    for i in range(args.num_pairs):
        rows_metadata.append(
            {
                "row_idx": i,
                "dominant_sample_id": batch["dominant_sample_id"][i],
                "recessive_sample_id": batch["recessive_sample_id"][i],
                "dominant_source_path": batch["dominant_source_path"][i],
                "recessive_source_path": batch["recessive_source_path"][i],
                "dominant_relative_path": batch.get("dominant_relative_path", [""] * args.num_pairs)[i],
                "recessive_relative_path": batch.get("recessive_relative_path", [""] * args.num_pairs)[i],
                "dominant_idx": int(batch["dominant_idx"][i].item()),
                "recessive_idx": int(batch["recessive_idx"][i].item()),
                "dominant_norm": float(batch["dominant_norm"][i].item()),
                "recessive_norm": float(batch["recessive_norm"][i].item()),
            }
        )

    grid_path = os.path.join(out_dir, f"val_average_grid_{args.num_pairs}.png")
    build_grid(
        dominant_images=dominant_images,
        average_images=average_images,
        recessive_images=recessive_images,
        rows_metadata=rows_metadata,
        out_path=grid_path,
        tile_size=args.tile_size,
    )

    metadata_path = os.path.join(out_dir, f"val_average_grid_{args.num_pairs}_metadata.csv")
    save_metadata(rows_metadata, metadata_path)

    tensors_path = os.path.join(out_dir, f"val_average_grid_{args.num_pairs}_decoded.pt")
    torch.save(
        {
            "dominant_images": dominant_images.detach().cpu(),
            "average_images": average_images.detach().cpu(),
            "recessive_images": recessive_images.detach().cpu(),
            "dominant_embedding": dominant.detach().cpu(),
            "average_embedding": average.detach().cpu(),
            "recessive_embedding": recessive.detach().cpu(),
            "rows_metadata": rows_metadata,
        },
        tensors_path,
    )

    print(f"Saved outputs to: {out_dir}")
    print(f"Grid image: {grid_path}")
    print(f"Metadata CSV: {metadata_path}")
    print(f"Decoded tensors: {tensors_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True, help="Experiment folder name under experiments/")
    parser.add_argument("--dataset_root", default="encoded_ffhq256_semantic_split", help="Folder containing semantic/train_zsem.pt and semantic/val_zsem.pt",)
    parser.add_argument("--diffae_checkpoint", required=True, help="Path to the DiffAE checkpoint used for semantic decoding")
    parser.add_argument("--num_pairs", type=int, default=20, help="Number of deterministic validation pairs to decode")
    parser.add_argument("--num_workers", type=int, default=0, help="Validation DataLoader worker count")
    parser.add_argument("--device", default=None, help="Device string, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size used in the saved grid")
    main(parser.parse_args())

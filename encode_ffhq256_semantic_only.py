import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from diffae.templates import ffhq256_autoenc
from diffae.config import PretrainConfig
from diffae.experiment import LitModel


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def build_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, limit: Optional[int] = None, image_size: int = 256):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Image folder not found: {self.root}")

        self.files = sorted(
            [p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        )
        if limit is not None:
            self.files = self.files[:limit]

        self.transform = build_transform(image_size=image_size)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        rel_path = path.relative_to(self.root)

        return {
            "img": img,
            "index": idx,
            "sample_id": path.stem,
            "source_path": str(path.resolve()),
            "relative_path": str(rel_path),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "img": torch.stack([b["img"] for b in batch], dim=0),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        "sample_id": [b["sample_id"] for b in batch],
        "source_path": [b["source_path"] for b in batch],
        "relative_path": [b["relative_path"] for b in batch],
    }


def load_model(checkpoint_path: str, device: str) -> LitModel:
    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(
        name="ffhq256_autoenc",
        path=checkpoint_path,
    )
    conf.latent_infer_path = None

    model = LitModel(conf)
    model = model.to(device)
    model.eval()
    model.ema_model.eval()
    return model


def make_split_indices(n: int, val_ratio: float, seed: int):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    n_val = min(n_val, n - 1) if n > 1 else n_val

    val_indices = sorted(indices[:n_val])
    train_indices = sorted(indices[n_val:])
    return train_indices, val_indices


def encode_split(
    model: LitModel,
    dataset: Dataset,
    split_name: str,
    split_indices: List[int],
    out_root: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    save_previews: bool,
    preview_count: int,
    t_invert: int,
    t_decode: int,
):
    subset = Subset(dataset, split_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    all_zsem: List[torch.Tensor] = []
    manifest_rows: List[Dict] = []
    preview_saved = 0

    preview_dir = out_root / "previews" / split_name
    preview_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        current_row = 0
        for batch in loader:
            imgs = batch["img"].to(device)
            z_sem = model.encode(imgs)
            all_zsem.append(z_sem.detach().cpu().to(torch.float32))

            for i in range(imgs.shape[0]):
                manifest_rows.append({
                    "split": split_name,
                    "dataset_index": int(batch["index"][i].item()),
                    "sample_id": batch["sample_id"][i],
                    "source_path": batch["source_path"][i],
                    "relative_path": batch["relative_path"][i],
                    "semantic_file": f"semantic/{split_name}_zsem.pt",
                    "semantic_row": current_row,
                })
                current_row += 1

            if save_previews and preview_saved < preview_count:
                n_prev = min(imgs.shape[0], preview_count - preview_saved)
                imgs_prev = imgs[:n_prev]
                z_prev = z_sem[:n_prev]

                xT_prev = model.encode_stochastic(imgs_prev, z_prev, T=t_invert)
                recon_prev = model.render(xT_prev, cond=z_prev, T=t_decode)

                for j in range(n_prev):
                    sample_id = batch["sample_id"][j]
                    save_image(
                        denorm(imgs_prev[j].detach().cpu()),
                        preview_dir / f"{sample_id}_orig.png",
                    )
                    save_image(
                        recon_prev[j].detach().cpu().clamp(0, 1),
                        preview_dir / f"{sample_id}_recon.png",
                    )
                    preview_saved += 1

    z_sem_all = torch.cat(all_zsem, dim=0) if all_zsem else torch.empty(0, 512)

    torch.save(
        {
            "split": split_name,
            "z_sem": z_sem_all,
            "sample_ids": [row["sample_id"] for row in manifest_rows],
            "source_paths": [row["source_path"] for row in manifest_rows],
            "relative_paths": [row["relative_path"] for row in manifest_rows],
        },
        out_root / "semantic" / f"{split_name}_zsem.pt",
    )

    return manifest_rows, z_sem_all.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="ffhq256_autoenc/last.ckpt")
    parser.add_argument("--out-root", type=str, default="encoded_ffhq256_semantic_split")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--preview-count", type=int, default=5)
    parser.add_argument("--t-invert", type=int, default=100)
    parser.add_argument("--t-decode", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out_root)
    (out_root / "semantic").mkdir(parents=True, exist_ok=True)
    (out_root / "previews").mkdir(parents=True, exist_ok=True)

    dataset = ImageFolderDataset(args.images_root, limit=args.limit, image_size=256)
    train_indices, val_indices = make_split_indices(
        n=len(dataset),
        val_ratio=args.val_ratio,
        seed=args.split_seed,
    )

    model = load_model(args.checkpoint, device)

    train_manifest, n_train = encode_split(
        model=model,
        dataset=dataset,
        split_name="train",
        split_indices=train_indices,
        out_root=out_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        save_previews=args.save_previews,
        preview_count=args.preview_count,
        t_invert=args.t_invert,
        t_decode=args.t_decode,
    )

    val_manifest, n_val = encode_split(
        model=model,
        dataset=dataset,
        split_name="val",
        split_indices=val_indices,
        out_root=out_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        save_previews=args.save_previews,
        preview_count=args.preview_count,
        t_invert=args.t_invert,
        t_decode=args.t_decode,
    )

    manifest_rows = train_manifest + val_manifest

    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "dataset_index",
                "sample_id",
                "source_path",
                "relative_path",
                "semantic_file",
                "semantic_row",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    meta = {
        "model_name": "ffhq256_autoenc",
        "checkpoint": args.checkpoint,
        "images_root": str(Path(args.images_root).resolve()),
        "num_images_total": len(dataset),
        "num_train": n_train,
        "num_val": n_val,
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "image_size": 256,
        "semantic_dtype": "float32",
        "normalization": "images normalized to [-1,1] with mean=std=0.5 per channel",
        "files": {
            "train_embeddings": "semantic/train_zsem.pt",
            "val_embeddings": "semantic/val_zsem.pt",
            "manifest": "manifest.csv",
        },
        "notes": [
            "Only semantic embeddings are stored.",
            "Stochastic code x_T should be computed on demand from the original image.",
            "semantic_row is the row inside the corresponding split file.",
        ],
    }

    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Saved semantic-only dataset with train/val split to: {out_root}")
    print(f"Train: {n_train} images")
    print(f"Val:   {n_val} images")


if __name__ == "__main__":
    main()
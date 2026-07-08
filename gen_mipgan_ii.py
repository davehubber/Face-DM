import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_MAD22_ROOT = "/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/MAD22/original_sorted"


class ImagePathDataset(Dataset):
    """Loads a fixed ordered list of image paths with the DiffAE FFHQ preprocessing."""

    def __init__(self, image_paths: Sequence[Path], image_size: int = 256):
        self.paths = list(image_paths)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        return {
            "img": self.transform(image),
            "index": index,
            "path": str(path),
            "filename": path.name,
            "stem": path.stem,
        }


def discover_images(root_dir: Path) -> List[Path]:
    return sorted(
        p for p in Path(root_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def build_stem_lookup(paths: Sequence[Path], label: str) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    duplicates = []

    for path in paths:
        if path.stem in lookup:
            duplicates.append((path.stem, lookup[path.stem], path))
        else:
            lookup[path.stem] = path

    if duplicates:
        examples = "\n".join(
            f"  {stem}: {first} <-> {second}"
            for stem, first, second in duplicates[:10]
        )
        raise ValueError(
            f"Duplicate filename stems found in {label}; source lookup would be ambiguous.\n{examples}"
        )

    return lookup


def parse_mad22_morph_filename(filename: str) -> Tuple[str, str]:
    """
    Parses MAD22 morph names such as:
        009_08-vs-144_08.jpg -> (009_08, 144_08)

    The first source is stored as the criminal identity, and the second source is
    stored as the accomplice identity.
    """
    stem = Path(filename).stem
    if "-vs-" not in stem:
        return "unknown", "unknown"

    criminal_stem, accomplice_stem = stem.split("-vs-", maxsplit=1)
    if not criminal_stem or not accomplice_stem:
        return "unknown", "unknown"

    return criminal_stem, accomplice_stem


def load_diffae_ffhq256_autoencoder(diffae_root: Path, checkpoint_path: Path, device: torch.device):
    """Loads the official DiffAE FFHQ256 autoencoder and returns it in evaluation mode."""
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

        model = LitModel(conf).to(device)
        model.eval()
        if hasattr(model, "ema_model"):
            model.ema_model.eval()

        for parameter in model.parameters():
            parameter.requires_grad_(False)
    finally:
        os.chdir(old_cwd)

    return model


@torch.no_grad()
def encode_paths(
    paths: Sequence[Path],
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_size: int,
    desc: str,
) -> Tuple[np.ndarray, List[dict]]:
    if len(paths) == 0:
        raise ValueError(f"No images were provided for: {desc}")

    dataset = ImagePathDataset(paths, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    embeddings = []
    rows = []

    for batch in tqdm(loader, desc=desc):
        images = batch["img"].to(device, non_blocking=True)
        z_sem = model.encode(images).detach().float().cpu().numpy()
        embeddings.append(z_sem)

        for local_i, idx in enumerate(batch["index"].tolist()):
            rows.append({
                "embedding_index": int(idx),
                "image_path": batch["path"][local_i],
                "filename": batch["filename"][local_i],
                "stem": batch["stem"][local_i],
            })

    return np.concatenate(embeddings, axis=0).astype(np.float32), rows


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_npy(path: Path, array: np.ndarray):
    if path.is_symlink():
        path.unlink()
    np.save(path, array.astype(np.float32))


def discover_valid_mipgan_ii_entries(mad22_root: Path, method: str):
    bonafide_root = mad22_root / "BonaFide"
    morph_root = mad22_root / method

    if not bonafide_root.exists():
        raise FileNotFoundError(f"MAD22 BonaFide folder not found: {bonafide_root}")
    if not morph_root.exists():
        raise FileNotFoundError(f"MAD22 {method} folder not found: {morph_root}")

    bonafide_paths = discover_images(bonafide_root)
    morph_paths = discover_images(morph_root)
    bonafide_lookup = build_stem_lookup(bonafide_paths, "MAD22/BonaFide")

    valid_entries = []
    invalid_rows = []
    missing_rows = []

    for morph_path in morph_paths:
        criminal_stem, accomplice_stem = parse_mad22_morph_filename(morph_path.name)

        if criminal_stem == "unknown" or accomplice_stem == "unknown":
            invalid_rows.append({
                "morph_path": str(morph_path),
                "morph_filename": morph_path.name,
                "reason": "Could not parse MAD22 morph filename",
            })
            continue

        missing_sources = []
        if criminal_stem not in bonafide_lookup:
            missing_sources.append(criminal_stem)
        if accomplice_stem not in bonafide_lookup:
            missing_sources.append(accomplice_stem)

        if missing_sources:
            missing_rows.append({
                "morph_path": str(morph_path),
                "morph_filename": morph_path.name,
                "criminal_stem": criminal_stem,
                "accomplice_stem": accomplice_stem,
                "missing_sources": ";".join(missing_sources),
            })
            continue

        valid_entries.append({
            "morph_path": morph_path,
            "morph_filename": morph_path.name,
            "method": method,
            "criminal_stem": criminal_stem,
            "accomplice_stem": accomplice_stem,
            "criminal_path": bonafide_lookup[criminal_stem],
            "accomplice_path": bonafide_lookup[accomplice_stem],
        })

    return valid_entries, invalid_rows, missing_rows, len(bonafide_paths), len(morph_paths)


def split_entries(entries: List[dict], train_ratio: float, seed: int):
    entries = list(entries)
    rng = random.Random(seed)
    rng.shuffle(entries)

    train_count = int(round(len(entries) * train_ratio))
    train_count = min(max(train_count, 0), len(entries))

    train_entries = entries[:train_count]
    test_entries = entries[train_count:]

    train_entries = sorted(train_entries, key=lambda row: row["morph_path"].name)
    test_entries = sorted(test_entries, key=lambda row: row["morph_path"].name)
    return train_entries, test_entries


def make_source_embedding_lookup(source_paths: Sequence[Path], source_embeddings: np.ndarray):
    return {
        path.stem: source_embeddings[index]
        for index, path in enumerate(source_paths)
    }


def write_split(
    out_dir: Path,
    split_name: str,
    entries: Sequence[dict],
    morph_embeddings: np.ndarray,
    source_embedding_lookup: Dict[str, np.ndarray],
    write_eval_aliases: bool,
):
    z_cri = np.stack([source_embedding_lookup[row["criminal_stem"]] for row in entries]).astype(np.float32)
    z_acc = np.stack([source_embedding_lookup[row["accomplice_stem"]] for row in entries]).astype(np.float32)

    save_npy(out_dir / f"{split_name}_morph_zsem.npy", morph_embeddings)
    save_npy(out_dir / f"{split_name}_z_cri.npy", z_cri)
    save_npy(out_dir / f"{split_name}_z_acc.npy", z_acc)

    metadata_rows = []
    for index, row in enumerate(entries):
        metadata_rows.append({
            "embedding_index": index,
            "split": split_name,
            "method": row["method"],
            "morph_path": str(row["morph_path"]),
            "morph_filename": row["morph_filename"],
            "criminal_stem": row["criminal_stem"],
            "accomplice_stem": row["accomplice_stem"],
            "criminal_bonafide_path": str(row["criminal_path"]),
            "accomplice_bonafide_path": str(row["accomplice_path"]),
        })

    fieldnames = [
        "embedding_index",
        "split",
        "method",
        "morph_path",
        "morph_filename",
        "criminal_stem",
        "accomplice_stem",
        "criminal_bonafide_path",
        "accomplice_bonafide_path",
    ]
    write_csv(out_dir / f"{split_name}_morph_metadata.csv", metadata_rows, fieldnames)

    # Compatibility with scripts that expect the previous eval_* naming.
    if write_eval_aliases and split_name == "test":
        save_npy(out_dir / "eval_morph_zsem.npy", morph_embeddings)
        save_npy(out_dir / "eval_z_cri.npy", z_cri)
        save_npy(out_dir / "eval_z_acc.npy", z_acc)
        write_csv(out_dir / "eval_morph_metadata.csv", metadata_rows, fieldnames)

    return {
        "morph_shape": list(morph_embeddings.shape),
        "z_cri_shape": list(z_cri.shape),
        "z_acc_shape": list(z_acc.shape),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a MAD22 MIPGAN_II DiffAE semantic dataset with an 80/20 morph-level split."
    )
    parser.add_argument("--mad22-root", type=str, default=DEFAULT_MAD22_ROOT)
    parser.add_argument("--method", type=str, default="MIPGAN_II")
    parser.add_argument("--out-dir", type=str, default="mipgan_ii_dataset")

    parser.add_argument("--diffae-root", type=str, default="/nas-ctm01/homes/dacordeiro/diffae/")
    parser.add_argument("--checkpoint", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt")

    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-eval-aliases",
        action="store_true",
        help="Do not write eval_* aliases for the test split.",
    )

    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1.")

    mad22_root = Path(args.mad22_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("\n--- Discovering MAD22 MIPGAN_II morphs and bona fides ---")
    entries, invalid_rows, missing_rows, bonafide_count, morph_count = discover_valid_mipgan_ii_entries(
        mad22_root=mad22_root,
        method=args.method,
    )

    if len(entries) == 0:
        raise ValueError("No valid morphs were found after filename parsing and source lookup.")

    train_entries, test_entries = split_entries(entries, args.train_ratio, args.seed)

    print(f"MAD22 root                    : {mad22_root}")
    print(f"Method                        : {args.method}")
    print(f"Bona fide images found         : {bonafide_count}")
    print(f"Morph images found             : {morph_count}")
    print(f"Valid morphs                   : {len(entries)}")
    print(f"Invalid morph filenames         : {len(invalid_rows)}")
    print(f"Morphs with missing sources     : {len(missing_rows)}")
    print(f"Train morphs                   : {len(train_entries)} ({len(train_entries) / len(entries) * 100:.2f}%)")
    print(f"Test morphs                    : {len(test_entries)} ({len(test_entries) / len(entries) * 100:.2f}%)")

    write_csv(
        out_dir / "invalid_morph_filenames.csv",
        invalid_rows,
        ["morph_path", "morph_filename", "reason"],
    )
    write_csv(
        out_dir / "missing_source_morphs.csv",
        missing_rows,
        ["morph_path", "morph_filename", "criminal_stem", "accomplice_stem", "missing_sources"],
    )

    print("\n--- Loading DiffAE model ---")
    model = load_diffae_ffhq256_autoencoder(Path(args.diffae_root), Path(args.checkpoint), device)
    print(f"Model loaded on {device}")

    train_morph_paths = [row["morph_path"] for row in train_entries]
    test_morph_paths = [row["morph_path"] for row in test_entries]

    all_source_paths = sorted(
        {row["criminal_path"] for row in entries} | {row["accomplice_path"] for row in entries},
        key=lambda path: path.stem,
    )

    print("\n--- Encoding morphs and source bona fides ---")
    train_morph_embeddings, _ = encode_paths(
        train_morph_paths,
        model,
        device,
        args.batch_size,
        args.num_workers,
        args.image_size,
        "Train MIPGAN_II morph z_sem",
    )
    test_morph_embeddings, _ = encode_paths(
        test_morph_paths,
        model,
        device,
        args.batch_size,
        args.num_workers,
        args.image_size,
        "Test MIPGAN_II morph z_sem",
    )
    source_embeddings, source_metadata_rows = encode_paths(
        all_source_paths,
        model,
        device,
        args.batch_size,
        args.num_workers,
        args.image_size,
        "Unique source bona fide z_sem",
    )

    save_npy(out_dir / "source_bonafide_zsem.npy", source_embeddings)
    write_csv(
        out_dir / "source_bonafide_metadata.csv",
        source_metadata_rows,
        ["embedding_index", "image_path", "filename", "stem"],
    )
    source_embedding_lookup = make_source_embedding_lookup(all_source_paths, source_embeddings)

    print("\n--- Writing row-aligned train/test arrays ---")
    train_shapes = write_split(
        out_dir=out_dir,
        split_name="train",
        entries=train_entries,
        morph_embeddings=train_morph_embeddings,
        source_embedding_lookup=source_embedding_lookup,
        write_eval_aliases=not args.no_eval_aliases,
    )
    test_shapes = write_split(
        out_dir=out_dir,
        split_name="test",
        entries=test_entries,
        morph_embeddings=test_morph_embeddings,
        source_embedding_lookup=source_embedding_lookup,
        write_eval_aliases=not args.no_eval_aliases,
    )

    manifest = {
        "mad22_root": str(mad22_root),
        "method": args.method,
        "split_type": "random_morph_level_split",
        "train_ratio": args.train_ratio,
        "test_ratio": 1.0 - args.train_ratio,
        "seed": args.seed,
        "output_dir": str(out_dir),
        "counts": {
            "bonafide_images_found": bonafide_count,
            "morph_images_found": morph_count,
            "valid_morphs": len(entries),
            "invalid_morph_filenames": len(invalid_rows),
            "morphs_with_missing_sources": len(missing_rows),
            "train_morphs": len(train_entries),
            "test_morphs": len(test_entries),
            "unique_source_bonafides_used": len(all_source_paths),
        },
        "shapes": {
            "train": train_shapes,
            "test": test_shapes,
            "source_bonafide_zsem": list(source_embeddings.shape),
        },
        "naming": {
            "z_cri": "semantic embedding of the first bona fide in the morph filename",
            "z_acc": "semantic embedding of the second bona fide in the morph filename",
            "eval_aliases": not args.no_eval_aliases,
        },
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("[SUCCESS] MAD22 MIPGAN_II DATASET CREATED")
    print("=" * 60)
    print(f"Output directory       : {out_dir}")
    print(f"Train morphs           : {len(train_entries)}")
    print(f"Test morphs            : {len(test_entries)}")
    print(f"Train morph shape      : {train_shapes['morph_shape']}")
    print(f"Test morph shape       : {test_shapes['morph_shape']}")
    print(f"Source bona fide shape : {list(source_embeddings.shape)}")
    if not args.no_eval_aliases:
        print("Eval aliases written   : eval_morph_zsem.npy, eval_z_cri.npy, eval_z_acc.npy, eval_morph_metadata.csv")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class MorphImageDataset(Dataset):
    """Flexible dataset to load images from a directory or a specific list of paths."""
    def __init__(self, image_paths: List[Path], image_size: int = 256):
        self.paths = sorted(image_paths)
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
    """Loads the official DiffAE FFHQ256 autoencoder model and puts it into evaluation mode."""
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


def parse_morph_filename(filename: str) -> tuple:
    """
    Parses a morph filename following the convention: morphed_img555005_img496816.png
    Returns just the stems: (img555005, img496816). This ignores file extensions.
    """
    path_obj = Path(filename)
    stem = path_obj.stem

    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "morphed":
        return parts[1], parts[2]
    return "unknown", "unknown"


def discover_images(root_dir: Path) -> List[Path]:
    """Finds all supported images under a given directory."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        p for p in Path(root_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def save_npy(path: Path, array: np.ndarray):
    if path.is_symlink():
        path.unlink()
    np.save(path, array)


def run_semantic_encoding(loader: DataLoader, model, device: torch.device, desc: str):
    """Helper loop to extract only z_sem embeddings."""
    embeddings = []
    metadata_rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs = batch["img"].to(device, non_blocking=True)
            z_sem = model.encode(imgs)
            z_sem = z_sem.detach().float().cpu().numpy()
            embeddings.append(z_sem)

            batch_indices = batch["index"].tolist()
            batch_paths = batch["path"]
            batch_filenames = batch["filename"]

            for local_i, idx in enumerate(batch_indices):
                metadata_rows.append({
                    "embedding_index": int(idx),
                    "image_path": batch_paths[local_i],
                    "filename": batch_filenames[local_i]
                })

    if not embeddings:
        raise ValueError(f"No images were encoded for: {desc}")

    concat_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    return concat_embeddings, metadata_rows


def write_bonafide_metadata(path: Path, metadata_rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename"])
        w.writeheader()
        w.writerows(metadata_rows)


def add_train_morph_metadata(raw_rows: list, train_bf_lookup: dict):
    metadata_rows = []

    for row in raw_rows:
        src_a_stem, src_b_stem = parse_morph_filename(row["filename"])
        row.update({
            "source_image_A_stem": src_a_stem,
            "source_image_B_stem": src_b_stem,
            "source_idx_A_in_train_bf": train_bf_lookup[src_a_stem],
            "source_idx_B_in_train_bf": train_bf_lookup[src_b_stem]
        })
        metadata_rows.append(row)

    return metadata_rows


def add_eval_morph_metadata(raw_rows: list, eval_bf_lookup: dict):
    metadata_rows = []

    for row in raw_rows:
        src_a_stem, src_b_stem = parse_morph_filename(row["filename"])
        row.update({
            "source_image_A_stem": src_a_stem,
            "source_image_B_stem": src_b_stem,
            "source_idx_A_in_eval_bf": eval_bf_lookup[src_a_stem],
            "source_idx_B_in_eval_bf": eval_bf_lookup[src_b_stem]
        })
        metadata_rows.append(row)

    return metadata_rows


def write_train_morph_metadata(path: Path, metadata_rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename",
                                         "source_image_A_stem", "source_image_B_stem",
                                         "source_idx_A_in_train_bf", "source_idx_B_in_train_bf"])
        w.writeheader()
        w.writerows(metadata_rows)


def write_eval_morph_metadata(path: Path, metadata_rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename",
                                         "source_image_A_stem", "source_image_B_stem",
                                         "source_idx_A_in_eval_bf", "source_idx_B_in_eval_bf"])
        w.writeheader()
        w.writerows(metadata_rows)


def main():
    parser = argparse.ArgumentParser(description="DiffAE Morphing Dataset Encoder Pipeline")

    # Critical Environment and Model Paths
    parser.add_argument("--diffae-root", type=str, default="/nas-ctm01/homes/dacordeiro/diffae/")
    parser.add_argument("--checkpoint", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt")
    parser.add_argument("--out-dir", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings")

    # Dataset Input Folders
    parser.add_argument("--train-bonafide-root", type=str, default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/os25k_m_t/", help="Path to bona fide folder")
    parser.add_argument("--train-morph-root", type=str, default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/m15k_t/", help="Path to morph folder")

    # Operational Hyperparameters
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic identity split")
    parser.add_argument("--reserved-identities", type=int, default=600, help="Number of bona fide source identities to reserve for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Paths setup
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---------------------------------------------------------
    # STEP 1: Discover Bona Fides and Morphs
    # ---------------------------------------------------------
    print("\n--- Discovering Bona Fides and Morphs ---")
    all_bf_paths = discover_images(Path(args.train_bonafide_root))
    bf_path_lookup = {p.stem: p for p in all_bf_paths}

    all_morph_paths = discover_images(Path(args.train_morph_root))

    valid_morph_entries = []
    invalid_morph_paths = []
    missing_source_morph_paths = []
    all_source_stems = set()

    for p in all_morph_paths:
        src_a_stem, src_b_stem = parse_morph_filename(p.name)

        if src_a_stem == "unknown" or src_b_stem == "unknown":
            invalid_morph_paths.append(p)
            continue

        if src_a_stem not in bf_path_lookup or src_b_stem not in bf_path_lookup:
            missing_source_morph_paths.append(p)
            continue

        valid_morph_entries.append((p, src_a_stem, src_b_stem))
        all_source_stems.add(src_a_stem)
        all_source_stems.add(src_b_stem)

    all_source_stems = sorted(all_source_stems)

    if args.reserved_identities > len(all_source_stems):
        raise ValueError(
            f"Requested {args.reserved_identities} reserved identities, "
            f"but only found {len(all_source_stems)} valid source identities."
        )

    rng = random.Random(args.seed)
    reserved_stems = set(rng.sample(all_source_stems, args.reserved_identities))
    remaining_stems = set(all_source_stems) - reserved_stems

    eval_morph_paths = []
    train_morph_paths = []
    mixed_morph_paths = []

    for morph_path, src_a_stem, src_b_stem in valid_morph_entries:
        src_a_reserved = src_a_stem in reserved_stems
        src_b_reserved = src_b_stem in reserved_stems

        if src_a_reserved and src_b_reserved:
            eval_morph_paths.append(morph_path)
        elif not src_a_reserved and not src_b_reserved:
            train_morph_paths.append(morph_path)
        else:
            mixed_morph_paths.append(morph_path)

    if len(eval_morph_paths) == 0:
        raise ValueError("The reserved identity group produced zero fully contained eval morphs. Increase --reserved-identities or change --seed.")

    if len(train_morph_paths) == 0:
        raise ValueError("The remaining identity group produced zero fully contained train morphs. Decrease --reserved-identities or change --seed.")

    train_bf_paths = [bf_path_lookup[stem] for stem in sorted(remaining_stems)]
    eval_bf_paths = [bf_path_lookup[stem] for stem in sorted(reserved_stems)]

    print(f"Total bona fide files found: {len(all_bf_paths)}")
    print(f"Total morph files found: {len(all_morph_paths)}")
    print(f"Invalid morph filenames: {len(invalid_morph_paths)}")
    print(f"Morphs with missing source bona fides: {len(missing_source_morph_paths)}")
    print(f"Valid morphs with both sources available: {len(valid_morph_entries)}")
    print(f"Unique valid source identities: {len(all_source_stems)}")
    print(f"Reserved eval identities: {len(reserved_stems)}")
    print(f"Remaining train identities: {len(remaining_stems)}")
    print(f"Train morphs fully inside remaining identities: {len(train_morph_paths)}")
    print(f"Eval morphs fully inside reserved identities: {len(eval_morph_paths)}")
    print(f"Discarded mixed train/eval morphs: {len(mixed_morph_paths)}")
    print(f"Sanity check morph total: {len(train_morph_paths) + len(eval_morph_paths) + len(mixed_morph_paths)}")

    # ---------------------------------------------------------
    # STEP 2: Initialize Model
    # ---------------------------------------------------------
    print("\n--- Initializing Model Pipeline ---")
    model = load_diffae_ffhq256_autoencoder(Path(args.diffae_root), Path(args.checkpoint), device)
    print(f"Model successfully loaded on target device: {device}")

    # ---------------------------------------------------------
    # STEP 3: Process Train Bona Fides
    # ---------------------------------------------------------
    print("\n--- Processing Train Bona Fide Split ---")
    train_bf_dataset = MorphImageDataset(train_bf_paths)
    train_bf_loader = DataLoader(train_bf_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    train_bf_embs, train_bf_meta = run_semantic_encoding(train_bf_loader, model, device, "Train Bona Fide z_sem")
    save_npy(out_dir / "train_bonafide_zsem.npy", train_bf_embs)
    write_bonafide_metadata(out_dir / "train_bonafide_metadata.csv", train_bf_meta)
    train_bf_lookup = {Path(row["filename"]).stem: row["embedding_index"] for row in train_bf_meta}

    # ---------------------------------------------------------
    # STEP 4: Process Eval Bona Fides
    # ---------------------------------------------------------
    print("\n--- Processing Eval Bona Fide Split ---")
    eval_bf_dataset = MorphImageDataset(eval_bf_paths)
    eval_bf_loader = DataLoader(eval_bf_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    eval_bf_embs, eval_bf_meta = run_semantic_encoding(eval_bf_loader, model, device, "Eval Bona Fide z_sem")
    save_npy(out_dir / "eval_bonafide_zsem.npy", eval_bf_embs)
    write_bonafide_metadata(out_dir / "eval_bonafide_metadata.csv", eval_bf_meta)
    eval_bf_lookup = {Path(row["filename"]).stem: row["embedding_index"] for row in eval_bf_meta}

    # ---------------------------------------------------------
    # STEP 5: Process Train Morphs
    # ---------------------------------------------------------
    print("\n--- Processing Train Morph Split ---")
    train_morph_dataset = MorphImageDataset(train_morph_paths)
    train_morph_loader = DataLoader(train_morph_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    train_morph_embs, train_morph_meta_raw = run_semantic_encoding(train_morph_loader, model, device, "Train Morph z_sem")
    save_npy(out_dir / "train_morph_zsem.npy", train_morph_embs)

    train_morph_meta = add_train_morph_metadata(train_morph_meta_raw, train_bf_lookup)
    write_train_morph_metadata(out_dir / "train_morph_metadata.csv", train_morph_meta)

    # ---------------------------------------------------------
    # STEP 6: Process Eval Morphs
    # ---------------------------------------------------------
    print("\n--- Processing Eval Morph Split ---")
    eval_morph_dataset = MorphImageDataset(eval_morph_paths)
    eval_morph_loader = DataLoader(eval_morph_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    eval_morph_embs, eval_morph_meta_raw = run_semantic_encoding(eval_morph_loader, model, device, "Eval Morph z_sem")
    save_npy(out_dir / "eval_morph_zsem.npy", eval_morph_embs)

    eval_morph_meta = add_eval_morph_metadata(eval_morph_meta_raw, eval_bf_lookup)
    write_eval_morph_metadata(out_dir / "eval_morph_metadata.csv", eval_morph_meta)

    print("\n" + "="*50)
    print(" [SUCCESS] PIPELINE PROCESSING COMPLETED")
    print("="*50)
    print(f" -> Train BF Embeddings Shape        : {train_bf_embs.shape}")
    print(f" -> Eval BF Embeddings Shape         : {eval_bf_embs.shape}")
    print(f" -> Train Morph Embeddings Shape     : {train_morph_embs.shape}")
    print(f" -> Eval Morph Embeddings Shape      : {eval_morph_embs.shape}")
    print(f" -> Discarded Mixed Morphs           : {len(mixed_morph_paths)}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

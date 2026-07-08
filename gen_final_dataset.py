import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAD22_METHODS = ["FaceMorpher_cropped", "MIPGAN_I_cropped", "MIPGAN_II_cropped", "OpenCV_cropped", "Webmorph_cropped"]


class MorphImageDataset(Dataset):
    """Loads a fixed list of image paths and applies the DiffAE FFHQ preprocessing."""

    def __init__(self, image_paths: Sequence[Path], image_size: int = 256):
        self.paths = list(image_paths)
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


def discover_images(root_dir: Path) -> List[Path]:
    """Finds all supported images under a given directory."""
    root_dir = Path(root_dir)
    return sorted(
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def build_stem_lookup(paths: Sequence[Path], label: str) -> Dict[str, Path]:
    """Builds a filename-stem lookup and fails early if duplicate stems are found."""
    lookup: Dict[str, Path] = {}
    duplicates = []

    for path in paths:
        stem = path.stem
        if stem in lookup:
            duplicates.append((stem, lookup[stem], path))
        else:
            lookup[stem] = path

    if duplicates:
        examples = "\n".join(
            f"  {stem}: {first}  <->  {second}"
            for stem, first, second in duplicates[:10]
        )
        raise ValueError(
            f"Duplicate bona fide filename stems found in {label}. "
            f"The source lookup would be ambiguous. Examples:\n{examples}"
        )

    return lookup


def parse_smdd_morph_filename(filename: str) -> Tuple[str, str]:
    """
    Parses SMDD-style names used by the previous pipeline, e.g.:
        morphed_img555005_img496816.png -> (img555005, img496816)

    The first returned source is treated as z_cri and the second as z_acc.
    """
    stem = Path(filename).stem

    if not stem.startswith("morphed_"):
        return "unknown", "unknown"

    parts = stem[len("morphed_"):].split("_")
    if len(parts) < 2:
        return "unknown", "unknown"

    return parts[0], parts[1]


def parse_mad22_morph_filename(filename: str) -> Tuple[str, str]:
    """
    Parses MAD22 names, e.g.:
        009_08-vs-144_08.jpg -> (009_08, 144_08)

    The first returned source is treated as z_cri and the second as z_acc.
    """
    stem = Path(filename).stem

    if "-vs-" not in stem:
        return "unknown", "unknown"

    first, second = stem.split("-vs-", maxsplit=1)
    if not first or not second:
        return "unknown", "unknown"

    return first, second


def save_npy(path: Path, array: np.ndarray):
    if path.is_symlink():
        path.unlink()
    np.save(path, array.astype(np.float32))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_semantic_encoding(
    paths: Sequence[Path],
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_size: int,
    desc: str,
):
    """Extracts z_sem embeddings for a fixed ordered list of paths."""
    if len(paths) == 0:
        raise ValueError(f"No images were provided for: {desc}")

    dataset = MorphImageDataset(paths, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

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
                    "filename": batch_filenames[local_i],
                    "stem": Path(batch_filenames[local_i]).stem,
                })

    concat_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    return concat_embeddings, metadata_rows


def discover_smdd_train_entries(smdd_morph_root: Path, smdd_bf_lookup: Dict[str, Path]):
    valid_entries = []
    invalid_rows = []
    missing_rows = []

    for morph_path in discover_images(smdd_morph_root):
        src_cri_stem, src_acc_stem = parse_smdd_morph_filename(morph_path.name)

        if src_cri_stem == "unknown" or src_acc_stem == "unknown":
            invalid_rows.append({
                "morph_path": str(morph_path),
                "morph_filename": morph_path.name,
                "reason": "Could not parse SMDD morph filename",
            })
            continue

        missing_sources = [
            stem for stem in (src_cri_stem, src_acc_stem)
            if stem not in smdd_bf_lookup
        ]
        if missing_sources:
            missing_rows.append({
                "morph_path": str(morph_path),
                "morph_filename": morph_path.name,
                "source_cri_stem": src_cri_stem,
                "source_acc_stem": src_acc_stem,
                "missing_sources": ";".join(missing_sources),
            })
            continue

        valid_entries.append({
            "dataset": "SMDD",
            "method": "SMDD",
            "morph_path": morph_path,
            "morph_filename": morph_path.name,
            "source_cri_stem": src_cri_stem,
            "source_acc_stem": src_acc_stem,
            "source_cri_path": smdd_bf_lookup[src_cri_stem],
            "source_acc_path": smdd_bf_lookup[src_acc_stem],
        })

    return valid_entries, invalid_rows, missing_rows


def discover_mad22_eval_entries(
    mad22_root: Path,
    mad22_bf_lookup: Dict[str, Path],
    methods: Sequence[str],
):
    valid_entries = []
    invalid_rows = []
    missing_rows = []

    for method in methods:
        method_dir = mad22_root / method
        if not method_dir.exists():
            raise FileNotFoundError(f"MAD22 method folder not found: {method_dir}")

        for morph_path in discover_images(method_dir):
            src_cri_stem, src_acc_stem = parse_mad22_morph_filename(morph_path.name)

            if src_cri_stem == "unknown" or src_acc_stem == "unknown":
                invalid_rows.append({
                    "method": method,
                    "morph_path": str(morph_path),
                    "morph_filename": morph_path.name,
                    "reason": "Could not parse MAD22 morph filename",
                })
                continue

            missing_sources = [
                stem for stem in (src_cri_stem, src_acc_stem)
                if stem not in mad22_bf_lookup
            ]
            if missing_sources:
                missing_rows.append({
                    "method": method,
                    "morph_path": str(morph_path),
                    "morph_filename": morph_path.name,
                    "source_cri_stem": src_cri_stem,
                    "source_acc_stem": src_acc_stem,
                    "missing_sources": ";".join(missing_sources),
                })
                continue

            valid_entries.append({
                "dataset": "MAD22",
                "method": method,
                "morph_path": morph_path,
                "morph_filename": morph_path.name,
                "source_cri_stem": src_cri_stem,
                "source_acc_stem": src_acc_stem,
                "source_cri_path": mad22_bf_lookup[src_cri_stem],
                "source_acc_path": mad22_bf_lookup[src_acc_stem],
            })

    return valid_entries, invalid_rows, missing_rows


def build_aligned_source_embeddings_and_metadata(
    entries: Sequence[dict],
    morph_meta_raw: Sequence[dict],
    bonafide_embeddings: np.ndarray,
    bonafide_meta: Sequence[dict],
    split_name: str,
):
    """
    Builds z_cri and z_acc arrays aligned with the morph embedding array.

    For row i:
      morph_zsem[i] corresponds to metadata row i.
      z_cri[i] is the first source bona fide in the morph filename.
      z_acc[i] is the second source bona fide in the morph filename.
    """
    bf_index_by_stem = {
        row["stem"]: int(row["embedding_index"])
        for row in bonafide_meta
    }
    bf_meta_by_stem = {
        row["stem"]: row
        for row in bonafide_meta
    }

    z_cri_rows = []
    z_acc_rows = []
    metadata_rows = []

    if len(entries) != len(morph_meta_raw):
        raise ValueError(
            f"Internal alignment error for {split_name}: "
            f"{len(entries)} entries but {len(morph_meta_raw)} encoded morph rows."
        )

    for i, (entry, morph_row) in enumerate(zip(entries, morph_meta_raw)):
        encoded_path = Path(morph_row["image_path"])
        if encoded_path != entry["morph_path"]:
            raise ValueError(
                f"Internal path alignment error for {split_name} at row {i}: "
                f"metadata has {encoded_path}, entry has {entry['morph_path']}"
            )

        cri_idx = bf_index_by_stem[entry["source_cri_stem"]]
        acc_idx = bf_index_by_stem[entry["source_acc_stem"]]
        cri_meta = bf_meta_by_stem[entry["source_cri_stem"]]
        acc_meta = bf_meta_by_stem[entry["source_acc_stem"]]

        z_cri_rows.append(bonafide_embeddings[cri_idx])
        z_acc_rows.append(bonafide_embeddings[acc_idx])

        metadata_rows.append({
            "embedding_index": i,
            "morph_embedding_index": i,
            "z_cri_index": i,
            "z_acc_index": i,
            "dataset": entry["dataset"],
            "method": entry["method"],
            "morph_path": str(entry["morph_path"]),
            "morph_filename": entry["morph_filename"],
            "source_cri_stem": entry["source_cri_stem"],
            "source_acc_stem": entry["source_acc_stem"],
            "source_cri_path": str(entry["source_cri_path"]),
            "source_acc_path": str(entry["source_acc_path"]),
            "source_cri_filename": cri_meta["filename"],
            "source_acc_filename": acc_meta["filename"],
            f"source_cri_idx_in_{split_name}_bonafide": cri_idx,
            f"source_acc_idx_in_{split_name}_bonafide": acc_idx,
        })

    z_cri = np.stack(z_cri_rows, axis=0).astype(np.float32)
    z_acc = np.stack(z_acc_rows, axis=0).astype(np.float32)
    return z_cri, z_acc, metadata_rows


def write_manifest(out_dir: Path):
    manifest = f"""Final DiffAE semantic morph dataset
===================================

Each split has three row-aligned embedding arrays:

  train_morph_zsem.npy  -> z_sem of the SMDD training morph image
  train_z_cri.npy       -> z_sem of the first bona fide source in the SMDD morph filename
  train_z_acc.npy       -> z_sem of the second bona fide source in the SMDD morph filename

  eval_morph_zsem.npy   -> z_sem of the MAD22 evaluation morph image
  eval_z_cri.npy        -> z_sem of the first bona fide source in the MAD22 morph filename
  eval_z_acc.npy        -> z_sem of the second bona fide source in the MAD22 morph filename

For row i, morph_zsem[i], z_cri[i], z_acc[i], and the corresponding metadata row i belong to the same morph sample.

MAD22 morph metadata includes the method column with one of:
  {', '.join(MAD22_METHODS)}

Additional unique bona fide embedding files are stored for traceability:
  train_bonafide_zsem.npy / train_bonafide_metadata.csv
  eval_bonafide_zsem.npy  / eval_bonafide_metadata.csv
"""
    (out_dir / "README_final_dataset.txt").write_text(manifest, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Encode SMDD training morphs and the full MAD22 evaluation set with DiffAE z_sem embeddings."
    )

    # Model paths
    parser.add_argument("--diffae-root", type=str, default="/nas-ctm01/homes/dacordeiro/diffae/")
    parser.add_argument("--checkpoint", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt")

    # Output folder: intentionally simple name requested by the user.
    parser.add_argument("--out-dir", type=str, default="final_dataset")

    # SMDD training input folders.
    parser.add_argument(
        "--smdd-bonafide-root",
        type=str,
        default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/os25k_m_t_cropped/",
        help="Path to the SMDD bona fide folder used to resolve morph source images.",
    )
    parser.add_argument(
        "--smdd-morph-root",
        type=str,
        default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/m15k_t_cropped/",
        help="Path to the SMDD training morph folder. All valid morphs here are used for training.",
    )

    # MAD22 evaluation input folders.
    parser.add_argument(
        "--mad22-root",
        type=str,
        default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/MAD22/original_sorted",
        help="Path to MAD22/original_sorted, containing BonaFide and the method folders.",
    )
    parser.add_argument(
        "--mad22-bonafide-subdir",
        type=str,
        default="BonaFide",
        help="Name of the MAD22 bona fide subfolder inside --mad22-root.",
    )
    parser.add_argument(
        "--mad22-methods",
        nargs="+",
        default=MAD22_METHODS,
        help="MAD22 method folders to include for evaluation.",
    )

    # Operational hyperparameters.
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    smdd_bonafide_root = Path(args.smdd_bonafide_root)
    smdd_morph_root = Path(args.smdd_morph_root)
    mad22_root = Path(args.mad22_root)
    mad22_bonafide_root = mad22_root / args.mad22_bonafide_subdir

    print("\n--- Discovering SMDD training data ---")
    smdd_bf_paths = discover_images(smdd_bonafide_root)
    if len(smdd_bf_paths) == 0:
        raise ValueError(f"No SMDD bona fide images found in: {smdd_bonafide_root}")
    smdd_bf_lookup = build_stem_lookup(smdd_bf_paths, "SMDD bona fide folder")

    train_entries, train_invalid, train_missing = discover_smdd_train_entries(smdd_morph_root, smdd_bf_lookup)
    if len(train_entries) == 0:
        raise ValueError("No valid SMDD training morphs were found.")

    print(f"SMDD bona fide images found      : {len(smdd_bf_paths)}")
    print(f"SMDD valid training morphs       : {len(train_entries)}")
    print(f"SMDD invalid morph filenames     : {len(train_invalid)}")
    print(f"SMDD morphs with missing sources : {len(train_missing)}")

    print("\n--- Discovering MAD22 evaluation data ---")
    mad22_bf_paths = discover_images(mad22_bonafide_root)
    if len(mad22_bf_paths) == 0:
        raise ValueError(f"No MAD22 bona fide images found in: {mad22_bonafide_root}")
    mad22_bf_lookup = build_stem_lookup(mad22_bf_paths, "MAD22 bona fide folder")

    eval_entries, eval_invalid, eval_missing = discover_mad22_eval_entries(
        mad22_root=mad22_root,
        mad22_bf_lookup=mad22_bf_lookup,
        methods=args.mad22_methods,
    )
    if len(eval_entries) == 0:
        raise ValueError("No valid MAD22 evaluation morphs were found.")

    eval_counts_by_method = {}
    for entry in eval_entries:
        eval_counts_by_method[entry["method"]] = eval_counts_by_method.get(entry["method"], 0) + 1

    print(f"MAD22 bona fide images found     : {len(mad22_bf_paths)}")
    print(f"MAD22 valid evaluation morphs    : {len(eval_entries)}")
    for method in args.mad22_methods:
        print(f"  {method:<12}: {eval_counts_by_method.get(method, 0)}")
    print(f"MAD22 invalid morph filenames    : {len(eval_invalid)}")
    print(f"MAD22 morphs with missing sources: {len(eval_missing)}")

    # Write diagnostic files before encoding, so path/parsing issues are visible even if encoding later fails.
    write_csv(out_dir / "skipped_train_invalid_filenames.csv", train_invalid, ["morph_path", "morph_filename", "reason"])
    write_csv(out_dir / "skipped_train_missing_sources.csv", train_missing, ["morph_path", "morph_filename", "source_cri_stem", "source_acc_stem", "missing_sources"])
    write_csv(out_dir / "skipped_eval_invalid_filenames.csv", eval_invalid, ["method", "morph_path", "morph_filename", "reason"])
    write_csv(out_dir / "skipped_eval_missing_sources.csv", eval_missing, ["method", "morph_path", "morph_filename", "source_cri_stem", "source_acc_stem", "missing_sources"])

    print("\n--- Initializing DiffAE model ---")
    model = load_diffae_ffhq256_autoencoder(Path(args.diffae_root), Path(args.checkpoint), device)
    print(f"Model successfully loaded on: {device}")

    print("\n--- Encoding SMDD bona fides ---")
    train_bf_embs, train_bf_meta = run_semantic_encoding(
        paths=smdd_bf_paths,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        desc="SMDD bona fide z_sem",
    )
    save_npy(out_dir / "train_bonafide_zsem.npy", train_bf_embs)
    write_csv(out_dir / "train_bonafide_metadata.csv", train_bf_meta, ["embedding_index", "image_path", "filename", "stem"])

    print("\n--- Encoding SMDD training morphs ---")
    train_morph_paths = [entry["morph_path"] for entry in train_entries]
    train_morph_embs, train_morph_meta_raw = run_semantic_encoding(
        paths=train_morph_paths,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        desc="SMDD training morph z_sem",
    )
    train_z_cri, train_z_acc, train_morph_meta = build_aligned_source_embeddings_and_metadata(
        entries=train_entries,
        morph_meta_raw=train_morph_meta_raw,
        bonafide_embeddings=train_bf_embs,
        bonafide_meta=train_bf_meta,
        split_name="train",
    )
    save_npy(out_dir / "train_morph_zsem.npy", train_morph_embs)
    save_npy(out_dir / "train_z_cri.npy", train_z_cri)
    save_npy(out_dir / "train_z_acc.npy", train_z_acc)
    train_morph_fields = [
        "embedding_index", "morph_embedding_index", "z_cri_index", "z_acc_index",
        "dataset", "method", "morph_path", "morph_filename",
        "source_cri_stem", "source_acc_stem", "source_cri_path", "source_acc_path",
        "source_cri_filename", "source_acc_filename",
        "source_cri_idx_in_train_bonafide", "source_acc_idx_in_train_bonafide",
    ]
    write_csv(out_dir / "train_morph_metadata.csv", train_morph_meta, train_morph_fields)

    print("\n--- Encoding MAD22 bona fides ---")
    eval_bf_embs, eval_bf_meta = run_semantic_encoding(
        paths=mad22_bf_paths,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        desc="MAD22 bona fide z_sem",
    )
    save_npy(out_dir / "eval_bonafide_zsem.npy", eval_bf_embs)
    write_csv(out_dir / "eval_bonafide_metadata.csv", eval_bf_meta, ["embedding_index", "image_path", "filename", "stem"])

    print("\n--- Encoding MAD22 evaluation morphs ---")
    eval_morph_paths = [entry["morph_path"] for entry in eval_entries]
    eval_morph_embs, eval_morph_meta_raw = run_semantic_encoding(
        paths=eval_morph_paths,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        desc="MAD22 evaluation morph z_sem",
    )
    eval_z_cri, eval_z_acc, eval_morph_meta = build_aligned_source_embeddings_and_metadata(
        entries=eval_entries,
        morph_meta_raw=eval_morph_meta_raw,
        bonafide_embeddings=eval_bf_embs,
        bonafide_meta=eval_bf_meta,
        split_name="eval",
    )
    save_npy(out_dir / "eval_morph_zsem.npy", eval_morph_embs)
    save_npy(out_dir / "eval_z_cri.npy", eval_z_cri)
    save_npy(out_dir / "eval_z_acc.npy", eval_z_acc)
    eval_morph_fields = [
        "embedding_index", "morph_embedding_index", "z_cri_index", "z_acc_index",
        "dataset", "method", "morph_path", "morph_filename",
        "source_cri_stem", "source_acc_stem", "source_cri_path", "source_acc_path",
        "source_cri_filename", "source_acc_filename",
        "source_cri_idx_in_eval_bonafide", "source_acc_idx_in_eval_bonafide",
    ]
    write_csv(out_dir / "eval_morph_metadata.csv", eval_morph_meta, eval_morph_fields)

    write_manifest(out_dir)

    print("\n" + "=" * 72)
    print("[SUCCESS] FINAL DATASET ENCODING COMPLETED")
    print("=" * 72)
    print(f"Output folder                  : {out_dir}")
    print(f"Train morph embeddings         : {train_morph_embs.shape}")
    print(f"Train z_cri embeddings         : {train_z_cri.shape}")
    print(f"Train z_acc embeddings         : {train_z_acc.shape}")
    print(f"Train unique bona fide embeds  : {train_bf_embs.shape}")
    print(f"Eval morph embeddings          : {eval_morph_embs.shape}")
    print(f"Eval z_cri embeddings          : {eval_z_cri.shape}")
    print(f"Eval z_acc embeddings          : {eval_z_acc.shape}")
    print(f"Eval unique bona fide embeds   : {eval_bf_embs.shape}")
    print(f"Skipped train invalid/missing  : {len(train_invalid)} / {len(train_missing)}")
    print(f"Skipped eval invalid/missing   : {len(eval_invalid)} / {len(eval_missing)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

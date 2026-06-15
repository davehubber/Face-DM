import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

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
    Returns (source_a_filename, source_b_filename) assuming matching file extensions.
    """
    path_obj = Path(filename)
    stem = path_obj.stem
    ext = path_obj.suffix

    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "morphed":
        return f"{parts[1]}{ext}", f"{parts[2]}{ext}"
    return "unknown", "unknown"


def discover_images(root_dir: Path) -> List[Path]:
    """Finds all supported images under a given directory."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        p for p in Path(root_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


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

    concat_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
    return concat_embeddings, metadata_rows


def main():
    parser = argparse.ArgumentParser(description="DiffAE Morphing Dataset Encoder Pipeline")

    # Critical Environment and Model Paths
    parser.add_argument("--diffae-root", type=str, default="/nas-ctm01/homes/dacordeiro/diffae/")
    parser.add_argument("--checkpoint", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt")
    parser.add_argument("--out-dir", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings")

    # Dataset Input Folders
    parser.add_argument("--train-bonafide-root", type=str, default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/os25k_m_t/", help="Path to training bona fide folder")
    parser.add_argument("--train-morph-root", type=str, default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/m15k_t/", help="Path to training morph folder")
    parser.add_argument("--eval-bonafide-root", type=str, default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD_eval/os25k_bf_e_crop224/", help="Path to evaluation bona fide folder")
    parser.add_argument("--eval-morph-root", type=str, default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD_eval/m15k_e_crop224/", help="Path to evaluation morph folder")

    # Operational Hyperparameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic selection of eval morph subset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Paths setup
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("\n--- Initializing Model Pipeline ---")
    model = load_diffae_ffhq256_autoencoder(Path(args.diffae_root), Path(args.checkpoint), device)
    print(f"Model successfully loaded on target device: {device}")

    # ---------------------------------------------------------
    # STEP 1: Process Train Bona Fide
    # ---------------------------------------------------------
    print("\n--- Processing Train Bona Fide Dataset ---")
    train_bf_paths = discover_images(Path(args.train_bonafide_root))
    train_bf_dataset = MorphImageDataset(train_bf_paths)
    train_bf_loader = DataLoader(train_bf_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    train_bf_embs, train_bf_meta = run_semantic_encoding(train_bf_loader, model, device, "Train Bona Fide z_sem")
    np.save(out_dir / "train_bonafide_zsem.npy", train_bf_embs)

    # Map filenames to their embedding indices to facilitate relational lookup maps
    train_bf_lookup = {row["filename"]: row["embedding_index"] for row in train_bf_meta}

    with open(out_dir / "train_bonafide_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename"])
        w.writeheader()
        w.writerows(train_bf_meta)

    # ---------------------------------------------------------
    # STEP 2: Process Train Morph (With Relational Indexing)
    # ---------------------------------------------------------
    print("\n--- Processing Train Morph Dataset ---")
    train_morph_paths = discover_images(Path(args.train_morph_root))
    train_morph_dataset = MorphImageDataset(train_morph_paths)
    train_morph_loader = DataLoader(train_morph_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    train_morph_embs, train_morph_meta_raw = run_semantic_encoding(train_morph_loader, model, device, "Train Morph z_sem")
    np.save(out_dir / "train_morph_zsem.npy", train_morph_embs)

    # Enhance morph metadata with direct source structural connections
    train_morph_meta = []
    for row in train_morph_meta_raw:
        src_a, src_b = parse_morph_filename(row["filename"])
        row.update({
            "source_image_A": src_a,
            "source_image_B": src_b,
            "source_idx_A_in_train_bf": train_bf_lookup.get(src_a, -1),
            "source_idx_B_in_train_bf": train_bf_lookup.get(src_b, -1)
        })
        train_morph_meta.append(row)

    with open(out_dir / "train_morph_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename",
                                         "source_image_A", "source_image_B",
                                         "source_idx_A_in_train_bf", "source_idx_B_in_train_bf"])
        w.writeheader()
        w.writerows(train_morph_meta)

    # ---------------------------------------------------------
    # STEP 3: Process Evaluation Bona Fide
    # ---------------------------------------------------------
    print("\n--- Processing Evaluation Bona Fide Dataset ---")
    eval_bf_paths = discover_images(Path(args.eval_bonafide_root))
    eval_bf_dataset = MorphImageDataset(eval_bf_paths)
    eval_bf_loader = DataLoader(eval_bf_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    eval_bf_embs, eval_bf_meta = run_semantic_encoding(eval_bf_loader, model, device, "Eval Bona Fide z_sem")
    np.save(out_dir / "eval_bonafide_zsem.npy", eval_bf_embs)

    eval_bf_lookup = {row["filename"]: row["embedding_index"] for row in eval_bf_meta}

    with open(out_dir / "eval_bonafide_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename"])
        w.writeheader()
        w.writerows(eval_bf_meta)

    # ---------------------------------------------------------
    # STEP 4: Subset Evaluation Morph Setup (Deterministic Selection)
    # ---------------------------------------------------------
    print("\n--- Subsetting and Sampling Evaluation Morphs ---")
    all_eval_morph_paths = discover_images(Path(args.eval_morph_root))
    print(f"Total available evaluation morphs found: {len(all_eval_morph_paths)}")

    if len(all_eval_morph_paths) < 1000:
        raise ValueError(f"Requested 1000 validation items, but folder only contains {len(all_eval_morph_paths)} files.")

    np.random.seed(args.seed)
    selected_indices = np.random.choice(len(all_eval_morph_paths), size=1000, replace=False)
    selected_morph_paths = [all_eval_morph_paths[i] for i in sorted(selected_indices)]

    eval_morph_dataset = MorphImageDataset(selected_morph_paths)
    eval_morph_loader = DataLoader(eval_morph_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # ---------------------------------------------------------
    # STEP 5: Dual Extraction Loop for Selected Evaluation Morphs (z_sem + x_T)
    # ---------------------------------------------------------
    eval_morph_zsem_list = []
    eval_morph_xt_list = []
    eval_morph_meta = []

    with torch.no_grad():
        for batch in tqdm(eval_morph_loader, desc="Encoding Eval Morph (z_sem + x_T)"):
            imgs = batch["img"].to(device, non_blocking=True)

            # Extract Semantic Codes
            z_sem = model.encode(imgs)
            z_sem_np = z_sem.detach().float().cpu().numpy()
            eval_morph_zsem_list.append(z_sem_np)

            # Extract Stochastic Noise-Like Codes (Horizon sequence T=250)
            xT = model.encode_stochastic(imgs, z_sem, T=250)
            xT_np = xT.detach().float().cpu().numpy()
            eval_morph_xt_list.append(xT_np)

            batch_indices = batch["index"].tolist()
            batch_paths = batch["path"]
            batch_filenames = batch["filename"]

            for local_i, idx in enumerate(batch_indices):
                filename = batch_filenames[local_i]
                src_a, src_b = parse_morph_filename(filename)

                eval_morph_meta.append({
                    "embedding_index": int(idx),
                    "image_path": batch_paths[local_i],
                    "filename": filename,
                    "source_image_A": src_a,
                    "source_image_B": src_b,
                    "source_idx_A_in_eval_bf": eval_bf_lookup.get(src_a, -1),
                    "source_idx_B_in_eval_bf": eval_bf_lookup.get(src_b, -1)
                })

    # Consolidate and Export Eval Morph Subsets
    eval_morph_zsem_final = np.concatenate(eval_morph_zsem_list, axis=0).astype(np.float32)
    eval_morph_xt_final = np.concatenate(eval_morph_xt_list, axis=0).astype(np.float32)

    np.save(out_dir / "eval_morph_zsem_1000.npy", eval_morph_zsem_final)
    np.save(out_dir / "eval_morph_xt_1000.npy", eval_morph_xt_final)

    with open(out_dir / "eval_morph_metadata_1000.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename",
                                         "source_image_A", "source_image_B",
                                         "source_idx_A_in_eval_bf", "source_idx_B_in_eval_bf"])
        w.writeheader()
        w.writerows(eval_morph_meta)

    # ---------------------------------------------------------
    # Final Structure Overview Display
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" [SUCCESS] PIPELINE PROCESSING COMPLETED")
    print("="*50)
    print(f"Artifacts output directory: {out_dir}")
    print(f" -> Train Bona Fide Embeddings Shape : {train_bf_embs.shape}")
    print(f" -> Train Morph Embeddings Shape     : {train_morph_embs.shape}")
    print(f" -> Eval Bona Fide Embeddings Shape  : {eval_bf_embs.shape}")
    print(f" -> Eval Morph Subset z_sem Shape    : {eval_morph_zsem_final.shape}")
    print(f" -> Eval Morph Subset x_T Shape      : {eval_morph_xt_final.shape}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

import argparse
import csv
import os
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
    parser.add_argument("--out-dir", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings_v1")

    # Dataset Input Folders (We only need the training folders now)
    parser.add_argument("--train-bonafide-root", type=str, default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/os25k_m_t/", help="Path to bona fide folder")
    parser.add_argument("--train-morph-root", type=str, default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/m15k_t/", help="Path to morph folder")

    # Operational Hyperparameters
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic selection of eval morph subset")
    parser.add_argument("--eval-size", type=int, default=1000, help="Number of morphs to reserve for evaluation")
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
    # STEP 1: Process Bona Fides (The Single Truth Array)
    # ---------------------------------------------------------
    print("\n--- Processing Bona Fide Dataset ---")
    bf_paths = discover_images(Path(args.train_bonafide_root))
    bf_dataset = MorphImageDataset(bf_paths)
    bf_loader = DataLoader(bf_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    bf_embs, bf_meta = run_semantic_encoding(bf_loader, model, device, "Bona Fide z_sem")
    np.save(out_dir / "train_bonafide_zsem.npy", bf_embs)
    
    # Symlink the eval to train so the training script doesn't need to be changed
    # (Since train and eval now share the same bona fides)
    eval_bf_path = out_dir / "eval_bonafide_zsem.npy"
    if not eval_bf_path.exists():
        os.symlink(out_dir / "train_bonafide_zsem.npy", eval_bf_path)

    # Use the STEM (no extensions) to create a bulletproof lookup dictionary
    bf_lookup = {Path(row["filename"]).stem: row["embedding_index"] for row in bf_meta}

    with open(out_dir / "train_bonafide_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename"])
        w.writeheader()
        w.writerows(bf_meta)

    # ---------------------------------------------------------
    # STEP 2: Pre-filter and Split Morphs
    # ---------------------------------------------------------
    print("\n--- Discovering and Filtering Morphs ---")
    all_morph_paths = discover_images(Path(args.train_morph_root))
    
    # Filter: Only keep morphs where BOTH source images are found in our bf_lookup
    valid_morph_paths = []
    for p in all_morph_paths:
        src_a_stem, src_b_stem = parse_morph_filename(p.name)
        if src_a_stem in bf_lookup and src_b_stem in bf_lookup:
            valid_morph_paths.append(p)
            
    print(f"Total morphs found: {len(all_morph_paths)}")
    print(f"Valid morphs (both sources exist): {len(valid_morph_paths)}")
    
    if len(valid_morph_paths) < args.eval_size:
        raise ValueError(f"Not enough valid morphs ({len(valid_morph_paths)}) to create a {args.eval_size} eval split.")

    # Deterministic Split
    np.random.seed(args.seed)
    np.random.shuffle(valid_morph_paths)
    eval_morph_paths = valid_morph_paths[:args.eval_size]
    train_morph_paths = valid_morph_paths[args.eval_size:]
    
    print(f"Allocating {len(train_morph_paths)} to Train, {len(eval_morph_paths)} to Eval.")

    # ---------------------------------------------------------
    # STEP 3: Process Train Morphs
    # ---------------------------------------------------------
    print("\n--- Processing Train Morph Split ---")
    train_morph_dataset = MorphImageDataset(train_morph_paths)
    train_morph_loader = DataLoader(train_morph_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    train_morph_embs, train_morph_meta_raw = run_semantic_encoding(train_morph_loader, model, device, "Train Morph z_sem")
    np.save(out_dir / "train_morph_zsem.npy", train_morph_embs)

    train_morph_meta = []
    for row in train_morph_meta_raw:
        src_a_stem, src_b_stem = parse_morph_filename(row["filename"])
        row.update({
            "source_image_A_stem": src_a_stem,
            "source_image_B_stem": src_b_stem,
            "source_idx_A_in_train_bf": bf_lookup[src_a_stem],
            "source_idx_B_in_train_bf": bf_lookup[src_b_stem]
        })
        train_morph_meta.append(row)

    with open(out_dir / "train_morph_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename",
                                         "source_image_A_stem", "source_image_B_stem",
                                         "source_idx_A_in_train_bf", "source_idx_B_in_train_bf"])
        w.writeheader()
        w.writerows(train_morph_meta)

    # ---------------------------------------------------------
    # STEP 4: Process Eval Morphs (z_sem + x_T)
    # ---------------------------------------------------------
    print(f"\n--- Processing Eval Morph Split ({args.eval_size} items) ---")
    eval_morph_dataset = MorphImageDataset(eval_morph_paths)
    eval_morph_loader = DataLoader(eval_morph_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    eval_morph_zsem_list = []
    eval_morph_xt_list = []
    eval_morph_meta = []

    with torch.no_grad():
        for batch in tqdm(eval_morph_loader, desc="Encoding Eval Morph (z_sem + x_T)"):
            imgs = batch["img"].to(device, non_blocking=True)

            z_sem = model.encode(imgs)
            z_sem_np = z_sem.detach().float().cpu().numpy()
            eval_morph_zsem_list.append(z_sem_np)

            # Stochastic Noise-Like Codes (T=250)
            xT = model.encode_stochastic(imgs, z_sem, T=250)
            xT_np = xT.detach().float().cpu().numpy()
            eval_morph_xt_list.append(xT_np)

            batch_indices = batch["index"].tolist()
            batch_paths = batch["path"]
            batch_filenames = batch["filename"]

            for local_i, idx in enumerate(batch_indices):
                filename = batch_filenames[local_i]
                src_a_stem, src_b_stem = parse_morph_filename(filename)

                eval_morph_meta.append({
                    "embedding_index": int(idx),
                    "image_path": batch_paths[local_i],
                    "filename": filename,
                    "source_image_A_stem": src_a_stem,
                    "source_image_B_stem": src_b_stem,
                    "source_idx_A_in_train_bf": bf_lookup[src_a_stem],
                    "source_idx_B_in_train_bf": bf_lookup[src_b_stem]
                })

    eval_morph_zsem_final = np.concatenate(eval_morph_zsem_list, axis=0).astype(np.float32)
    eval_morph_xt_final = np.concatenate(eval_morph_xt_list, axis=0).astype(np.float32)

    np.save(out_dir / f"eval_morph_zsem_{args.eval_size}.npy", eval_morph_zsem_final)
    np.save(out_dir / f"eval_morph_xt_{args.eval_size}.npy", eval_morph_xt_final)

    with open(out_dir / f"eval_morph_metadata_{args.eval_size}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename",
                                         "source_image_A_stem", "source_image_B_stem",
                                         "source_idx_A_in_train_bf", "source_idx_B_in_train_bf"])
        w.writeheader()
        w.writerows(eval_morph_meta)

    print("\n" + "="*50)
    print(" [SUCCESS] PIPELINE PROCESSING COMPLETED")
    print("="*50)
    print(f" -> Train BF Embeddings Shape        : {bf_embs.shape}")
    print(f" -> Train Morph Embeddings Shape     : {train_morph_embs.shape}")
    print(f" -> Eval Morph Subset z_sem Shape    : {eval_morph_zsem_final.shape}")
    print(f" -> Eval Morph Subset x_T Shape      : {eval_morph_xt_final.shape}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()

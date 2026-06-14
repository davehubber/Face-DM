import os
import sys
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers import StableUnCLIPImg2ImgPipeline

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
IMAGE_ROOT = Path("/nas-ctm01/datasets/public/ffhq256/")
OUT_DIR = Path("/nas-ctm01/homes/dacordeiro/Face-DM/unclip_embeddings/")
LIMIT = None         # Set to an integer for testing/debugging
BATCH_SIZE = 64      # Adjust based on your GPU VRAM (64 is usually safe for 16GB+ cards)
NUM_WORKERS = 4      # Number of CPU workers parallelizing image I/O and resizing


class UnCLIPEncodingDataset(Dataset):
    """Parallelized worker dataset for fast CPU image pre-loading and resizing."""
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            # Downstream unCLIP 2.1 operates at 768x768
            img = img.resize((768, 768), Image.Resampling.LANCZOS)
            return {
                "img": img, 
                "image_path": str(img_path), 
                "filename": img_path.name, 
                "error": None
            }
        except Exception as e:
            return {
                "img": None, 
                "image_path": str(img_path), 
                "filename": img_path.name, 
                "error": str(e)
            }


def fast_collate_fn(batch):
    """Passes lists downstream without force-stacking into rigid tensors."""
    return batch


def encode_unclip_dataset_batched(image_root: Path, out_dir: Path, limit=None, batch_size=64, num_workers=4):
    """Extracts unCLIP embeddings using multi-worker prefetching and batched CUDA execution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = sorted(
        list(image_root.rglob("*.png")) +
        list(image_root.rglob("*.jpg")) +
        list(image_root.rglob("*.jpeg"))
    )
    if limit is not None:
        image_paths = image_paths[:limit]

    print(f"[1/3] Found {len(image_paths)} files. Initializing parallelized DataLoader (BS={batch_size})...")
    
    dataset = UnCLIPEncodingDataset(image_paths)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        collate_fn=fast_collate_fn
    )

    print("Loading Stable unCLIP 2.1 pipeline onto GPU...")
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-1-unclip",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    embeddings = []
    norms = []
    metadata_rows = []

    for batch in tqdm(dataloader, desc="Extracting Batched L2 Embeddings"):
        # Isolate entries that failed during disk access/decoding
        valid_samples = [x for x in batch if x["error"] is None]
        for x in batch:
            if x["error"] is not None:
                print(f"\nSkipping {x['filename']} due to error: {x['error']}")

        if not valid_samples:
            continue

        # Extract images list for the batch feature extractor
        imgs_list = [x["img"] for x in valid_samples]

        with torch.no_grad():
            # Process entire image list through CLIP feature extractor simultaneously
            pixel_values = pipe.feature_extractor(images=imgs_list, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device="cuda", dtype=pipe.image_encoder.dtype)

            # Batched inference over the GPU matrix
            clip_embeddings = pipe.image_encoder(pixel_values).image_embeds
            embs_np = clip_embeddings.cpu().numpy().astype(np.float32)

        # Vectorized calculations per batch element
        for i, sample in enumerate(valid_samples):
            emb = embs_np[i]
            norm = np.linalg.norm(emb)
            
            # Compute safe L2 normalization
            emb_l2 = emb if (norm == 0 or not np.isfinite(norm)) else emb / norm

            embeddings.append(emb_l2)
            norms.append(norm)
            metadata_rows.append({
                "embedding_index": len(embeddings) - 1,
                "image_path": sample["image_path"],
                "filename": sample["filename"],
                "raw_norm": float(norm)
            })

    embeddings = np.stack(embeddings, axis=0).astype(np.float32)
    norms = np.array(norms, dtype=np.float32)
    
    master_npy_path = out_dir / "ffhq256_unclip_zsem.npy"
    master_norms_path = out_dir / "ffhq256_unclip_zsem_norms.npy"
    master_csv_path = out_dir / "ffhq256_unclip_zsem_metadata.csv"

    np.save(master_npy_path, embeddings)
    np.save(master_norms_path, norms)
    
    with open(master_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename", "raw_norm"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Master batched matrices saved: {embeddings.shape}")
    return master_npy_path, master_norms_path, master_csv_path


def joint_split_unclip(npy_path: Path, norms_path: Path, csv_path: Path, out_dir: Path):
    """Splits master dataset allocating a flat array of exactly 1000 single test embeddings."""
    print("\n[2/3] Slicing splits into individual flat structures (1000 single test items)...")
    unclip_embs = np.load(npy_path).astype(np.float32)
    unclip_norms = np.load(norms_path).astype(np.float32)
    
    with open(csv_path, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    num_samples = len(unclip_embs)
    num_test = 1000 if num_samples >= 1100 else int(num_samples * 0.2)
    if num_test == 0 and num_samples > 1:
        num_test = 1

    np.random.seed(42)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    test_idx = indices[:num_test]
    train_idx = indices[num_test:]

    print(f"Split completed: {len(train_idx)} train arrays, {num_test} flat test arrays.")

    # Save Train Split
    np.save(out_dir / "ffhq256_unclip_zsem_train.npy", unclip_embs[train_idx])
    np.save(out_dir / "ffhq256_unclip_zsem_train_norms.npy", unclip_norms[train_idx])

    # Compute Z-Score Statistics on Normalized Train Split
    train_mean = np.mean(unclip_embs[train_idx], axis=0) if len(train_idx) > 0 else np.zeros(unclip_embs.shape[1])
    train_std = np.std(unclip_embs[train_idx], axis=0) if len(train_idx) > 0 else np.ones(unclip_embs.shape[1])
    train_std = np.where(train_std == 0, 1.0, train_std)
    np.save(out_dir / "ffhq256_unclip_zsem_train_mean.npy", train_mean)
    np.save(out_dir / "ffhq256_unclip_zsem_train_std.npy", train_std)

    # Save Train Metadata
    with open(out_dir / "ffhq256_unclip_zsem_train_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename", "raw_norm"])
        writer.writeheader()
        for idx in train_idx:
            writer.writerow(metadata_rows[idx])

    # Save Flat Test Split
    np.save(out_dir / "ffhq256_unclip_zsem_test.npy", unclip_embs[test_idx])
    np.save(out_dir / "ffhq256_unclip_zsem_test_norms.npy", unclip_norms[test_idx])

    # Save Test Metadata
    with open(out_dir / "ffhq256_unclip_zsem_test_metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename", "raw_norm"])
        writer.writeheader()
        for idx in test_idx:
            writer.writerow(metadata_rows[idx])


def verify_and_decode_grid(out_dir: Path, output_grid_name="comparacao_unclip_flat_batched.jpg"):
    """Decodes the first 5 individual test embeddings by denormalizing their vectors."""
    print("\n[3/3] Verifying pipeline: Reconstructing first 5 flat test embeddings...")
    
    test_embs = np.load(out_dir / "ffhq256_unclip_zsem_test.npy")
    test_norms = np.load(out_dir / "ffhq256_unclip_zsem_test_norms.npy")
    test_metadata_path = out_dir / "ffhq256_unclip_zsem_test_metadata.csv"
    
    with open(test_metadata_path, "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-1-unclip",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=False)
    generator = torch.Generator(device="cuda").manual_seed(0)

    num_rows = min(5, len(test_embs))
    target_dim = 256
    
    grid_width = target_dim * 2
    grid_height = target_dim * num_rows
    grid_image = Image.new("RGB", (grid_width, grid_height))

    for idx in range(num_rows):
        row_meta = metadata_rows[idx]
        orig_img = Image.open(Path(row_meta["image_path"])).convert("RGB")
        input_768 = orig_img.resize((768, 768), Image.Resampling.LANCZOS)

        # Scale back up using the parallel norm storage array to enable pipeline parsing
        raw_emb = test_embs[idx] * test_norms[idx]
        emb_tensor = torch.from_numpy(raw_emb).unsqueeze(0).to("cuda", dtype=torch.float16)

        with torch.no_grad():
            decoded_img = pipe(
                image=input_768,
                image_embeds=emb_tensor,
                prompt="",
                noise_level=0,
                num_inference_steps=50,
                guidance_scale=1.0,
                generator=generator,
                height=768,
                width=768
            ).images[0]

        vis_orig = orig_img.resize((target_dim, target_dim), Image.Resampling.LANCZOS)
        vis_dec = decoded_img.resize((target_dim, target_dim), Image.Resampling.LANCZOS)

        y_offset = idx * target_dim
        grid_image.paste(vis_orig, (0, y_offset))
        grid_image.paste(vis_dec, (target_dim, y_offset))

    grid_out_path = out_dir / output_grid_name
    grid_image.save(grid_out_path)
    print(f"\n[SUCCESS] Matrix saved to: {grid_out_path}")


if __name__ == "__main__":
    master_npy, master_norms, master_csv = encode_unclip_dataset_batched(
        IMAGE_ROOT, OUT_DIR, limit=LIMIT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    joint_split_unclip(master_npy, master_norms, master_csv, OUT_DIR)
    verify_and_decode_grid(OUT_DIR)

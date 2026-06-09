import argparse, csv, json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from diffusers import StableUnCLIPImg2ImgPipeline

class ImageFolder(Dataset):
    def __init__(self, image_root: str, limit: Optional[int] = None):
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        self.paths = sorted(p for p in Path(image_root).resolve().rglob("*") if p.is_file() and p.suffix.lower() in exts)
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError(f"No images found in {image_root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return {"image": image, "index": i, "path": str(path), "filename": path.name}

def collate(batch):
    return {"images": [b["image"] for b in batch], "index": [b["index"] for b in batch], "path": [b["path"] for b in batch], "filename": [b["filename"] for b in batch]}

def make_splits(n: int, test_size: int = 1000, seed: int = 42):
    if n <= test_size:
        raise ValueError(f"Need more than {test_size} images, got {n}.")
    np.random.seed(seed)
    indices = np.arange(n)
    np.random.shuffle(indices)
    test_idx_A, train_idx = indices[:test_size], indices[test_size:]
    rng = np.random.default_rng(seed)
    test_idx_B = test_idx_A.copy()
    while True:
        rng.shuffle(test_idx_B)
        if np.all(test_idx_A != test_idx_B):
            return train_idx, test_idx_A, test_idx_B

def save_metadata(path: Path, rows, indices):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["embedding_index", "image_path", "filename"])
        writer.writeheader()
        for i in indices:
            writer.writerow(rows[int(i)])

def save_pair_metadata(path: Path, rows, idx_A, idx_B):
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = ["pair_index", "embedding_index_A", "image_path_A", "filename_A", "embedding_index_B", "image_path_B", "filename_B"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pair_i, (a, b) in enumerate(zip(idx_A, idx_B)):
            ra, rb = rows[int(a)], rows[int(b)]
            writer.writerow({"pair_index": pair_i, "embedding_index_A": ra["embedding_index"], "image_path_A": ra["image_path"], "filename_A": ra["filename"], "embedding_index_B": rb["embedding_index"], "image_path_B": rb["image_path"], "filename_B": rb["filename"]})

def save_split_arrays(out_dir: Path, prefix: str, name: str, arr, train_idx, test_idx_A, test_idx_B):
    np.save(out_dir / f"{prefix}_{name}.npy", arr)
    np.save(out_dir / f"{prefix}_{name}_train.npy", arr[train_idx])
    np.save(out_dir / f"{prefix}_{name}_test_pairs.npy", np.stack([arr[test_idx_A], arr[test_idx_B]], axis=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default="/nas-ctm01/datasets/public/AFHQ/resized/train/wild/")
    parser.add_argument("--out-dir", required="/nas-ctm01/homes/dacordeiro/Face-DM/stableunclip_embeddings/AFHQ")
    parser.add_argument("--prefix", default="stableunclip")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1-unclip")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--noise-level", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    dataset = ImageFolder(args.image_root, args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, drop_last=False)
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.image_encoder.to(device).eval()
    pipe.image_normalizer.to(device)

    rows, clip_l2, clip_norm, decoder_embeds = [], [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Encoding Stable unCLIP embeddings"):
            x = pipe.feature_extractor(images=batch["images"], return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
            raw = pipe.image_encoder(x).image_embeds
            raw_f32 = raw.float()
            norm = raw_f32.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            prepared = pipe.noise_image_embeddings(raw, noise_level=args.noise_level, noise=torch.zeros_like(raw))
            clip_l2.append(F.normalize(raw_f32, p=2, dim=-1).cpu().numpy())
            clip_norm.append(norm.cpu().numpy())
            decoder_embeds.append(prepared.float().cpu().numpy())
            rows.extend({"embedding_index": int(i), "image_path": p, "filename": f} for i, p, f in zip(batch["index"], batch["path"], batch["filename"]))

    clip_l2 = np.concatenate(clip_l2).astype(np.float32)
    clip_norm = np.concatenate(clip_norm).astype(np.float32)
    decoder_embeds = np.concatenate(decoder_embeds).astype(np.float32)
    train_idx, test_idx_A, test_idx_B = make_splits(len(dataset), args.test_size, seed=42)

    save_split_arrays(out_dir, args.prefix, "clip_l2", clip_l2, train_idx, test_idx_A, test_idx_B)
    save_split_arrays(out_dir, args.prefix, "clip_norm", clip_norm, train_idx, test_idx_A, test_idx_B)
    save_split_arrays(out_dir, args.prefix, f"decoder_noise{args.noise_level}", decoder_embeds, train_idx, test_idx_A, test_idx_B)
    np.savez(out_dir / f"{args.prefix}_split_indices.npz", train_idx=train_idx, test_idx_A=test_idx_A, test_idx_B=test_idx_B)
    save_metadata(out_dir / f"{args.prefix}_metadata.csv", rows, np.arange(len(rows)))
    save_metadata(out_dir / f"{args.prefix}_train_metadata.csv", rows, train_idx)
    save_pair_metadata(out_dir / f"{args.prefix}_test_pairs_metadata.csv", rows, test_idx_A, test_idx_B)

    config = {"model_id": args.model_id, "noise_level": args.noise_level, "seed": 42, "test_size": args.test_size, "clip_l2_shape": list(clip_l2.shape), "clip_norm_shape": list(clip_norm.shape), "decoder_embeds_shape": list(decoder_embeds.shape), "raw_clip_reconstruction": "raw_clip = clip_l2 * clip_norm", "decoder_usage": "Pass decoder_noise*.npy rows as image_embeds to StableUnCLIPImg2ImgPipeline; do not pass clip_l2 directly.", "note": "decoder_noise*.npy already contains the unCLIP image conditioning format with the noise-level embedding appended."}
    with open(out_dir / f"{args.prefix}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved to: {out_dir}")
    print(f"Images: {len(dataset)} | Train: {len(train_idx)} | Test pairs: {len(test_idx_A)}")
    print(f"clip_l2: {clip_l2.shape} | clip_norm: {clip_norm.shape} | decoder_embeds: {decoder_embeds.shape}")

if __name__ == "__main__":
    main()
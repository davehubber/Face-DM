import argparse, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from diffusers import StableUnCLIPImg2ImgPipeline

def load_rgb(path, size):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return img.resize(size, Image.Resampling.LANCZOS)

def make_grid(rows, labels, cell_size, out_path):
    w, h = cell_size
    label_h = 28
    grid = Image.new("RGB", (len(labels) * w, label_h + len(rows) * h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    for j, label in enumerate(labels):
        draw.text((j * w + 8, 6), label, fill=(0, 0, 0))
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            grid.paste(img, (j * w, label_h + i * h))
    grid.save(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pairs-clip-l2", default="stableunclip_embeddings/AFHQ/stableunclip_clip_l2_test_pairs.npy", help="Path to *_clip_l2_test_pairs.npy")
    parser.add_argument("--test-pairs-clip-norm", default="stableunclip_embeddings/AFHQ/stableunclip_clip_norm_test_pairs.npy", help="Path to *_clip_norm_test_pairs.npy")
    parser.add_argument("--test-pairs-metadata", default="stableunclip_embeddings/AFHQ/stableunclip_test_pairs_metadata.csv", help="Path to *_test_pairs_metadata.csv")
    parser.add_argument("--out-path", default="stableunclip_embeddings/AFHQ/verify_first10_pairs.png", help="Path to output grid image")
    parser.add_argument("--model-id", default="sd2-community/stable-diffusion-2-1-unclip")
    parser.add_argument("--num-pairs", type=int, default=10)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--noise-level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    clip_l2 = np.load(args.test_pairs_clip_l2).astype(np.float32)
    clip_norm = np.load(args.test_pairs_clip_norm).astype(np.float32)
    embeds = clip_l2 * clip_norm

    with open(args.test_pairs_metadata, "r", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))

    n = min(args.num_pairs, len(meta), embeds.shape[0])
    if clip_l2.ndim != 3 or clip_l2.shape[1] != 2:
        raise ValueError(f"Expected clip_l2 shape [N,2,D], got {clip_l2.shape}")
    if clip_norm.shape[:2] != clip_l2.shape[:2]:
        raise ValueError(f"clip_norm shape {clip_norm.shape} does not match clip_l2 shape {clip_l2.shape}")

    device = torch.device(args.device)
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(args.model_id, torch_dtype=dtype).to(device)

    rows = []
    for i in range(n):
        row = meta[i]
        orig_a = load_rgb(row["image_path_A"], (args.width, args.height))
        orig_b = load_rgb(row["image_path_B"], (args.width, args.height))

        emb_a = torch.from_numpy(embeds[i, 0:1]).to(device=device, dtype=dtype)
        emb_b = torch.from_numpy(embeds[i, 1:2]).to(device=device, dtype=dtype)

        gen_a = torch.Generator(device=device).manual_seed(args.seed + 2 * i)
        gen_b = torch.Generator(device=device).manual_seed(args.seed + 2 * i + 1)

        dec_a = pipe(image=None, image_embeds=emb_a, prompt="", height=args.height, width=args.width, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, noise_level=args.noise_level, generator=gen_a).images[0]
        dec_b = pipe(image=None, image_embeds=emb_b, prompt="", height=args.height, width=args.width, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, noise_level=args.noise_level, generator=gen_b).images[0]

        rows.append([orig_a, orig_b, dec_a, dec_b])
        print(f"Decoded pair {i + 1}/{n}")

    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_grid(rows, ["original_A", "original_B", "decoded_A", "decoded_B"], (args.width, args.height), out_path)
    print(f"Saved verification grid to: {out_path}")

if __name__ == "__main__":
    main()

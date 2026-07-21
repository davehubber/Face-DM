import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import StableUnCLIPImg2ImgPipeline


# ==========================================
# CONFIG
# ==========================================
OUT_DIR = Path("/nas-ctm01/homes/dacordeiro/Face-DM/unclip_embeddings/")
MODEL_ID = "sd2-community/stable-diffusion-2-1-unclip"
OUTPUT_GRID_NAME = "comparacao_unclip_flat_batched.jpg"
NUM_ROWS = 5
VIS_SIZE = 256
GEN_HEIGHT = 768
GEN_WIDTH = 768
SEED = 0


def decode_saved_unclip_grid(
    out_dir: Path,
    model_id: str = MODEL_ID,
    output_grid_name: str = OUTPUT_GRID_NAME,
    num_rows: int = NUM_ROWS,
    vis_size: int = VIS_SIZE,
    gen_height: int = GEN_HEIGHT,
    gen_width: int = GEN_WIDTH,
    seed: int = SEED,
):
    print("Loading saved test embeddings, norms, and metadata...")
    test_embs = np.load(out_dir / "ffhq256_unclip_zsem_test.npy").astype(np.float32)
    test_norms = np.load(out_dir / "ffhq256_unclip_zsem_test_norms.npy").astype(np.float32)

    with open(out_dir / "ffhq256_unclip_zsem_test_metadata.csv", "r", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    num_rows = min(num_rows, len(test_embs), len(metadata_rows))
    if num_rows == 0:
        raise ValueError("No test embeddings found to decode.")

    print("Loading Stable unCLIP pipeline...")
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    grid = Image.new("RGB", (vis_size * 2, vis_size * num_rows))

    for idx in range(num_rows):
        row_meta = metadata_rows[idx]

        # Load original image only for visual comparison in the grid
        orig_img = Image.open(Path(row_meta["image_path"])).convert("RGB")

        # Reconstruct raw embedding from stored L2-normalized embedding and saved norm
        raw_emb = test_embs[idx] * test_norms[idx]
        emb_tensor = torch.from_numpy(raw_emb).unsqueeze(0).to("cuda", dtype=torch.float16)

        print(f"Decoding row {idx + 1}/{num_rows}: {row_meta['filename']}")

        with torch.no_grad():
            decoded_img = pipe(
                image_embeds=emb_tensor,   # IMPORTANT: only image_embeds, not image=
                prompt="",
                noise_level=0,
                num_inference_steps=50,
                guidance_scale=1.0,
                generator=generator,
                height=gen_height,
                width=gen_width
            ).images[0]

        vis_orig = orig_img.resize((vis_size, vis_size), Image.Resampling.LANCZOS)
        vis_dec = decoded_img.resize((vis_size, vis_size), Image.Resampling.LANCZOS)

        y = idx * vis_size
        grid.paste(vis_orig, (0, y))
        grid.paste(vis_dec, (vis_size, y))

    out_path = out_dir / output_grid_name
    grid.save(out_path)
    print(f"\nSuccess! Comparison grid saved to: {out_path}")


if __name__ == "__main__":
    decode_saved_unclip_grid(OUT_DIR)
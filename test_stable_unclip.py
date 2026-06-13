import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from PIL import Image

def generate_comparison_grid(img_path, output_path="reconstruction_grid.jpg"):
    # 1. Load original image
    try:
        original_img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"Erro: Imagem não encontrada no caminho {img_path}")
        return

    # Stable unCLIP 2.1 is naturally used at 768 resolution
    original_img = original_img.resize((768, 768), Image.Resampling.LANCZOS)

    # 2. Load the actual Stable unCLIP img2img pipeline
    print("A carregar o modelo Stable unCLIP 2.1 img2img...")
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-1-unclip",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)

    # Optional, but useful if you want reproducible comparisons
    generator = torch.Generator(device="cuda").manual_seed(0)

    # 3. Extract CLIP embedding explicitly, only for inspection/debugging
    # The pipeline will internally do this again during generation.
    with torch.no_grad():
        pixel_values = pipe.feature_extractor(
            images=original_img,
            return_tensors="pt"
        ).pixel_values.to(device="cuda", dtype=pipe.image_encoder.dtype)

        clip_embedding = pipe.image_encoder(pixel_values).image_embeds

    print("CLIP embedding shape:", tuple(clip_embedding.shape))

    # 4. Generate image from the image CLIP embedding
    print("A reconstruir/variar a imagem a partir do embedding CLIP...")
    reconstructed_img = pipe(
        image=original_img,
        prompt="",
        noise_level=0,
        num_inference_steps=50,
        guidance_scale=1.0,
        generator=generator,
        height=768,
        width=768
    ).images[0]

    # 5. Create comparison grid
    grid_width = original_img.width * 2
    grid_height = original_img.height

    grid = Image.new("RGB", (grid_width, grid_height))
    grid.paste(original_img, (0, 0))
    grid.paste(reconstructed_img, (original_img.width, 0))

    grid.save(output_path)
    print(f"Sucesso! Grelha de comparação guardada em: {output_path}")

if __name__ == "__main__":
    INPUT_IMAGE = "/nas-ctm01/datasets/public/AFHQ/resized/train/wild/flickr_wild_000002.jpg"
    OUTPUT_GRID = "comparacao_unclip_standard.jpg"

    generate_comparison_grid(INPUT_IMAGE, OUTPUT_GRID)

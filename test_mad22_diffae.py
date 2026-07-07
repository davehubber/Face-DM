import argparse
import os
import sys
from pathlib import Path
import torch
import torchvision
from PIL import Image
from torchvision import transforms

def load_diffae_ffhq256_autoencoder(diffae_root: Path, checkpoint_path: Path, device: torch.device):
    """Loads the official DiffAE FFHQ256 autoencoder model securely."""
    diffae_root = Path(diffae_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

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

def main():
    parser = argparse.ArgumentParser(description="Encode and decode a single image via DiffAE.")
    parser.add_argument("image_path", type=str, default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/MAD22/original_sorted/BonaFide/001_03.jpg", help="Path to the input image to process")
    parser.add_argument("--diffae-root", type=str, default="/nas-ctm01/homes/dacordeiro/diffae/")
    parser.add_argument("--checkpoint", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt")
    parser.add_argument("--steps", type=int, default=20, help="Number of DDIM steps for decoding reconstruction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the Autoencoder
    print("Loading DiffAE model...")
    model = load_diffae_ffhq256_autoencoder(Path(args.diffae_root), Path(args.checkpoint), device)

    # 2. Load and Preprocess the Target Image
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Image not found at configuration path: {img_path}")
        return

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    raw_image = Image.open(img_path).convert("RGB")
    img_tensor = transform(raw_image).unsqueeze(0).to(device)  # Shape: (1, 3, 256, 256)

    # 3. Encode to Latent Spaces
    print("Extracting semantic (z_sem) and stochastic (x_T) codes...")
    with torch.no_grad():
        z_sem = model.encode(img_tensor)
        # Using T=250 for stochastic noise extraction matching your pipeline convention
        x_T = model.encode_stochastic(img_tensor, z_sem, T=250)

    # 4. Decode / Render Back to Image Space
    print(f"Rendering reconstruction canvas using {args.steps} DDIM steps...")
    with torch.no_grad():
        reconstructed_tensor = model.render(x_T, z_sem, T=args.steps)

    # 5. Save Output to Current Directory
    output_name = f"reconstructed_{img_path.stem}.png"
    output_path = Path(".") / output_name
    
    torchvision.utils.save_image(reconstructed_tensor, output_path, normalize=False)
    print(f"\n[SUCCESS] Execution finished. Output saved to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
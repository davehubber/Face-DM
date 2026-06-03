import argparse
import pickle
import sys
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# Setup paths to your DiffAE repository
PATH_TO_DIFF_MODEL = "/nas-ctm01/homes/dacordeiro/diffae/"
sys.path.append(PATH_TO_DIFF_MODEL)

from templates import ffhq256_autoenc
from experiment import LitModel

def load_img_tensor(path: str, img_size: int = 256) -> torch.Tensor:
    """Loads and preprocesses an image for DiffAE."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Render latent traversals using DiffAE.")
    parser.add_argument("--data-dir", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings")
    parser.add_argument("--prefix", type=str, default="ffhq256_diffae_zsem")
    parser.add_argument("--diffae-ckpt", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the pre-calculated vectors from Script 1
    handoff_path = data_dir / f"{args.prefix}_traversal_data.pkl"
    print(f"Loading traversal data from {handoff_path.name}...")
    with open(handoff_path, "rb") as f:
        traversal_data = pickle.load(f)

    # 2. Initialize DiffAE
    print("Initializing DiffAE model...")
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load(args.diffae_ckpt, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    # 3. Render the Grids
    for pc_name, data in traversal_data.items():
        print(f"Rendering images for {pc_name}...")
        
        path_high = data["path_high"]
        path_low = data["path_low"]
        seq_zsem_high = data["seq_zsem_high"]
        seq_zsem_low = data["seq_zsem_low"]

        grid_images = []
        
        with torch.no_grad():
            # Process HIGH sequence
            img_high = load_img_tensor(path_high, conf.img_size).to(device)
            cond_high = model.encode(img_high)
            xT_high = model.encode_stochastic(img_high, cond_high, T=250)
            
            for z_sem in seq_zsem_high:
                z_tensor = torch.tensor(z_sem, dtype=torch.float32, device=device).unsqueeze(0)
                out = model.render(xT_high, z_tensor, T=20)
                grid_images.append(out.squeeze(0).cpu()) 

            # Process LOW sequence
            img_low = load_img_tensor(path_low, conf.img_size).to(device)
            cond_low = model.encode(img_low)
            xT_low = model.encode_stochastic(img_low, cond_low, T=250)
            
            for z_sem in seq_zsem_low:
                z_tensor = torch.tensor(z_sem, dtype=torch.float32, device=device).unsqueeze(0)
                out = model.render(xT_low, z_tensor, T=20)
                grid_images.append(out.squeeze(0).cpu())

        # Save Visual Grid
        grid_out_path = data_dir / f"{pc_name}_traversal_grid.png"
        save_image(grid_images, grid_out_path, nrow=3, normalize=False)
        print(f"Saved {pc_name} traversal grid to: {grid_out_path.name}")

    print("\n[SUCCESS] Phase 2: DiffAE Rendering Complete.")

if __name__ == "__main__":
    main()

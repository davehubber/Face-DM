import os
import sys
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# --- Paths ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH_TO_DIFFAE = "/nas-ctm01/homes/dacordeiro/diffae"
CHECKPOINT_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt"

# TODO: Point this to one specific morph image to test
TEST_IMAGE_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/MorDIFF_crop/neutral_front_female_aligned/morphed/morph_097_03_and_112_03.png"
OUTPUT_SAVE_PATH = "reconstruction_check.png"

# Load DiffAE modules
sys.path.insert(0, PATH_TO_DIFFAE)
old_cwd = os.getcwd()
os.chdir(PATH_TO_DIFFAE)
from templates import ffhq256_autoenc
from experiment import LitModel
os.chdir(old_cwd)

def main():
    print(f"--- Loading DiffAE Model on {DEVICE} ---")
    conf = ffhq256_autoenc()
    conf.latent_infer_path = None  # Ensure fresh extraction
    
    model = LitModel(conf)
    state = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    
    model.ema_model.eval()
    model.ema_model.to(DEVICE)
    
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"--- Processing Image ---")
    if not os.path.exists(TEST_IMAGE_PATH):
        raise FileNotFoundError(f"Could not find test image at: {TEST_IMAGE_PATH}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(TEST_IMAGE_PATH).convert('RGB')
    
    # [1, 3, 256, 256] tensor
    batch = transform(img).unsqueeze(0).to(DEVICE)

    print("--- Encoding ---")
    # 1. Semantic Code (z_sem)
    cond = model.encode(batch)
    print(f"Semantic Latent Shape: {cond.shape}")
    
    # 2. Stochastic Code (x_T) - Using T=250 as in the original script
    T = 250
    xT = model.encode_stochastic(batch, cond, T=T)
    print(f"Stochastic Latent Shape: {xT.shape}")

    print("--- Decoding ---")
    # Render using the standard T=20 for speed, or T=250 for maximum fidelity
    # Using T=20 here since it matches how MorDIFF generated them
    pred = model.render(xT, cond, T=20)

    print("--- Generating Visual Comparison ---")
    # De-normalize images from [-1, 1] to [0, 1] for matplotlib
    input_img_vis = (batch[0].cpu().permute(1, 2, 0) + 1) / 2
    recon_img_vis = pred[0].detach().cpu().permute(1, 2, 0).clamp(0, 1)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(input_img_vis.numpy().clip(0, 1))
    axes[0].set_title("Input to Autoencoder\n(After Resize & Crop)")
    axes[0].axis('off')
    
    axes[1].imshow(recon_img_vis.numpy().clip(0, 1))
    axes[1].set_title("DiffAE Reconstruction\n(Rendered from $z_{sem}$ + $x_T$)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_SAVE_PATH, dpi=150)
    plt.close()
    
    print(f"\n[SUCCESS] Visual comparison saved to: {OUTPUT_SAVE_PATH}")

if __name__ == "__main__":
    main()

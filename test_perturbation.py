import argparse
import math
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid


def load_diffae_autoencoder(checkpoint_path: str, device: torch.device):
    from templates import ffhq256_autoenc
    from config import PretrainConfig
    from experiment import LitModel

    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(name="ffhq256_autoenc", path=checkpoint_path)
    conf.latent_infer_path = None

    model = LitModel(conf).to(device)
    model.eval()
    model.ema_model.eval()
    return model


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the dataset
    split_path = os.path.join(args.dataset_root, "semantic", "val_zsem.pt")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Could not find dataset at {split_path}")
        
    print(f"Loading dataset from {split_path}...")
    pack = torch.load(split_path, map_location="cpu")
    
    # 2. Extract a single sample
    idx = args.sample_idx
    original_z_sem = pack["z_sem"][idx].to(device)
    source_path = pack["source_paths"][idx]
    
    # 3. Create the perturbed embedding
    # The expected MSE of adding N(0, std) noise is std^2
    target_mse = args.target_mse
    noise_std = math.sqrt(target_mse)
    noise = torch.randn_like(original_z_sem) * noise_std
    perturbed_z_sem = original_z_sem + noise
    
    # Verify the actual MSE of this specific random sample
    actual_mse = torch.nn.functional.mse_loss(original_z_sem, perturbed_z_sem).item()
    print(f"Target MSE: {target_mse:.5f} | Actual applied MSE: {actual_mse:.5f}")

    # 4. Prepare the original image for inversion
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    print(f"Loading image: {source_path}")
    pil_img = Image.open(source_path).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 5. Load Autoencoder
    print(f"Loading DiffAE Autoencoder...")
    diffae = load_diffae_autoencoder(args.diffae_checkpoint, device)

    # 6. DiffAE Inversion (to get the spatial structural latent xT)
    # Note: We encode using the original image and original z_sem
    print(f"Inverting image for {args.decode_t} steps...")
    original_z_sem_batched = original_z_sem.unsqueeze(0)
    perturbed_z_sem_batched = perturbed_z_sem.unsqueeze(0)
    
    xT = diffae.encode_stochastic(img_tensor, original_z_sem_batched, T=args.decode_t)

    # 7. Render both versions
    print(f"Rendering original embedding...")
    decoded_original = diffae.render(xT, cond=original_z_sem_batched, T=args.decode_t).detach().cpu().clamp(0, 1)
    
    print(f"Rendering perturbed embedding (MSE={target_mse})...")
    decoded_perturbed = diffae.render(xT, cond=perturbed_z_sem_batched, T=args.decode_t).detach().cpu().clamp(0, 1)

    # Convert the actual source tensor to 0-1 range for display
    source_display = ((img_tensor.detach().cpu() + 1.0) / 2.0).clamp(0, 1)

    # 8. Build and Save Grid
    grid = make_grid(
        torch.cat([source_display, decoded_original, decoded_perturbed], dim=0),
        nrow=3,
        padding=8,
        pad_value=1.0,
    )

    out_dir = "perturbation_tests"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"perturbation_idx{idx}_mse{target_mse}.png")
    
    transforms.ToPILImage()(grid).save(out_path)
    print(f"\nSaved comparison grid to: {out_path}")
    print("Left: Source Image | Middle: Decoded Original | Right: Decoded Perturbed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test perceptual impact of an MSE error on a semantic embedding.")
    parser.add_argument("--dataset_root", default="/nas-ctm01/homes/dacordeiro/Face-DM/encoded_ffhq256_semantic_split", help="Path to your dataset root")
    parser.add_argument("--diffae_checkpoint", default="/nas-ctm01/homes/dacordeiro/Face-DM/checkpoints/ffhq256_autoenc/last.ckpt", help="DiffAE checkpoint")
    parser.add_argument("--sample_idx", default=0, type=int, help="Index of the image in the validation set to test")
    parser.add_argument("--target_mse", default=0.004, type=float, help="The target MSE magnitude to perturb the embedding by")
    parser.add_argument("--decode_t", default=100, type=int, help="Timesteps for DiffAE inversion and rendering")
    
    args = parser.parse_args()
    main(args)
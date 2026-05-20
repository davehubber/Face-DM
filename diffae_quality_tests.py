import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


# --- Model Loading and Utils ---

class FFHQImageFolder(Dataset):
    def __init__(self, image_root: Path, image_size: int = 256):
        self.image_root = Path(image_root)

        exts = {".png", ".jpg", ".jpeg", ".webp"}
        self.paths = sorted(
            p for p in self.image_root.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )

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
            "path": str(path),
            "filename": path.name,
        }


def load_diffae_ffhq256_autoencoder(
    diffae_root: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    """
    Loads the official DiffAE FFHQ256 autoencoder and returns the LitModel.
    """
    diffae_root = Path(diffae_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    if not diffae_root.exists():
        raise FileNotFoundError(f"DiffAE repo not found: {diffae_root}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Make repo importable and handle local paths.
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
        conf.latent_infer_path = None  # Only use autoencoding.

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


def input_tensor_to_vis(x: torch.Tensor) -> torch.Tensor:
    """Converts an input tensor from [-1, 1] to [0, 1]."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def render_tensor_to_vis(x: torch.Tensor) -> torch.Tensor:
    """Standardizes render output to [0, 1]."""
    x = x.detach().float().cpu()
    if float(x.min()) < -0.05:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


# --- Metric Computation ---

def compute_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """Computes PSNR between two tensors."""
    mse = F.mse_loss(img1, img2, reduction="mean")

    if mse.item() == 0.0:
        return float("inf")

    max_val_tensor = torch.tensor(
        max_val,
        dtype=mse.dtype,
        device=mse.device,
    )

    psnr = 20.0 * torch.log10(max_val_tensor / torch.sqrt(mse))
    return psnr.item()


def compute_mse_loss(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Computes basic MSE loss."""
    return F.mse_loss(img1, img2, reduction="mean").item()


def sample_unique_pairs(num_images: int, num_pairs: int, rng: np.random.Generator):
    """
    Samples unique unordered pairs without requiring each image to be used only once.

    Returns a list of tuples: [(idx1, idx2), ...]
    """
    max_pairs = num_images * (num_images - 1) // 2
    num_pairs = min(num_pairs, max_pairs)

    pairs = set()

    while len(pairs) < num_pairs:
        idx1, idx2 = rng.choice(num_images, size=2, replace=False)
        idx1 = int(idx1)
        idx2 = int(idx2)

        if idx1 > idx2:
            idx1, idx2 = idx2, idx1

        pairs.add((idx1, idx2))

    return list(pairs)


# --- Main Logic ---

def run_tests():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--diffae-root",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/diffae/",
        help="Path to cloned DiffAE repository",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt",
        help="Path to DiffAE FFHQ256 autoencoder checkpoint",
    )
    parser.add_argument(
        "--target-image-root",
        type=str,
        required=True,
        help="Path to the small test folder of images. Images should be FFHQ-aligned.",
    )
    parser.add_argument(
        "--out-base-dir",
        type=str,
        default="./diffae_quality_tests",
        help="Base directory for output",
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--t-steps",
        type=int,
        default=200,
        help="DDIM inversion/render steps.",
    )
    parser.add_argument(
        "--interpolate-pairs",
        type=int,
        default=10,
        help="Number of random pairs to interpolate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # Setup paths
    diffae_root = Path(args.diffae_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    target_image_root = Path(args.target_image_root).resolve()

    out_base = Path(args.out_base_dir).resolve()
    recon_dir = out_base / "reconstruction"
    interp_dir = out_base / "interpolation"
    recon_dir.mkdir(parents=True, exist_ok=True)
    interp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"Loading DiffAE model from: {checkpoint_path}...")
    model = load_diffae_ffhq256_autoencoder(
        diffae_root=diffae_root,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Load dataset
    dataset = FFHQImageFolder(
        image_root=target_image_root,
        image_size=args.image_size,
    )

    num_images = len(dataset)
    print(f"Found {num_images} images in test folder.")

    if num_images < 2:
        print("Error: Need at least two images to test interpolation.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # --- Part 1: Self-Reconstruction and Metrics ---

    print("\n--- Phase 1: Self-Reconstruction and Metric Computation ---")

    all_z_sem = []
    all_stochastic_latents = []
    recon_metrics = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing images"):
            imgs_raw = batch["img"].to(device, non_blocking=True)
            filenames = batch["filename"]

            batch_size_curr = imgs_raw.shape[0]

            # 1. Encode semantic and stochastic codes
            z_sem = model.encode(imgs_raw)

            stochastic_latent = model.encode_stochastic(
                imgs_raw,
                z_sem,
                T=args.t_steps,
            )

            all_z_sem.append(z_sem.detach().cpu())
            all_stochastic_latents.append(stochastic_latent.detach().cpu())

            # 2. Render back to images
            recons = model.render(
                stochastic_latent,
                z_sem,
                T=args.t_steps,
            )

            # 3. Compute MSE and PSNR
            original_vis_imgs = input_tensor_to_vis(imgs_raw).detach().cpu()
            recon_vis_imgs = render_tensor_to_vis(recons)

            for i in range(batch_size_curr):
                fname = filenames[i]

                orig_img = original_vis_imgs[i]
                recon_img = recon_vis_imgs[i]

                mse = compute_mse_loss(orig_img, recon_img)
                psnr = compute_psnr(orig_img, recon_img)

                recon_metrics[fname] = {
                    "mse": mse,
                    "psnr": psnr,
                }

                # Save original/reconstruction comparison image
                comparison_grid = torch.stack([orig_img, recon_img], dim=0)

                base_name = Path(fname).stem
                save_image(
                    comparison_grid,
                    recon_dir / f"{base_name}_comparison.png",
                    nrow=2,
                )

    # Compile and save aggregate metrics
    mses = [m["mse"] for m in recon_metrics.values()]
    psnrs = [
        m["psnr"]
        for m in recon_metrics.values()
        if m["psnr"] != float("inf")
    ]

    mean_mse = float(np.mean(mses)) if mses else float("nan")
    std_mse = float(np.std(mses)) if mses else float("nan")

    mean_psnr = float(np.mean(psnrs)) if psnrs else float("inf")
    std_psnr = float(np.std(psnrs)) if psnrs else 0.0

    metrics_txt = f"""# DiffAE FFHQ256 Reconstruction Metrics
# Configuration:
- Checkpoint: {checkpoint_path.name}
- Image Size: {args.image_size}
- T_inv/T_step: {args.t_steps}

# Aggregate Statistics:
- Number of images processed: {len(recon_metrics)}
- Mean MSE: {mean_mse:.8f} (+/- {std_mse:.8f})
- Mean PSNR: {mean_psnr:.6f} dB (+/- {std_psnr:.6f})

# Per-Image Results (MSE, PSNR):
"""

    for fname, m in recon_metrics.items():
        metrics_txt += (
            f"{fname}: "
            f"MSE={m['mse']:.8f}, "
            f"PSNR={m['psnr']:.6f}\n"
        )

    metrics_path = out_base / "reconstruction_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(metrics_txt)

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved comparison images to: {recon_dir}")

    # --- Part 2: Interpolation ---

    print("\n--- Phase 2: Testing Code Interpolation ---")

    z_sem_all = torch.cat(all_z_sem, dim=0)
    stochastic_all = torch.cat(all_stochastic_latents, dim=0)

    if z_sem_all.shape[0] != num_images:
        raise RuntimeError(
            f"Expected {num_images} semantic codes, got {z_sem_all.shape[0]}."
        )

    if stochastic_all.shape[0] != num_images:
        raise RuntimeError(
            f"Expected {num_images} stochastic latents, got {stochastic_all.shape[0]}."
        )

    pairs_to_do = min(
        args.interpolate_pairs,
        num_images * (num_images - 1) // 2,
    )

    print(f"Performing {pairs_to_do} random pairwise interpolations...")

    random_pairs = sample_unique_pairs(
        num_images=num_images,
        num_pairs=pairs_to_do,
        rng=rng,
    )

    all_filenames = [p.name for p in dataset.paths]

    model.to(device)
    model.eval()

    with torch.no_grad():
        for pair_idx, (idx1, idx2) in tqdm(
            enumerate(random_pairs),
            desc="Interpolating",
            total=pairs_to_do,
        ):
            fname1 = all_filenames[idx1]
            fname2 = all_filenames[idx2]

            z1 = z_sem_all[idx1].unsqueeze(0).to(device)
            z2 = z_sem_all[idx2].unsqueeze(0).to(device)

            s1 = stochastic_all[idx1].unsqueeze(0).to(device)
            s2 = stochastic_all[idx2].unsqueeze(0).to(device)

            alpha = 0.5
            interp_z = z1 * (1.0 - alpha) + z2 * alpha
            interp_s = s1 * (1.0 - alpha) + s2 * alpha

            interp_recon = model.render(
                interp_s,
                interp_z,
                T=args.t_steps,
            )

            interp_vis = render_tensor_to_vis(interp_recon)

            base1 = Path(fname1).stem
            base2 = Path(fname2).stem
            interp_fname = f"interp_{pair_idx:04d}_{base1}_{base2}.png"

            save_image(
                interp_vis,
                interp_dir / interp_fname,
            )

    print(f"Saved interpolated images to: {interp_dir}")
    print(f"Complete results available in: {out_base}")


if __name__ == "__main__":
    run_tests()

import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from train import get_unet, ColdDiffusion, calculate_metrics
from utils import to_uint8, order_by_brightness, save_images

class RealMorphDataset(Dataset):
    def __init__(self, mordiff_path, frll_path, transform=None):
        self.mordiff_path = mordiff_path
        self.frll_path = frll_path
        self.transform = transform
        self.pairs = []

        # Dynamically scan the 4 subfolders via finding 'morphed' directories
        for root, dirs, files in os.walk(mordiff_path):
            if os.path.basename(root) == "morphed":
                parent_folder = os.path.basename(os.path.dirname(root))
                
                # Map MorDIFF subfolder attributes to correct FRLL source folders
                if "neutral" in parent_folder.lower():
                    frll_subfolder = "neutral_front"
                elif "smiling" in parent_folder.lower():
                    frll_subfolder = "smiling_front"
                else:
                    print(f"⚠️ Warning: Could not resolve FRLL folder for {parent_folder}. Skipping.")
                    continue

                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        # Parse filenames: e.g., morph_081_03_and_122_03.png
                        base_name = os.path.splitext(f)[0]
                        if base_name.startswith("morph_") and "_and_" in base_name:
                            parts = base_name.replace("morph_", "").split("_and_")
                            if len(parts) == 2:
                                src1_filename = f"{parts[0]}.jpg"
                                src2_filename = f"{parts[1]}.jpg"
                                
                                morph_path = os.path.join(root, f)
                                src1_path = os.path.join(frll_path, frll_subfolder, src1_filename)
                                src2_path = os.path.join(frll_path, frll_subfolder, src2_filename)

                                if os.path.exists(src1_path) and os.path.exists(src2_path):
                                    self.pairs.append({
                                        "morph": morph_path,
                                        "src1": src1_path,
                                        "src2": src2_path
                                    })

        print(f"Loaded {len(self.pairs)} valid morph-to-source evaluation pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        item = self.pairs[index]
        
        morph_img = Image.open(item["morph"]).convert("RGB")
        src1_img = Image.open(item["src1"]).convert("RGB")
        src2_img = Image.open(item["src2"]).convert("RGB")

        if self.transform is not None:
            morph_tensor = self.transform(morph_img)
            src1_tensor = self.transform(src1_img)
            src2_tensor = self.transform(src2_img)
        else:
            # Fallback if no transform is passed (ensures 64x64 resolution safety)
            default_size = (64, 64)
            morph_tensor = TF.to_tensor(TF.resize(morph_img, default_size))
            src1_tensor = TF.to_tensor(TF.resize(src1_img, default_size))
            src2_tensor = TF.to_tensor(TF.resize(src2_img, default_size))

        # Re-use training logic to order the ground truth targets by brightness
        bright_tensor, dark_tensor = order_by_brightness(src1_tensor, src2_tensor)
        
        return morph_tensor, bright_tensor, dark_tensor


def run_morphing_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Establish targets for experiments workspace structure
    base_experiment_dir = os.path.join("experiments", args.run_name)
    output_dir = os.path.join(base_experiment_dir, "morphing_eval")
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)

    # Initialize model architecture and load weights
    model = get_unet(args.image_size)
    model_path = os.path.join(base_experiment_dir, "checkpoints", "unet_ema.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not locate checkpoint file at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )
    lpips_model = lpips.LPIPS(net="alex").to(device)

    # FIXED: Added the explicit Resize step here to match your model's required input dimensions
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = RealMorphDataset(args.mordiff_path, args.frll_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ssim_bright, ssim_dark, psnr_bright, psnr_dark, lpips_bright, lpips_dark = [], [], [], [], [], []
    success_bright_list, success_dark_list = [], []

    grid_predicted_bright, grid_predicted_dark, grid_bright, grid_dark, grid_mixed = [], [], [], [], []
    collected_for_grid = 0
    total_evaluated = 0
    total_swaps = 0

    print("🚀 Starting iterative extraction on real morph datasets...")
    for mixed_images, bright_images, dark_images in dataloader:
        mixed_images = mixed_images.to(device)
        bright_images = bright_images.to(device)
        dark_images = dark_images.to(device)

        # Run iterative reverse diffusion process using the morph image as initialization
        predicted_bright, predicted_dark = diffusion.sample(model, mixed_images, alpha_init=args.alpha_init)

        bright_uint8 = to_uint8(bright_images)
        dark_uint8 = to_uint8(dark_images)
        mixed_uint8 = to_uint8(mixed_images)

        # Collect up to 50 samples for the qualitative visualization grid
        if collected_for_grid < 50:
            grid_predicted_bright.append(predicted_bright.cpu())
            grid_predicted_dark.append(predicted_dark.cpu())
            grid_bright.append(bright_uint8.cpu())
            grid_dark.append(dark_uint8.cpu())
            grid_mixed.append(mixed_uint8.cpu())
            collected_for_grid += len(bright_uint8)

        bright_np = bright_uint8.cpu().permute(0, 2, 3, 1).numpy()
        predicted_bright_np = predicted_bright.cpu().permute(0, 2, 3, 1).numpy()
        dark_np = dark_uint8.cpu().permute(0, 2, 3, 1).numpy()
        predicted_dark_np = predicted_dark.cpu().permute(0, 2, 3, 1).numpy()
        mixed_np = mixed_uint8.cpu().permute(0, 2, 3, 1).numpy()

        for k in range(len(bright_np)):
            sb, sd, pb, pd, lb, ld, is_swapped, sec_b, sec_d = calculate_metrics(
                bright_np[k],
                dark_np[k],
                predicted_bright_np[k],
                predicted_dark_np[k],
                bright_uint8[k],
                dark_uint8[k],
                predicted_bright[k],
                predicted_dark[k],
                lpips_model,
                mixed_np[k]
            )

            if is_swapped:
                total_swaps += 1

            ssim_bright.append(sb)
            ssim_dark.append(sd)
            psnr_bright.append(pb)
            psnr_dark.append(pd)
            lpips_bright.append(lb)
            lpips_dark.append(ld)
            success_bright_list.append(sec_b)
            success_dark_list.append(sec_d)
            total_evaluated += 1

    # Save visual tracking overview grid
    if collected_for_grid > 0:
        save_images(
            torch.cat(grid_predicted_bright)[:50],
            torch.cat(grid_predicted_dark)[:50],
            torch.cat(grid_bright)[:50],
            torch.cat(grid_dark)[:50],
            os.path.join(output_dir, "samples", "morph_eval_grid_50.jpg"),
            input_images=torch.cat(grid_mixed)[:50],
        )

    # Format metrics report
    metrics_report = (
        f"--- Iterative Real Morphing Evaluation Metrics ({args.run_name}) ---\n"
        f"Total Evaluated Morph Pairs: {total_evaluated}\n"
        f"Identity Assignment Swap Rate: {(total_swaps / max(total_evaluated, 1)) * 100:.2f}%\n"
        f"SSIM Bright Component: {np.average(ssim_bright):.4f}\n"
        f"SSIM Dark Component: {np.average(ssim_dark):.4f}\n"
        f"PSNR Bright Component: {np.average(psnr_bright):.4f}\n"
        f"PSNR Dark Component: {np.average(psnr_dark):.4f}\n"
        f"LPIPS Bright Component: {np.average(lpips_bright):.4f}\n"
        f"LPIPS Dark Component: {np.average(lpips_dark):.4f}\n"
        f"Separation Success Rate Bright (%S): {np.average(success_bright_list) * 100:.2f}%\n"
        f"Separation Success Rate Dark (%S): {np.average(success_dark_list) * 100:.2f}%\n"
    )
    
    print(f"\n{metrics_report}")
    with open(os.path.join(output_dir, "results", "morph_final_metrics.txt"), "w") as f:
        f.write(metrics_report)
        
    print(f"✅ Morphing evaluation outputs completely saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mordiff_path", default="MorDIFF_crop", help="Path to MorDIFF root folder")
    parser.add_argument("--frll_path", default="FRLL", help="Path to FRLL root folder containing un-morphed targets")
    parser.add_argument("--run_name", default="faces_dual_iso", help="Name of the checkpoint run directory")
    
    # Diffusion Constants matched to configuration
    parser.add_argument("--alpha_max", default=0.5, type=float)
    parser.add_argument("--alpha_init", default=0.5, type=float)
    parser.add_argument("--max_timesteps", default=300, type=int)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--batch_size", default=256, type=int)

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)
    
    run_morphing_evaluation(args)

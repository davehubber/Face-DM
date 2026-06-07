import math
import os

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import optim

from utils import get_data, save_images, setup_logging, to_uint8


class ColdDiffusion:
    def __init__(self, max_timesteps=250, alpha_max=0.5, device="cuda", lpips_net="alex"):
        self.max_timesteps = max_timesteps
        self.device = device
        self.alteration_per_t = alpha_max / max_timesteps

        self.lpips_net = lpips_net
        self.lpips_model = None

    def _get_lpips_model(self):
        if self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net=self.lpips_net).to(self.device)
            self.lpips_model.eval()
        return self.lpips_model

    def mix_images(self, bright_image, dark_image, t):
        weight = (self.alteration_per_t * t)[:, None, None, None]
        # Applied square root to both coefficients
        return bright_image * torch.sqrt(1.0 - weight) + dark_image * torch.sqrt(weight)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def sample(self, model, mixed_image, alpha_init=0.5):
        n = len(mixed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)

        # Track swap state per sample: shape (n, 1, 1, 1)
        is_swapped = torch.zeros(n, 1, 1, 1, dtype=torch.bool, device=self.device)
        first_dark = None
        
        # Retrieve the initialized LPIPS network instance
        lpips_model = self._get_lpips_model()

        model.eval()
        with torch.no_grad():
            x_t = mixed_image.to(self.device)

            for i in reversed(range(1, init_timestep + 1)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)

                model_out = model(x_t, t).sample
                p1 = model_out[:, :3]  # Assumed Bright (initially)
                p2 = model_out[:, 3:]  # Assumed Dark (initially)

                if i == init_timestep:
                    # Capture the initial dark prediction anchor
                    predicted_dark = p2
                    first_dark = p2.clone()
                else:
                    # Clamp to [-1, 1] range to ensure stable LPIPS perceptual distance behavior
                    p1_clamped = torch.clamp(p1, -1.0, 1.0)
                    p2_clamped = torch.clamp(p2, -1.0, 1.0)
                    first_dark_clamped = torch.clamp(first_dark, -1.0, 1.0)

                    # Compute batch-wise perceptual distances (returns shape: [n, 1, 1, 1])
                    dist_p1 = lpips_model(p1_clamped, first_dark_clamped)
                    dist_p2 = lpips_model(p2_clamped, first_dark_clamped)
                    
                    # Check for an identity inversion ONLY if a swap hasn't been locked in yet
                    new_swap = (~is_swapped) & (dist_p1 < dist_p2)
                    is_swapped = is_swapped | new_swap
                    
                    # Route channels dynamically based on the persistent mask
                    predicted_dark = torch.where(is_swapped, p1, p2)

                # Mathematically resolve the bright counterpart
                predicted_bright = (mixed_image - math.sqrt(1.0 - alpha_init) * predicted_dark) / math.sqrt(alpha_init)

                # --- CLAMPING LOGIC ---
                predicted_dark = torch.clamp(predicted_dark, -1.0, 1.0)
                predicted_bright = torch.clamp(predicted_bright, -1.0, 1.0)
                # --------------------------

                # Pass dark image first to mixing method
                x_t = x_t - self.mix_images(predicted_dark, predicted_bright, t) + self.mix_images(
                    predicted_dark, predicted_bright, t - 1
                )

        model.train()

        # Final extraction from converged dark trajectory
        predicted_bright = (mixed_image - math.sqrt(1.0 - alpha_init) * x_t) / math.sqrt(alpha_init)
        return to_uint8(predicted_bright), to_uint8(x_t)


def get_unet(image_size):
    resolution = image_size[0] if isinstance(image_size, tuple) else image_size
    return UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=6, # Updated to predict both images
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def evaluate_validation_loss(model, dataloader, diffusion, accelerator):
    """Calculates fast validation loss by sampling random timesteps."""
    model.eval()
    loss_sum = torch.zeros(1, device=accelerator.device)
    loss_count = torch.zeros(1, device=accelerator.device)

    with torch.no_grad():
        for bright_images, dark_images in dataloader:
            t = diffusion.sample_timesteps(bright_images.shape[0]).to(accelerator.device)
            x_t = diffusion.mix_images(bright_images, dark_images, t)

            model_out = model(x_t, t).sample
            pred_bright_unet, pred_dark_unet = torch.chunk(model_out, 2, dim=1)
            
            # Loss evaluates both UNet predictions against ground truths
            loss = F.mse_loss(pred_bright_unet, bright_images, reduction="sum") + \
                   F.mse_loss(pred_dark_unet, dark_images, reduction="sum")

            loss_sum += loss.detach()
            loss_count += bright_images.numel()

    avg_val_loss = (accelerator.gather(loss_sum).sum() / accelerator.gather(loss_count).sum()).item()
    model.train()
    return avg_val_loss


def save_training_preview(unet, diffusion, fixed_bright_images, fixed_dark_images, alpha_init, save_dir, is_best=False):
    fixed_mixed = fixed_bright_images * math.sqrt(1.0 - alpha_init) + fixed_dark_images * math.sqrt(alpha_init)
    predicted_bright, predicted_dark = diffusion.sample(unet, fixed_mixed, alpha_init)

    save_path = os.path.join(save_dir, "best.jpg" if is_best else "latest.jpg")
    save_images(
        predicted_bright,
        predicted_dark,
        to_uint8(fixed_bright_images),
        to_uint8(fixed_dark_images),
        save_path,
        input_images=to_uint8(fixed_mixed),
    )


def train(args):
    base_dir = setup_logging(args.run_name)

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device

    train_dataloader = get_data(args, "train")
    val_dataloader = get_data(args, "val")

    model = get_unet(args.image_size)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=steps_per_epoch * args.epochs,
    )

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )

    ema_model = EMAModel(model.parameters(), inv_gamma=1.0, power=0.75, max_value=0.9999)
    ema_model.to(device)

    if accelerator.is_main_process:
        wandb.init(project="Face-DM", name=args.run_name, config=vars(args))

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    fixed_bright_images, fixed_dark_images = [], []
    for bright_images, dark_images in val_dataloader:
        fixed_bright_images.append(bright_images)
        fixed_dark_images.append(dark_images)
        if sum(batch.shape[0] for batch in fixed_bright_images) >= args.num_fixed_samples:
            break
    fixed_bright_images = torch.cat(fixed_bright_images)[: args.num_fixed_samples].to(device)
    fixed_dark_images = torch.cat(fixed_dark_images)[: args.num_fixed_samples].to(device)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss_sum = 0.0
        epoch_train_loss_count = 0

        for bright_images, dark_images in train_dataloader:
            t = diffusion.sample_timesteps(bright_images.shape[0]).to(device)

            with accelerator.accumulate(model):
                x_t = diffusion.mix_images(bright_images, dark_images, t)

                model_out = model(x_t, t).sample
                pred_bright_unet, pred_dark_unet = torch.chunk(model_out, 2, dim=1)
                
                # Combine losses to force the UNet to learn representations for both
                loss = F.mse_loss(pred_bright_unet, bright_images) + F.mse_loss(pred_dark_unet, dark_images)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                ema_model.step(model.parameters())
                synced_loss = accelerator.gather(loss.detach()).mean().item()
                epoch_train_loss_sum += synced_loss
                epoch_train_loss_count += 1

        accelerator.wait_for_everyone()

        epoch_train_loss = epoch_train_loss_sum / max(epoch_train_loss_count, 1)
        unet = accelerator.unwrap_model(model)

        ema_model.store(unet.parameters())
        ema_model.copy_to(unet.parameters())
        
        val_loss = evaluate_validation_loss(model, val_dataloader, diffusion, accelerator)
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        if accelerator.is_main_process:
            wandb.log({"train_loss": epoch_train_loss, "val_loss": val_loss}, step=epoch + 1)
            torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "unet_ema.pt"))

            if is_best:
                torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "unet_ema_best.pt"))

            if (epoch + 1) % args.sample_every == 0:
                save_training_preview(
                    unet=unet,
                    diffusion=diffusion,
                    fixed_bright_images=fixed_bright_images,
                    fixed_dark_images=fixed_dark_images,
                    alpha_init=args.alpha_init,
                    save_dir=os.path.join(base_dir, "samples", "train_fixed"),
                    is_best=is_best,
                )

        ema_model.restore(unet.parameters())
        accelerator.wait_for_everyone()


def calculate_metrics(bright_img, dark_img, pred_bright, pred_dark, bright_tensor, dark_tensor, pred_bright_tensor, pred_dark_tensor, lpips_model):
    def prep_lpips(tensor):
        return (tensor.unsqueeze(0).float() - 127.5) / 127.5

    with torch.no_grad():
        # Direct LPIPS mappings
        l_bb = lpips_model(prep_lpips(pred_bright_tensor), prep_lpips(bright_tensor)).item()
        l_dd = lpips_model(prep_lpips(pred_dark_tensor), prep_lpips(dark_tensor)).item()
        
        # Crossed LPIPS mappings
        l_bd = lpips_model(prep_lpips(pred_bright_tensor), prep_lpips(dark_tensor)).item()
        l_db = lpips_model(prep_lpips(pred_dark_tensor), prep_lpips(bright_tensor)).item()

    # Alignment Check: If crossed mapping has lower perceptual distance sum, a swap occurred
    is_swapped = (l_bd + l_db) < (l_bb + l_dd)

    if is_swapped:
        # Swap the numpy arrays to mathematically align identities
        pred_bright, pred_dark = pred_dark, pred_bright
        
        # Return the corresponding crossed values
        final_lpips_bright = l_db
        final_lpips_dark = l_bd
    else:
        final_lpips_bright = l_bb
        final_lpips_dark = l_dd

    # SSIM and PSNR calculated with correctly aligned outputs
    ssim_bright = structural_similarity(bright_img, pred_bright, data_range=255, channel_axis=-1)
    ssim_dark = structural_similarity(dark_img, pred_dark, data_range=255, channel_axis=-1)
    
    psnr_bright = peak_signal_noise_ratio(bright_img, pred_bright, data_range=255)
    psnr_dark = peak_signal_noise_ratio(dark_img, pred_dark, data_range=255)
    
    return ssim_bright, ssim_dark, psnr_bright, psnr_dark, final_lpips_bright, final_lpips_dark, is_swapped


def eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    val_dataloader = get_data(args, "val")
    model = get_unet(args.image_size)

    model, val_dataloader = accelerator.prepare(model, val_dataloader)

    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )
    lpips_model = lpips.LPIPS(net="alex").to(device)

    ssim_bright, ssim_dark, lpips_bright, lpips_dark, psnr_bright, psnr_dark = [], [], [], [], [], []

    grid_predicted_bright, grid_predicted_dark, grid_bright, grid_dark, grid_mixed = [], [], [], [], []
    collected_for_grid = 0
    total_evaluated = 0
    total_swaps = 0

    for bright_images, dark_images in val_dataloader:
        mixed_images = bright_images * math.sqrt(1.0 - args.alpha_init) + dark_images * math.sqrt(args.alpha_init)
        
        predicted_bright, predicted_dark = diffusion.sample(model, mixed_images, alpha_init=args.alpha_init)    

        bright_uint8 = to_uint8(bright_images)
        dark_uint8 = to_uint8(dark_images)
        mixed_uint8 = to_uint8(mixed_images)

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

        with torch.no_grad():
            for k in range(len(bright_np)):
                sb, sd, pb, pd, lb, ld, is_swapped = calculate_metrics(
                    bright_np[k],
                    dark_np[k],
                    predicted_bright_np[k],
                    predicted_dark_np[k],
                    bright_uint8[k],
                    dark_uint8[k],
                    predicted_bright[k],
                    predicted_dark[k],
                    lpips_model,
                )

                if is_swapped:
                    total_swaps += 1

                ssim_bright.append(sb)
                ssim_dark.append(sd)
                psnr_bright.append(pb)
                psnr_dark.append(pd)
                lpips_bright.append(lb)
                lpips_dark.append(ld)
                total_evaluated += 1

    if collected_for_grid > 0:
        save_images(
            torch.cat(grid_predicted_bright)[:50],
            torch.cat(grid_predicted_dark)[:50],
            torch.cat(grid_bright)[:50],
            torch.cat(grid_dark)[:50],
            os.path.join(base_dir, "samples", "eval", "eval_grid_50.jpg"),
            input_images=torch.cat(grid_mixed)[:50],
        )

    metrics_report = (
        f"--- Iterative Evaluation Metrics (Entire Validation Set) ---\n"
        f"Total Evaluated: {total_evaluated}\n"
        f"LPIPS Alignment Swap Rate: {(total_swaps / max(total_evaluated, 1)) * 100:.2f}%\n"
        f"SSIM Bright: {np.average(ssim_bright):.4f}\n"
        f"SSIM Dark: {np.average(ssim_dark):.4f}\n"
        f"PSNR Bright: {np.average(psnr_bright):.4f}\n"
        f"PSNR Dark: {np.average(psnr_dark):.4f}\n"
        f"LPIPS Bright: {np.average(lpips_bright):.4f}\n"
        f"LPIPS Dark: {np.average(lpips_dark):.4f}\n"
    )
    print(f"\n{metrics_report}")

    with open(os.path.join(base_dir, "results", "final_metrics.txt"), "w") as f:
        f.write(metrics_report)


def one_shot_eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    val_dataloader = get_data(args, "val")
    model = get_unet(args.image_size)

    model, val_dataloader = accelerator.prepare(model, val_dataloader)
    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )
    lpips_model = lpips.LPIPS(net="alex").to(device)

    ssim_bright, ssim_dark, psnr_bright, psnr_dark, lpips_bright, lpips_dark = [], [], [], [], [], []

    grid_predicted_bright, grid_predicted_dark, grid_bright, grid_dark, grid_mixed = [], [], [], [], []
    collected_for_grid = 0
    total_evaluated = 0
    total_swaps = 0

    for bright_images, dark_images in val_dataloader:
        n = len(bright_images)
        
        mixed_images = bright_images * math.sqrt(1.0 - args.alpha_init) + dark_images * math.sqrt(args.alpha_init)
        
        init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
        t = torch.full((n,), init_timestep, device=device, dtype=torch.long)

        with torch.no_grad():
            model_out = model(mixed_images, t).sample
            
            # Use only the bright prediction
            predicted_bright = model_out[:, :3]
            
            # Mathematically extract the dark prediction
            predicted_dark = (mixed_images - math.sqrt(1.0 - args.alpha_init) * predicted_bright) / math.sqrt(args.alpha_init)
            
            # Apply bounds clamping for the one-shot preview (optional but recommended for consistency)
            predicted_bright = torch.clamp(predicted_bright, -1.0, 1.0)
            predicted_dark = torch.clamp(predicted_dark, -1.0, 1.0)

        bright_uint8 = to_uint8(bright_images)
        dark_uint8 = to_uint8(dark_images)
        predicted_bright = to_uint8(predicted_bright)
        predicted_dark = to_uint8(predicted_dark)
        mixed_uint8 = to_uint8(mixed_images)

        if collected_for_grid < 50:
            grid_predicted_bright.append(predicted_bright.cpu())
            grid_predicted_dark.append(predicted_dark.cpu())
            grid_bright.append(bright_uint8.cpu())
            grid_dark.append(dark_uint8.cpu())
            grid_mixed.append(mixed_uint8.cpu())
            collected_for_grid += n

        bright_np = bright_uint8.cpu().permute(0, 2, 3, 1).numpy()
        predicted_bright_np = predicted_bright.cpu().permute(0, 2, 3, 1).numpy()
        dark_np = dark_uint8.cpu().permute(0, 2, 3, 1).numpy()
        predicted_dark_np = predicted_dark.cpu().permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            for k in range(n):
                sb, sd, pb, pd, lb, ld, is_swapped = calculate_metrics(
                    bright_np[k],
                    dark_np[k],
                    predicted_bright_np[k],
                    predicted_dark_np[k],
                    bright_uint8[k],
                    dark_uint8[k],
                    predicted_bright[k],
                    predicted_dark[k],
                    lpips_model,
                )

                if is_swapped:
                    total_swaps += 1

                ssim_bright.append(sb)
                ssim_dark.append(sd)
                psnr_bright.append(pb)
                psnr_dark.append(pd)
                lpips_bright.append(lb)
                lpips_dark.append(ld)
                total_evaluated += 1

    if collected_for_grid > 0:
        save_images(
            torch.cat(grid_predicted_bright)[:50],
            torch.cat(grid_predicted_dark)[:50],
            torch.cat(grid_bright)[:50],
            torch.cat(grid_dark)[:50],
            os.path.join(base_dir, "samples", "one_shot", "one_shot_grid_50.jpg"),
            input_images=torch.cat(grid_mixed)[:50],
        )

    metrics_report = (
        f"--- One-Shot Evaluation Metrics (Entire Validation Set) ---\n"
        f"Total Evaluated: {total_evaluated}\n"
        f"LPIPS Alignment Swap Rate: {(total_swaps / max(total_evaluated, 1)) * 100:.2f}%\n"
        f"SSIM Bright: {np.average(ssim_bright):.4f}\n"
        f"SSIM Dark: {np.average(ssim_dark):.4f}\n"
        f"PSNR Bright: {np.average(psnr_bright):.4f}\n"
        f"PSNR Dark: {np.average(psnr_dark):.4f}\n"
        f"LPIPS Bright: {np.average(lpips_bright):.4f}\n"
        f"LPIPS Dark: {np.average(lpips_dark):.4f}\n"
    )
    print(f"\n{metrics_report}")

    with open(os.path.join(base_dir, "results", "one_shot_metrics.txt"), "w") as f:
        f.write(metrics_report)

def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Path to the image dataset")
    parser.add_argument("--run_name", required=True, help="Name of the experiment folder")

    parser.add_argument("--test_images", default=1000, type=int, help="Number of distinct test images and test pairs")
    parser.add_argument("--train_samples_per_epoch", default=30000, type=int, help="Number of random train pairs per epoch")
    parser.add_argument("--num_workers", default=None, type=int, help="DataLoader worker count")

    parser.add_argument("--alpha_max", default=0.5, type=float, help="Maximum dark-image weight at the last timestep")
    parser.add_argument("--alpha_init", default=0.5, type=float, help="Dark-image weight used for previews and evaluation")
    parser.add_argument("--max_timesteps", default=300, type=int, help="Number of diffusion timesteps")
    parser.add_argument("--image_size", default=64, type=int, help="Square image size")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--epochs", default=150, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    
    parser.add_argument("--sample_every", default=25, type=int, help="Run iterative sampling preview every N epochs")
    parser.add_argument("--num_fixed_samples", default=10, type=int, help="Number of fixed validation pairs for previews")

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)

    #train(args)
    eval(args)
    #one_shot_eval(args)


if __name__ == "__main__":
    launch()

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

    def mix_images(self, image_1, image_2, t):
        weight = (self.alteration_per_t * t)[:, None, None, None]
        return image_1 * torch.sqrt(1.0 - weight) + image_2 * torch.sqrt(weight)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def sample(self, model, mixed_image, alpha_init=0.5):
        n = len(mixed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)

        model.eval()
        with torch.no_grad():
            x_t = mixed_image.to(self.device)
            
            # Track previous step's predictions to maintain trajectory alignment
            prev_pred_1 = None
            prev_pred_2 = None

            for i in reversed(range(1, init_timestep + 1)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)

                model_out = model(x_t, t).sample
                pred_A = model_out[:, :3]
                pred_B = model_out[:, 3:]
                
                if prev_pred_1 is None:
                    # First step: establish the baseline identity for this trajectory
                    aligned_pred_1 = pred_A
                    aligned_pred_2 = pred_B
                else:
                    # Calculate frame-to-frame MSE to detect if the model swapped outputs
                    mse_straight = F.mse_loss(pred_A, prev_pred_1, reduction="none").mean(dim=(1, 2, 3)) + \
                                   F.mse_loss(pred_B, prev_pred_2, reduction="none").mean(dim=(1, 2, 3))
                    mse_swapped = F.mse_loss(pred_A, prev_pred_2, reduction="none").mean(dim=(1, 2, 3)) + \
                                  F.mse_loss(pred_B, prev_pred_1, reduction="none").mean(dim=(1, 2, 3))

                    # Boolean mask: True where a swap occurred
                    swap_mask = (mse_swapped < mse_straight).view(-1, 1, 1, 1)

                    # Re-align predictions to match historical trajectory
                    aligned_pred_1 = torch.where(swap_mask, pred_B, pred_A)
                    aligned_pred_2 = torch.where(swap_mask, pred_A, pred_B)

                # Update history for the next iteration
                prev_pred_1 = aligned_pred_1
                prev_pred_2 = aligned_pred_2

                # --- CLAMPING & MATHEMATICAL EXTRACTION ---
                aligned_pred_1 = torch.clamp(aligned_pred_1, -1.0, 1.0)
                
                # Extract the second image relying purely on the aligned first image
                extracted_pred_2 = (mixed_image - math.sqrt(1.0 - alpha_init) * aligned_pred_1) / math.sqrt(alpha_init)
                extracted_pred_2 = torch.clamp(extracted_pred_2, -1.0, 1.0)

                # Step backwards
                x_t = x_t - self.mix_images(aligned_pred_1, extracted_pred_2, t) + self.mix_images(
                    aligned_pred_1, extracted_pred_2, t - 1
                )

        model.train()

        # Final extraction for returning
        final_pred_2 = (mixed_image - math.sqrt(1.0 - alpha_init) * x_t) / math.sqrt(alpha_init)
        return to_uint8(x_t), to_uint8(final_pred_2)       


def get_unet(image_size):
    resolution = image_size[0] if isinstance(image_size, tuple) else image_size
    return UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=6,
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
        for images_1, images_2 in dataloader:
            t = diffusion.sample_timesteps(images_1.shape[0]).to(accelerator.device)
            x_t = diffusion.mix_images(images_1, images_2, t)

            model_out = model(x_t, t).sample
            pred_A, pred_B = torch.chunk(model_out, 2, dim=1)
            
            # Permutation Invariant Validation Loss
            loss_straight = F.mse_loss(pred_A, images_1, reduction="none").mean(dim=(1, 2, 3)) + \
                            F.mse_loss(pred_B, images_2, reduction="none").mean(dim=(1, 2, 3))
            loss_swapped = F.mse_loss(pred_A, images_2, reduction="none").mean(dim=(1, 2, 3)) + \
                           F.mse_loss(pred_B, images_1, reduction="none").mean(dim=(1, 2, 3))

            loss = torch.minimum(loss_straight, loss_swapped).sum()

            loss_sum += loss.detach()
            loss_count += images_1.numel()

    avg_val_loss = (accelerator.gather(loss_sum).sum() / accelerator.gather(loss_count).sum()).item()
    model.train()
    return avg_val_loss


def save_training_preview(unet, diffusion, fixed_images_1, fixed_images_2, alpha_init, save_dir, is_best=False):
    fixed_mixed = fixed_images_1 * math.sqrt(1.0 - alpha_init) + fixed_images_2 * math.sqrt(alpha_init)
    predicted_1, predicted_2 = diffusion.sample(unet, fixed_mixed, alpha_init)

    save_path = os.path.join(save_dir, "best.jpg" if is_best else "latest.jpg")
    save_images(
        predicted_1,
        predicted_2,
        to_uint8(fixed_images_1),
        to_uint8(fixed_images_2),
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

    fixed_images_1, fixed_images_2 = [], []
    for images_1, images_2 in val_dataloader:
        fixed_images_1.append(images_1)
        fixed_images_2.append(images_2)
        if sum(batch.shape[0] for batch in fixed_images_1) >= args.num_fixed_samples:
            break
    fixed_images_1 = torch.cat(fixed_images_1)[: args.num_fixed_samples].to(device)
    fixed_images_2 = torch.cat(fixed_images_2)[: args.num_fixed_samples].to(device)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss_sum = 0.0
        epoch_train_loss_count = 0

        for images_1, images_2 in train_dataloader:
            t = diffusion.sample_timesteps(images_1.shape[0]).to(device)

            with accelerator.accumulate(model):
                x_t = diffusion.mix_images(images_1, images_2, t)

                model_out = model(x_t, t).sample
                pred_A, pred_B = torch.chunk(model_out, 2, dim=1)
                
                # Permutation Invariant Loss
                loss_straight = F.mse_loss(pred_A, images_1, reduction="none").mean(dim=(1, 2, 3)) + \
                                F.mse_loss(pred_B, images_2, reduction="none").mean(dim=(1, 2, 3))
                
                loss_swapped  = F.mse_loss(pred_A, images_2, reduction="none").mean(dim=(1, 2, 3)) + \
                                F.mse_loss(pred_B, images_1, reduction="none").mean(dim=(1, 2, 3))

                # Take the minimum error path for each pair in the batch
                loss = torch.minimum(loss_straight, loss_swapped).mean()

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
                    fixed_images_1=fixed_images_1,
                    fixed_images_2=fixed_images_2,
                    alpha_init=args.alpha_init,
                    save_dir=os.path.join(base_dir, "samples", "train_fixed"),
                    is_best=is_best,
                )

        ema_model.restore(unet.parameters())
        accelerator.wait_for_everyone()


def calculate_metrics(img_1, img_2, pred_1, pred_2, tensor_1, tensor_2, pred_tensor_1, pred_tensor_2, lpips_model):
    def prep_lpips(tensor):
        return (tensor.unsqueeze(0).float() - 127.5) / 127.5

    with torch.no_grad():
        # Direct LPIPS mappings
        l_11 = lpips_model(prep_lpips(pred_tensor_1), prep_lpips(tensor_1)).item()
        l_22 = lpips_model(prep_lpips(pred_tensor_2), prep_lpips(tensor_2)).item()
        
        # Crossed LPIPS mappings
        l_12 = lpips_model(prep_lpips(pred_tensor_1), prep_lpips(tensor_2)).item()
        l_21 = lpips_model(prep_lpips(pred_tensor_2), prep_lpips(tensor_1)).item()

    # Alignment Check: If crossed mapping has lower perceptual distance sum, a swap occurred
    is_swapped = (l_12 + l_21) < (l_11 + l_22)

    if is_swapped:
        # Swap the numpy arrays to mathematically align identities
        pred_1, pred_2 = pred_2, pred_1
        
        # Return the corresponding crossed values
        final_lpips_1 = l_21
        final_lpips_2 = l_12
    else:
        final_lpips_1 = l_11
        final_lpips_2 = l_22

    # SSIM and PSNR calculated with correctly aligned outputs
    ssim_1 = structural_similarity(img_1, pred_1, data_range=255, channel_axis=-1)
    ssim_2 = structural_similarity(img_2, pred_2, data_range=255, channel_axis=-1)
    
    psnr_1 = peak_signal_noise_ratio(img_1, pred_1, data_range=255)
    psnr_2 = peak_signal_noise_ratio(img_2, pred_2, data_range=255)
    
    return ssim_1, ssim_2, psnr_1, psnr_2, final_lpips_1, final_lpips_2, is_swapped


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

    ssim_1, ssim_2, lpips_1, lpips_2, psnr_1, psnr_2 = [], [], [], [], [], []

    grid_predicted_1, grid_predicted_2, grid_1, grid_2, grid_mixed = [], [], [], [], []
    collected_for_grid = 0
    total_evaluated = 0
    total_swaps = 0

    for images_1, images_2 in val_dataloader:
        mixed_images = images_1 * math.sqrt(1.0 - args.alpha_init) + images_2 * math.sqrt(args.alpha_init)
        
        predicted_1, predicted_2 = diffusion.sample(model, mixed_images, alpha_init=args.alpha_init)    

        uint8_1 = to_uint8(images_1)
        uint8_2 = to_uint8(images_2)
        mixed_uint8 = to_uint8(mixed_images)

        if collected_for_grid < 50:
            grid_predicted_1.append(predicted_1.cpu())
            grid_predicted_2.append(predicted_2.cpu())
            grid_1.append(uint8_1.cpu())
            grid_2.append(uint8_2.cpu())
            grid_mixed.append(mixed_uint8.cpu())
            collected_for_grid += len(uint8_1)

        np_1 = uint8_1.cpu().permute(0, 2, 3, 1).numpy()
        predicted_np_1 = predicted_1.cpu().permute(0, 2, 3, 1).numpy()
        np_2 = uint8_2.cpu().permute(0, 2, 3, 1).numpy()
        predicted_np_2 = predicted_2.cpu().permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            for k in range(len(np_1)):
                s1, s2, p1, p2, l1, l2, is_swapped = calculate_metrics(
                    np_1[k],
                    np_2[k],
                    predicted_np_1[k],
                    predicted_np_2[k],
                    uint8_1[k],
                    uint8_2[k],
                    predicted_1[k],
                    predicted_2[k],
                    lpips_model,
                )

                if is_swapped:
                    total_swaps += 1

                ssim_1.append(s1)
                ssim_2.append(s2)
                psnr_1.append(p1)
                psnr_2.append(p2)
                lpips_1.append(l1)
                lpips_2.append(l2)
                total_evaluated += 1

    if collected_for_grid > 0:
        save_images(
            torch.cat(grid_predicted_1)[:50],
            torch.cat(grid_predicted_2)[:50],
            torch.cat(grid_1)[:50],
            torch.cat(grid_2)[:50],
            os.path.join(base_dir, "samples", "eval", "eval_grid_50.jpg"),
            input_images=torch.cat(grid_mixed)[:50],
        )

    metrics_report = (
        f"--- Iterative Evaluation Metrics (Entire Validation Set) ---\n"
        f"Total Evaluated: {total_evaluated}\n"
        f"LPIPS Alignment Swap Rate: {(total_swaps / max(total_evaluated, 1)) * 100:.2f}%\n"
        f"SSIM Image 1: {np.average(ssim_1):.4f}\n"
        f"SSIM Image 2: {np.average(ssim_2):.4f}\n"
        f"PSNR Image 1: {np.average(psnr_1):.4f}\n"
        f"PSNR Image 2: {np.average(psnr_2):.4f}\n"
        f"LPIPS Image 1: {np.average(lpips_1):.4f}\n"
        f"LPIPS Image 2: {np.average(lpips_2):.4f}\n"
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

    ssim_1, ssim_2, psnr_1, psnr_2, lpips_1, lpips_2 = [], [], [], [], [], []

    grid_predicted_1, grid_predicted_2, grid_1, grid_2, grid_mixed = [], [], [], [], []
    collected_for_grid = 0
    total_evaluated = 0
    total_swaps = 0

    for images_1, images_2 in val_dataloader:
        n = len(images_1)
        
        mixed_images = images_1 * math.sqrt(1.0 - args.alpha_init) + images_2 * math.sqrt(args.alpha_init)
        
        init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
        t = torch.full((n,), init_timestep, device=device, dtype=torch.long)

        with torch.no_grad():
            model_out = model(mixed_images, t).sample
            
            # Use only the Image 1 prediction
            predicted_1 = model_out[:, :3]
            
            # Mathematically extract the Image 2 prediction
            predicted_2 = (mixed_images - math.sqrt(1.0 - args.alpha_init) * predicted_1) / math.sqrt(args.alpha_init)
            
            # Apply bounds clamping for the one-shot preview
            predicted_1 = torch.clamp(predicted_1, -1.0, 1.0)
            predicted_2 = torch.clamp(predicted_2, -1.0, 1.0)

        uint8_1 = to_uint8(images_1)
        uint8_2 = to_uint8(images_2)
        predicted_1 = to_uint8(predicted_1)
        predicted_2 = to_uint8(predicted_2)
        mixed_uint8 = to_uint8(mixed_images)

        if collected_for_grid < 50:
            grid_predicted_1.append(predicted_1.cpu())
            grid_predicted_2.append(predicted_2.cpu())
            grid_1.append(uint8_1.cpu())
            grid_2.append(uint8_2.cpu())
            grid_mixed.append(mixed_uint8.cpu())
            collected_for_grid += n

        np_1 = uint8_1.cpu().permute(0, 2, 3, 1).numpy()
        predicted_np_1 = predicted_1.cpu().permute(0, 2, 3, 1).numpy()
        np_2 = uint8_2.cpu().permute(0, 2, 3, 1).numpy()
        predicted_np_2 = predicted_2.cpu().permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            for k in range(n):
                s1, s2, p1, p2, l1, l2, is_swapped = calculate_metrics(
                    np_1[k],
                    np_2[k],
                    predicted_np_1[k],
                    predicted_np_2[k],
                    uint8_1[k],
                    uint8_2[k],
                    predicted_1[k],
                    predicted_2[k],
                    lpips_model,
                )

                if is_swapped:
                    total_swaps += 1

                ssim_1.append(s1)
                ssim_2.append(s2)
                psnr_1.append(p1)
                psnr_2.append(p2)
                lpips_1.append(l1)
                lpips_2.append(l2)
                total_evaluated += 1

    if collected_for_grid > 0:
        save_images(
            torch.cat(grid_predicted_1)[:50],
            torch.cat(grid_predicted_2)[:50],
            torch.cat(grid_1)[:50],
            torch.cat(grid_2)[:50],
            os.path.join(base_dir, "samples", "one_shot", "one_shot_grid_50.jpg"),
            input_images=torch.cat(grid_mixed)[:50],
        )

    metrics_report = (
        f"--- One-Shot Evaluation Metrics (Entire Validation Set) ---\n"
        f"Total Evaluated: {total_evaluated}\n"
        f"LPIPS Alignment Swap Rate: {(total_swaps / max(total_evaluated, 1)) * 100:.2f}%\n"
        f"SSIM Image 1: {np.average(ssim_1):.4f}\n"
        f"SSIM Image 2: {np.average(ssim_2):.4f}\n"
        f"PSNR Image 1: {np.average(psnr_1):.4f}\n"
        f"PSNR Image 2: {np.average(psnr_2):.4f}\n"
        f"LPIPS Image 1: {np.average(lpips_1):.4f}\n"
        f"LPIPS Image 2: {np.average(lpips_2):.4f}\n"
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

    parser.add_argument("--alpha_max", default=0.5, type=float, help="Maximum image_2 weight at the last timestep")
    parser.add_argument("--alpha_init", default=0.5, type=float, help="Image_2 weight used for previews and evaluation")
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

    train(args)
    eval(args)
    one_shot_eval(args)


if __name__ == "__main__":
    launch()
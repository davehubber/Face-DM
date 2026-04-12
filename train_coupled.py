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

from utils_coupled import get_data, save_images, setup_logging, to_uint8

from PIL import Image, ImageDraw, ImageFont


class ColdDiffusion:
    def __init__(self, max_timesteps=250, alpha_max=0.5, device="cuda"):
        self.max_timesteps = max_timesteps
        self.device = device
        self.alteration_per_t = alpha_max / max_timesteps

    def mix_images(self, bright_image, dark_image, t):
        weight = (self.alteration_per_t * t)[:, None, None, None]
        return bright_image * (1.0 - weight) + dark_image * weight

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def sample(self, model, mixed_image, alpha_init=0.5):
        n = len(mixed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)

        model.eval()
        with torch.no_grad():
            x_t = mixed_image.to(self.device)

            for i in reversed(range(1, init_timestep + 1)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)

                model_out = model(x_t, t).sample
                predicted_bright = model_out[:, :3]
                predicted_dark = model_out[:, 3:]

                x_t = x_t - self.mix_images(predicted_bright, predicted_dark, t) + self.mix_images(
                    predicted_bright, predicted_dark, t - 1
                )

        model.train()

        predicted_dark = (mixed_image - (1.0 - alpha_init) * x_t) / alpha_init
        return to_uint8(x_t), to_uint8(predicted_dark)


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


def build_fixed_validation_timesteps(batch_size, batch_idx, max_timesteps, device):
    base = torch.arange(batch_size, device=device, dtype=torch.long)
    return ((base + batch_idx * batch_size) % max_timesteps) + 1


def evaluate_validation_loss(model, dataloader, diffusion, accelerator):
    model.eval()
    loss_sum = torch.zeros(1, device=accelerator.device)
    loss_count = torch.zeros(1, device=accelerator.device)

    with torch.no_grad():
        for batch_idx, (bright_images, dark_images) in enumerate(dataloader):
            t = build_fixed_validation_timesteps(
                batch_size=bright_images.shape[0],
                batch_idx=batch_idx,
                max_timesteps=diffusion.max_timesteps,
                device=accelerator.device,
            )
            x_t = diffusion.mix_images(bright_images, dark_images, t)

            model_out = model(x_t, t).sample
            predicted_bright = model_out[:, :3]
            predicted_dark = model_out[:, 3:]

            loss_bright = F.mse_loss(predicted_bright, bright_images, reduction="sum")
            loss_dark = F.mse_loss(predicted_dark, dark_images, reduction="sum")

            loss_sum += (loss_bright + loss_dark).detach()
            loss_count += bright_images.numel() + dark_images.numel()

    avg_val_loss = (accelerator.gather(loss_sum).sum() / accelerator.gather(loss_count).sum()).item()
    model.train()
    return avg_val_loss


def save_training_preview(unet, diffusion, fixed_bright_images, fixed_dark_images, alpha_init, save_dir, is_best=False):
    fixed_mixed = fixed_bright_images * (1 - alpha_init) + fixed_dark_images * alpha_init
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
                predicted_bright = model_out[:, :3]
                predicted_dark = model_out[:, 3:]

                loss_bright = F.mse_loss(predicted_bright, bright_images)
                loss_dark = F.mse_loss(predicted_dark, dark_images)
                loss = 0.5 * (loss_bright + loss_dark)

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

        if (epoch + 1) % args.val_every != 0:
            continue

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
            save_training_preview(
                unet=unet,
                diffusion=diffusion,
                fixed_bright_images=fixed_bright_images,
                fixed_dark_images=fixed_dark_images,
                alpha_init=args.alpha_init,
                save_dir=os.path.join(base_dir, "samples", "train_fixed"),
                is_best=False,
            )

            if is_best:
                torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "unet_ema_best.pt"))
                save_training_preview(
                    unet=unet,
                    diffusion=diffusion,
                    fixed_bright_images=fixed_bright_images,
                    fixed_dark_images=fixed_dark_images,
                    alpha_init=args.alpha_init,
                    save_dir=os.path.join(base_dir, "samples", "train_fixed"),
                    is_best=True,
                )

        ema_model.restore(unet.parameters())
        accelerator.wait_for_everyone()


def calculate_metrics(target_1, target_2, pred_1, pred_2):
    ssim_1 = structural_similarity(target_1, pred_1, data_range=255, channel_axis=-1)
    ssim_2 = structural_similarity(target_2, pred_2, data_range=255, channel_axis=-1)
    psnr_1 = peak_signal_noise_ratio(target_1, pred_1, data_range=255)
    psnr_2 = peak_signal_noise_ratio(target_2, pred_2, data_range=255)
    return ssim_1, ssim_2, psnr_1, psnr_2


def calculate_lpips(lpips_model, target_tensor, pred_tensor):
    return lpips_model(
        (target_tensor.unsqueeze(0).float() - 127.5) / 127.5,
        (pred_tensor.unsqueeze(0).float() - 127.5) / 127.5,
    ).item()


def calculate_permutation_invariant_metrics(
    bright_image,
    dark_image,
    predicted_bright_image,
    predicted_dark_image,
    bright_tensor,
    dark_tensor,
    predicted_bright_tensor,
    predicted_dark_tensor,
    lpips_model,
):
    sb_direct, sd_direct, pb_direct, pd_direct = calculate_metrics(
        bright_image, dark_image, predicted_bright_image, predicted_dark_image
    )
    lb_direct = calculate_lpips(lpips_model, bright_tensor, predicted_bright_tensor)
    ld_direct = calculate_lpips(lpips_model, dark_tensor, predicted_dark_tensor)

    sb_swap, sd_swap, pb_swap, pd_swap = calculate_metrics(
        bright_image, dark_image, predicted_dark_image, predicted_bright_image
    )
    lb_swap = calculate_lpips(lpips_model, bright_tensor, predicted_dark_tensor)
    ld_swap = calculate_lpips(lpips_model, dark_tensor, predicted_bright_tensor)

    if (sb_swap + sd_swap) > (sb_direct + sd_direct):
        return sb_swap, sd_swap, pb_swap, pd_swap, lb_swap, ld_swap

    return sb_direct, sd_direct, pb_direct, pd_direct, lb_direct, ld_direct


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
    success_count_bright = 0
    success_count_dark = 0
    total_items = 0

    grid_predicted_bright, grid_predicted_dark, grid_bright, grid_dark, grid_mixed = [], [], [], [], []
    collected_for_grid = 0

    for bright_images, dark_images in val_dataloader:
        mixed_images = bright_images * (1 - args.alpha_init) + dark_images * args.alpha_init
        predicted_bright, predicted_dark = diffusion.sample(model, mixed_images, args.alpha_init)

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
        mixed_np = mixed_uint8.cpu().permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            for k in range(len(bright_np)):
                sb, sd, pb, pd, lb, ld = calculate_permutation_invariant_metrics(
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

                ssim_mixed_bright = structural_similarity(bright_np[k], mixed_np[k], data_range=255, channel_axis=-1)
                ssim_mixed_dark = structural_similarity(dark_np[k], mixed_np[k], data_range=255, channel_axis=-1)

                if sb > ssim_mixed_bright:
                    success_count_bright += 1
                if sd > ssim_mixed_dark:
                    success_count_dark += 1
                total_items += 1

                ssim_bright.append(sb)
                ssim_dark.append(sd)
                psnr_bright.append(pb)
                psnr_dark.append(pd)
                lpips_bright.append(lb)
                lpips_dark.append(ld)

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
        f"SSIM Bright: {np.average(ssim_bright):.4f}\n"
        f"SSIM Dark: {np.average(ssim_dark):.4f}\n"
        f"PSNR Bright: {np.average(psnr_bright):.4f}\n"
        f"PSNR Dark: {np.average(psnr_dark):.4f}\n"
        f"LPIPS Bright: {np.average(lpips_bright):.4f}\n"
        f"LPIPS Dark: {np.average(lpips_dark):.4f}\n"
        f"Success Rate Bright (%S): {(success_count_bright / total_items) * 100:.2f}%\n"
        f"Success Rate Dark (%S): {(success_count_dark / total_items) * 100:.2f}%\n"
    )
    print(f"\n{metrics_report}")

    with open(os.path.join(base_dir, "results", "final_metrics.txt"), "w") as f:
        f.write(metrics_report)

def _run_iterative_eval(args, max_items=None, grid_filename="eval_grid_50.jpg", metrics_filename="final_metrics.txt", title_suffix="Entire Validation Set"):
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
    success_count_bright = 0
    success_count_dark = 0
    total_items = 0

    grid_predicted_bright, grid_predicted_dark, grid_bright, grid_dark, grid_mixed = [], [], [], [], []
    collected_for_grid = 0
    limit = float("inf") if max_items is None else int(max_items)

    for bright_images, dark_images in val_dataloader:
        if total_items >= limit:
            break

        remaining = limit - total_items
        if remaining < len(bright_images):
            bright_images = bright_images[:remaining]
            dark_images = dark_images[:remaining]

        mixed_images = bright_images * (1 - args.alpha_init) + dark_images * args.alpha_init
        predicted_bright, predicted_dark = diffusion.sample(model, mixed_images, args.alpha_init)

        bright_uint8 = to_uint8(bright_images)
        dark_uint8 = to_uint8(dark_images)
        mixed_uint8 = to_uint8(mixed_images)

        if collected_for_grid < min(50, limit):
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

        with torch.no_grad():
            for k in range(len(bright_np)):
                sb, sd, pb, pd, lb, ld = calculate_permutation_invariant_metrics(
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

                ssim_mixed_bright = structural_similarity(bright_np[k], mixed_np[k], data_range=255, channel_axis=-1)
                ssim_mixed_dark = structural_similarity(dark_np[k], mixed_np[k], data_range=255, channel_axis=-1)

                if sb > ssim_mixed_bright:
                    success_count_bright += 1
                if sd > ssim_mixed_dark:
                    success_count_dark += 1
                total_items += 1

                ssim_bright.append(sb)
                ssim_dark.append(sd)
                psnr_bright.append(pb)
                psnr_dark.append(pd)
                lpips_bright.append(lb)
                lpips_dark.append(ld)

    if collected_for_grid > 0:
        n_grid = min(50, len(ssim_bright))
        save_images(
            torch.cat(grid_predicted_bright)[:n_grid],
            torch.cat(grid_predicted_dark)[:n_grid],
            torch.cat(grid_bright)[:n_grid],
            torch.cat(grid_dark)[:n_grid],
            os.path.join(base_dir, "samples", "eval", grid_filename),
            input_images=torch.cat(grid_mixed)[:n_grid],
        )

    metrics_report = (
        f"--- Iterative Evaluation Metrics ({title_suffix}) ---\n"
        f"SSIM Bright: {np.average(ssim_bright):.4f}\n"
        f"SSIM Dark: {np.average(ssim_dark):.4f}\n"
        f"PSNR Bright: {np.average(psnr_bright):.4f}\n"
        f"PSNR Dark: {np.average(psnr_dark):.4f}\n"
        f"LPIPS Bright: {np.average(lpips_bright):.4f}\n"
        f"LPIPS Dark: {np.average(lpips_dark):.4f}\n"
        f"Success Rate Bright (%S): {(success_count_bright / total_items) * 100:.2f}%\n"
        f"Success Rate Dark (%S): {(success_count_dark / total_items) * 100:.2f}%\n"
    )
    print(f"\n{metrics_report}")

    with open(os.path.join(base_dir, "results", metrics_filename), "w") as f:
        f.write(metrics_report)


def eval_fixed_50(args):
    _run_iterative_eval(
        args,
        max_items=50,
        grid_filename="eval_grid_fixed50.jpg",
        metrics_filename="final_metrics_fixed50.txt",
        title_suffix="Fixed 50 Validation Images",
    )

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
    success_count_bright = 0
    success_count_dark = 0
    total_items = 0

    grid_predicted_bright, grid_predicted_dark, grid_bright, grid_dark, grid_mixed = [], [], [], [], []
    collected_for_grid = 0

    for bright_images, dark_images in val_dataloader:
        n = len(bright_images)
        mixed_images = bright_images * (1 - args.alpha_init) + dark_images * args.alpha_init
        init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
        t = torch.full((n,), init_timestep, device=device, dtype=torch.long)

        with torch.no_grad():
            model_out = model(mixed_images, t).sample
            predicted_bright = model_out[:, :3]
            predicted_dark = (mixed_images - (1 - args.alpha_init) * predicted_bright) / args.alpha_init

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
        mixed_np = mixed_uint8.cpu().permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            for k in range(n):
                sb, sd, pb, pd, lb, ld = calculate_permutation_invariant_metrics(
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

                ssim_mixed_bright = structural_similarity(bright_np[k], mixed_np[k], data_range=255, channel_axis=-1)
                ssim_mixed_dark = structural_similarity(dark_np[k], mixed_np[k], data_range=255, channel_axis=-1)

                if sb > ssim_mixed_bright:
                    success_count_bright += 1
                if sd > ssim_mixed_dark:
                    success_count_dark += 1
                total_items += 1

                ssim_bright.append(sb)
                ssim_dark.append(sd)
                psnr_bright.append(pb)
                psnr_dark.append(pd)
                lpips_bright.append(lb)
                lpips_dark.append(ld)

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
        f"SSIM Bright: {np.average(ssim_bright):.4f}\n"
        f"SSIM Dark: {np.average(ssim_dark):.4f}\n"
        f"PSNR Bright: {np.average(psnr_bright):.4f}\n"
        f"PSNR Dark: {np.average(psnr_dark):.4f}\n"
        f"LPIPS Bright: {np.average(lpips_bright):.4f}\n"
        f"LPIPS Dark: {np.average(lpips_dark):.4f}\n"
        f"Success Rate Bright (%S): {(success_count_bright / total_items) * 100:.2f}%\n"
        f"Success Rate Dark (%S): {(success_count_dark / total_items) * 100:.2f}%\n"
    )
    print(f"\n{metrics_report}")

    with open(os.path.join(base_dir, "results", "one_shot_metrics.txt"), "w") as f:
        f.write(metrics_report)

def _tensor_to_pil_image(tensor):
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor_uint8 = to_uint8(tensor.unsqueeze(0))[0]
    array = tensor_uint8.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _mse_01(pred, target):
    pred_01 = (pred.detach().clamp(-1, 1) + 1.0) / 2.0
    target_01 = (target.detach().clamp(-1, 1) + 1.0) / 2.0
    return F.mse_loss(pred_01, target_01).item()


def _lpips_minus1_1(lpips_model, pred, target):
    pred = pred.detach().clamp(-1, 1).unsqueeze(0)
    target = target.detach().clamp(-1, 1).unsqueeze(0)
    return lpips_model(pred, target).item()


def _get_single_val_pair(dataloader, sample_index, device):
    if sample_index < 0:
        raise ValueError(f"sample_index must be >= 0, got {sample_index}")

    running_index = 0
    for bright_batch, dark_batch in dataloader:
        batch_size = bright_batch.shape[0]
        if running_index + batch_size > sample_index:
            local_index = sample_index - running_index
            return (
                bright_batch[local_index : local_index + 1].to(device),
                dark_batch[local_index : local_index + 1].to(device),
            )
        running_index += batch_size

    raise IndexError(f"sample_index={sample_index} is out of range for the validation set")


def _save_single_dark_trajectory_grid(rows, gt_dark_tensor, save_path):
    font = ImageFont.load_default()

    gt_pil = _tensor_to_pil_image(gt_dark_tensor)
    image_w, image_h = gt_pil.size

    label_w = 58
    cell_w = max(image_w, 92)
    text_h = 26
    row_h = image_h + text_h
    col_gap = 12
    row_gap = 6
    header_h = 18
    outer_pad = 8

    width = outer_pad * 2 + label_w + (3 * cell_w) + (2 * col_gap)
    height = outer_pad * 2 + header_h + len(rows) * row_h + max(0, len(rows) - 1) * row_gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    headers = ["model dark", "bright->dark", "GT dark"]
    for col, title in enumerate(headers):
        x0 = outer_pad + label_w + col * (cell_w + col_gap)
        draw.text((x0, outer_pad), title, fill="black", font=font)

    gt_text = "MSE: 0.0000\nLPIPS: 0.0000"

    for row_idx, row in enumerate(rows):
        y0 = outer_pad + header_h + row_idx * (row_h + row_gap)

        draw.text((outer_pad, y0 + image_h // 2 - 5), f"t={row['t']}", fill="black", font=font)

        entries = [
            (
                _tensor_to_pil_image(row["pred_dark_model"]),
                f"MSE: {row['mse_model']:.4f}\nLPIPS: {row['lpips_model']:.4f}",
            ),
            (
                _tensor_to_pil_image(row["pred_dark_math"]),
                f"MSE: {row['mse_math']:.4f}\nLPIPS: {row['lpips_math']:.4f}",
            ),
            (
                gt_pil,
                gt_text,
            ),
        ]

        for col, (img, text) in enumerate(entries):
            x0 = outer_pad + label_w + col * (cell_w + col_gap)
            img_x = x0 + (cell_w - img.width) // 2

            canvas.paste(img, (img_x, y0))
            draw.multiline_text(
                (x0, y0 + image_h + 2),
                text,
                fill="black",
                font=font,
                spacing=0,
            )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)


def eval_single_dark_trajectory(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.join("experiments", args.run_name)

    val_dataloader = get_data(args, "val")

    model = get_unet(args.image_size).to(device)
    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    bright_image, dark_image = _get_single_val_pair(
        val_dataloader,
        args.single_eval_index,
        device,
    )

    # Start from the final timestep so you always get one row per timestep.
    full_t = torch.tensor([diffusion.max_timesteps], device=device, dtype=torch.long)
    mixed_image = diffusion.mix_images(bright_image, dark_image, full_t)
    alpha_eval = diffusion.alteration_per_t * diffusion.max_timesteps

    if alpha_eval <= 0:
        raise ValueError(f"alpha_eval must be > 0, got {alpha_eval}")

    rows = []
    x_t = mixed_image.clone()

    with torch.no_grad():
        for i in reversed(range(1, diffusion.max_timesteps + 1)):
            t = torch.tensor([i], device=device, dtype=torch.long)

            model_out = model(x_t, t).sample
            predicted_bright_raw = model_out[:, :3]
            predicted_dark_model_raw = model_out[:, 3:]

            predicted_dark_math_raw = (
                mixed_image - (1.0 - alpha_eval) * predicted_bright_raw
            ) / alpha_eval

            predicted_dark_model = predicted_dark_model_raw.clamp(-1, 1)
            predicted_dark_math = predicted_dark_math_raw.clamp(-1, 1)

            rows.append(
                {
                    "t": i,
                    "pred_dark_model": predicted_dark_model[0].cpu(),
                    "pred_dark_math": predicted_dark_math[0].cpu(),
                    "mse_model": _mse_01(predicted_dark_model[0], dark_image[0]),
                    "mse_math": _mse_01(predicted_dark_math[0], dark_image[0]),
                    "lpips_model": _lpips_minus1_1(lpips_model, predicted_dark_model[0], dark_image[0]),
                    "lpips_math": _lpips_minus1_1(lpips_model, predicted_dark_math[0], dark_image[0]),
                }
            )

            x_t = (
                x_t
                - diffusion.mix_images(predicted_bright_raw, predicted_dark_model_raw, t)
                + diffusion.mix_images(predicted_bright_raw, predicted_dark_model_raw, t - 1)
            )

    save_path = os.path.join(
        base_dir,
        "samples",
        "eval",
        f"single_dark_trajectory_idx{args.single_eval_index}.png",
    )

    _save_single_dark_trajectory_grid(
        rows=rows,
        gt_dark_tensor=dark_image[0].cpu(),
        save_path=save_path,
    )

    print(f"Saved single-image dark trajectory grid to: {save_path}")

def _tensor_to_pil_image(tensor):
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor_uint8 = to_uint8(tensor.unsqueeze(0))[0]
    array = tensor_uint8.permute(1, 2, 0).numpy()
    return Image.fromarray(array)


def _get_single_val_pair(dataloader, sample_index, device):
    if sample_index < 0:
        raise ValueError(f"sample_index must be >= 0, got {sample_index}")

    running_index = 0
    for bright_batch, dark_batch in dataloader:
        batch_size = bright_batch.shape[0]
        if running_index + batch_size > sample_index:
            local_index = sample_index - running_index
            return (
                bright_batch[local_index : local_index + 1].to(device),
                dark_batch[local_index : local_index + 1].to(device),
            )
        running_index += batch_size

    raise IndexError(f"sample_index={sample_index} is out of range for the validation set")


def _save_two_prediction_trajectory_grid(rows, save_path):
    font = ImageFont.load_default()

    sample_img = _tensor_to_pil_image(rows[0]["pred_bright"])
    image_w, image_h = sample_img.size

    label_w = 58
    cell_w = image_w
    text_h = 14
    row_h = image_h + text_h
    col_gap = 12
    row_gap = 6
    header_h = 18
    outer_pad = 8

    width = outer_pad * 2 + label_w + (2 * cell_w) + col_gap
    height = outer_pad * 2 + header_h + len(rows) * row_h + max(0, len(rows) - 1) * row_gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    headers = ["pred bright", "pred dark"]
    for col, title in enumerate(headers):
        x0 = outer_pad + label_w + col * (cell_w + col_gap)
        draw.text((x0, outer_pad), title, fill="black", font=font)

    for row_idx, row in enumerate(rows):
        y0 = outer_pad + header_h + row_idx * (row_h + row_gap)

        draw.text((outer_pad, y0 + image_h // 2 - 5), f"t={row['t']}", fill="black", font=font)

        entries = [
            _tensor_to_pil_image(row["pred_bright"]),
            _tensor_to_pil_image(row["pred_dark"]),
        ]

        for col, img in enumerate(entries):
            x0 = outer_pad + label_w + col * (cell_w + col_gap)
            canvas.paste(img, (x0, y0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)


def eval_single_dark_dominant_path(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.join("experiments", args.run_name)

    val_dataloader = get_data(args, "val")

    model = get_unet(args.image_size).to(device)
    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )

    bright_image, dark_image = _get_single_val_pair(
        val_dataloader,
        args.single_eval_index,
        device,
    )

    # Dark-dominant corrupted input: dark is passed first to mix_images.
    full_t = torch.tensor([diffusion.max_timesteps], device=device, dtype=torch.long)
    x_t = diffusion.mix_images(dark_image, bright_image, full_t).clone()

    rows = []

    with torch.no_grad():
        for i in reversed(range(1, diffusion.max_timesteps + 1)):
            t = torch.tensor([i], device=device, dtype=torch.long)

            model_out = model(x_t, t).sample
            predicted_bright_raw = model_out[:, :3]
            predicted_dark_raw = model_out[:, 3:]

            rows.append(
                {
                    "t": i,
                    "pred_bright": predicted_bright_raw[0].cpu(),
                    "pred_dark": predicted_dark_raw[0].cpu(),
                }
            )

            # Reverse path now treats predicted dark as the anchor / dominant image.
            x_t = (
                x_t
                - diffusion.mix_images(predicted_dark_raw, predicted_bright_raw, t)
                + diffusion.mix_images(predicted_dark_raw, predicted_bright_raw, t - 1)
            )

    save_path = os.path.join(
        base_dir,
        "samples",
        "eval",
        f"single_dark_dominant_path_idx{args.single_eval_index}.png",
    )

    _save_two_prediction_trajectory_grid(rows, save_path)
    print(f"Saved dark-dominant path trajectory grid to: {save_path}")

def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Path to the image dataset")
    parser.add_argument("--run_name", required=True, help="Name of the experiment folder")

    parser.add_argument("--train_fraction", default=0.8, type=float, help="Fraction of source images used for training")
    parser.add_argument("--train_samples_per_epoch", default=None, type=int, help="Number of random train pairs per epoch")
    parser.add_argument("--val_samples", default=None, type=int, help="Number of deterministic validation pairs")
    parser.add_argument("--num_workers", default=None, type=int, help="DataLoader worker count")

    parser.add_argument("--alpha_max", default=0.5, type=float, help="Maximum dark-image weight at the last timestep")
    parser.add_argument("--alpha_init", default=0.5, type=float, help="Dark-image weight used for previews and evaluation")
    parser.add_argument("--max_timesteps", default=300, type=int, help="Number of diffusion timesteps")
    parser.add_argument("--image_size", default=64, type=int, help="Square image size")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--epochs", default=150, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--val_every", default=1, type=int, help="Run validation every N epochs")
    parser.add_argument("--num_fixed_samples", default=10, type=int, help="Number of fixed validation pairs for previews")

    parser.add_argument("--single_eval_index", default=1, type=int, help="Index of the deterministic validation sample to visualize")

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)

    eval_single_dark_dominant_path(args)
    eval_single_dark_trajectory(args)
    #eval_fixed_50(args)
    #train(args)
    #eval(args)
    #one_shot_eval(args)


if __name__ == "__main__":
    launch()
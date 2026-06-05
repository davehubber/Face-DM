import math
import os

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
from accelerate import Accelerator
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from PIL import Image
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
        # Standard linear mixing
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
                predicted_bright = model_out
                
                # Standard linear extraction
                predicted_dark = (mixed_image - (1.0 - alpha_init) * predicted_bright) / alpha_init

                x_t = x_t - self.mix_images(predicted_bright, predicted_dark, t) + self.mix_images(
                    predicted_bright, predicted_dark, t - 1
                )

        model.train()

        # Standard linear extraction for the final returned tensor
        predicted_dark = (mixed_image - (1.0 - alpha_init) * x_t) / alpha_init
        return to_uint8(x_t), to_uint8(predicted_dark)       


def get_unet(image_size):
    resolution = image_size[0] if isinstance(image_size, tuple) else image_size
    return UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
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

            predicted_bright = model(x_t, t).sample
            
            loss = F.mse_loss(predicted_bright, bright_images, reduction="sum")

            loss_sum += loss.detach()
            loss_count += bright_images.numel()

    avg_val_loss = (accelerator.gather(loss_sum).sum() / accelerator.gather(loss_count).sum()).item()
    model.train()
    return avg_val_loss


def save_training_preview(unet, diffusion, fixed_bright_images, fixed_dark_images, alpha_init, save_dir, is_best=False):
    fixed_mixed = fixed_bright_images * (1.0 - alpha_init) + fixed_dark_images * alpha_init
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

                predicted_bright = model(x_t, t).sample
                
                loss = F.mse_loss(predicted_bright, bright_images)

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

            if (epoch + 1) % args.sample_every == 0 or is_best:
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
    ssim_bright = structural_similarity(bright_img, pred_bright, data_range=255, channel_axis=-1)
    ssim_dark = structural_similarity(dark_img, pred_dark, data_range=255, channel_axis=-1)
    
    psnr_bright = peak_signal_noise_ratio(bright_img, pred_bright, data_range=255)
    psnr_dark = peak_signal_noise_ratio(dark_img, pred_dark, data_range=255)
    
    lpips_bright = lpips_model(
        (bright_tensor.unsqueeze(0).float() - 127.5) / 127.5,
        (pred_bright_tensor.unsqueeze(0).float() - 127.5) / 127.5,
    ).item()
    
    lpips_dark = lpips_model(
        (dark_tensor.unsqueeze(0).float() - 127.5) / 127.5,
        (pred_dark_tensor.unsqueeze(0).float() - 127.5) / 127.5,
    ).item()
    
    return ssim_bright, ssim_dark, psnr_bright, psnr_dark, lpips_bright, lpips_dark


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

    for bright_images, dark_images in val_dataloader:
        mixed_images = bright_images * (1.0 - args.alpha_init) + dark_images * args.alpha_init
        
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
                sb, sd, pb, pd, lb, ld = calculate_metrics(
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

    for bright_images, dark_images in val_dataloader:
        n = len(bright_images)
        
        mixed_images = bright_images * (1.0 - args.alpha_init) + dark_images * args.alpha_init
        
        init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
        t = torch.full((n,), init_timestep, device=device, dtype=torch.long)

        with torch.no_grad():
            predicted_bright = model(mixed_images, t).sample
            
            predicted_dark = (mixed_images - (1.0 - args.alpha_init) * predicted_bright) / args.alpha_init

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
                sb, sd, pb, pd, lb, ld = calculate_metrics(
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
    )
    print(f"\n{metrics_report}")

    with open(os.path.join(base_dir, "results", "one_shot_metrics.txt"), "w") as f:
        f.write(metrics_report)


def visualize_sampling_path(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)
    save_dir = os.path.join(base_dir, "samples", "path_visualization")
    os.makedirs(save_dir, exist_ok=True)

    val_dataloader = get_data(args, "val")
    model = get_unet(args.image_size)
    
    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )

    fixed_bright_images, fixed_dark_images = [], []
    for bright_images, dark_images in val_dataloader:
        fixed_bright_images.append(bright_images)
        fixed_dark_images.append(dark_images)
        if sum(batch.shape[0] for batch in fixed_bright_images) >= args.num_fixed_samples:
            break
            
    fixed_bright_images = torch.cat(fixed_bright_images)[: args.num_fixed_samples].to(device)
    fixed_dark_images = torch.cat(fixed_dark_images)[: args.num_fixed_samples].to(device)

    # Extract exactly the 2nd pair (Index 1)
    bright_gt = fixed_bright_images[1:2]
    dark_gt = fixed_dark_images[1:2]

    # Calculate initial mixing state (Linear)
    alpha_init = args.alpha_init
    mixed_image = bright_gt * (1.0 - alpha_init) + dark_gt * alpha_init
    init_timestep = math.ceil(alpha_init / diffusion.alteration_per_t)

    path_pred_bright = []
    path_pred_dark = []

    print(f"Generating path visualization for pair 2 over {init_timestep} timesteps...")

    with torch.no_grad():
        x_t = mixed_image.clone()
        for i in reversed(range(1, init_timestep + 1)):
            t = torch.full((1,), i, device=device, dtype=torch.long)

            predicted_bright = model(x_t, t).sample
            
            # Linear extraction
            predicted_dark = (mixed_image - (1.0 - alpha_init) * predicted_bright) / alpha_init

            path_pred_bright.append(to_uint8(predicted_bright.clone()))
            path_pred_dark.append(to_uint8(predicted_dark.clone()))

            x_t = x_t - diffusion.mix_images(predicted_bright, predicted_dark, t) + diffusion.mix_images(
                predicted_bright, predicted_dark, t - 1
            )

    bright_gt_uint8 = to_uint8(bright_gt)[0] 
    dark_gt_uint8 = to_uint8(dark_gt)[0]

    row_gt_bright = [bright_gt_uint8] * init_timestep
    row_gt_dark = [dark_gt_uint8] * init_timestep
    row_pred_bright = [p[0] for p in path_pred_bright]
    row_pred_dark = [p[0] for p in path_pred_dark]

    grid_list = row_gt_bright + row_gt_dark + row_pred_bright + row_pred_dark
    grid_tensor = torch.stack(grid_list)

    grid = torchvision.utils.make_grid(grid_tensor, nrow=init_timestep, padding=2, pad_value=255)
    ndarr = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    save_path = os.path.join(save_dir, "sampling_path_pair_2.jpg")
    Image.fromarray(ndarr).save(save_path)
    print(f"Saved successfully to: {save_path}")


def evaluate_full_validation_swaps(args):
    import torchvision
    from PIL import Image, ImageDraw, ImageFont
    
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)
    save_dir = os.path.join(base_dir, "samples", "severe_swaps")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

    val_dataloader = get_data(args, "val")
    model = get_unet(args.image_size)
    
    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )
    
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    print(f"\n--- Running Full Validation Swap Evaluation ---")
    
    total_evaluated = 0
    total_swaps = 0
    mse_swaps_count = 0
    lpips_swaps_count = 0
    overlap_count = 0
    mse_only_count = 0
    lpips_only_count = 0

    detected_swaps = []
    mismatch_swaps = []

    alpha_init = args.alpha_init

    for bright_images, dark_images in val_dataloader:
        # Move batches to GPU immediately
        bright_images = bright_images.to(device)
        dark_images = dark_images.to(device)
        
        n = len(bright_images)
        total_evaluated += n
        
        # Standard Linear Mixing
        mixed_images = bright_images * (1.0 - alpha_init) + dark_images * alpha_init
        
        # Inference
        pred_bright_uint8, pred_dark_uint8 = diffusion.sample(model, mixed_images, alpha_init)

        bright_gt_uint8 = to_uint8(bright_images)
        dark_gt_uint8 = to_uint8(dark_images)
        mixed_uint8 = to_uint8(mixed_images)

        for i in range(n):
            b_gt = bright_gt_uint8[i]
            d_gt = dark_gt_uint8[i]
            p_b = pred_bright_uint8[i]
            p_d = pred_dark_uint8[i]
            m_img = mixed_uint8[i]

            mse_pb_gtb = F.mse_loss(p_b.float(), b_gt.float()).item()
            mse_pb_gtd = F.mse_loss(p_b.float(), d_gt.float()).item()
            mse_swapped = mse_pb_gtd < mse_pb_gtb

            def prep_lpips(tensor):
                return (tensor.unsqueeze(0).float() - 127.5) / 127.5

            with torch.no_grad():
                lpips_pb_gtb = lpips_model(prep_lpips(p_b), prep_lpips(b_gt)).item()
                lpips_pb_gtd = lpips_model(prep_lpips(p_b), prep_lpips(d_gt)).item()
            
            lpips_swapped = lpips_pb_gtd < lpips_pb_gtb

            is_mismatch = (mse_swapped != lpips_swapped)

            if mse_swapped or lpips_swapped:
                total_swaps += 1
                
                if mse_swapped: mse_swaps_count += 1
                if lpips_swapped: lpips_swaps_count += 1
                
                if mse_swapped and lpips_swapped:
                    overlap_count += 1
                elif mse_swapped and not lpips_swapped:
                    mse_only_count += 1
                elif lpips_swapped and not mse_swapped:
                    lpips_only_count += 1

                severity = mse_pb_gtb - mse_pb_gtd
                
                swap_data = {
                    "severity": severity,
                    "mse_swapped": mse_swapped,
                    "lpips_swapped": lpips_swapped,
                    "mse_gap": severity,
                    "lpips_gap": lpips_pb_gtb - lpips_pb_gtd,
                    "tensors": (m_img, b_gt, p_b, d_gt, p_d)
                }

                detected_swaps.append(swap_data)
                
                if is_mismatch and len(mismatch_swaps) < 5:
                    mismatch_swaps.append(swap_data)

    # Sort swaps by severity (descending)
    detected_swaps.sort(key=lambda x: x["severity"], reverse=True)
    top_10_swaps = detected_swaps[:10]

    # --- Write the Text Report ---
    report_path = os.path.join(base_dir, "results", "identity_swap_report.txt")
    always_overlap = (mse_only_count == 0 and lpips_only_count == 0)
    overlap_status = "YES" if always_overlap else "NO"

    report = (
        f"==================================================\n"
        f"       FULL VALIDATION IDENTITY SWAP REPORT       \n"
        f"==================================================\n\n"
        f"Total Pairs Evaluated: {total_evaluated}\n"
        f"Total Swaps Detected (Any Metric): {total_swaps} ({(total_swaps / total_evaluated) * 100:.2f}%)\n\n"
        f"--- Metric Breakdown ---\n"
        f"Swaps detected by MSE: {mse_swaps_count}\n"
        f"Swaps detected by LPIPS: {lpips_swaps_count}\n\n"
        f"--- Overlap Analysis ---\n"
        f"Do LPIPS and MSE overlap every time? {overlap_status}\n"
        f"Perfect Overlaps (Both triggered): {overlap_count}\n"
        f"Discrepancy: MSE triggered, LPIPS did not: {mse_only_count}\n"
        f"Discrepancy: LPIPS triggered, MSE did not: {lpips_only_count}\n\n"
    )

    with open(report_path, "w") as f:
        f.write(report)

    # --- Scientific Grid Builder Helper ---
    def build_scientific_grid(swap_list, row_labels, title):
        if not swap_list:
            return None
        
        all_tensors = [t for swap in swap_list for t in swap["tensors"]]
        grid_tensor = torch.stack(all_tensors)
        
        # 5 columns: Mixed, GT Bright, Pred Bright, GT Dark, Pred Dark
        grid = torchvision.utils.make_grid(grid_tensor, nrow=5, padding=4, pad_value=255)
        ndarr = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img_pil = Image.fromarray(ndarr)
        
        img_w, img_h = img_pil.size
        row_h = img_h // len(swap_list)
        col_w = img_w // 5
        
        top_margin = 30
        left_margin = 160 # Room for descriptive row text
        
        final_img = Image.new("RGB", (img_w + left_margin, img_h + top_margin), "white")
        final_img.paste(img_pil, (left_margin, top_margin))
        
        draw = ImageDraw.Draw(final_img)
        
        col_headers = ["Mixed Input", "GT Bright", "Predicted Bright", "GT Dark", "Predicted Dark"]
        for idx, text in enumerate(col_headers):
            # Center approximation
            text_w = len(text) * 6 
            x = left_margin + (idx * col_w) + (col_w // 2) - (text_w // 2)
            draw.text((x, top_margin // 2 - 5), text, fill="black")
            
        for idx, text in enumerate(row_labels):
            y = top_margin + (idx * row_h) + (row_h // 2) - 15
            draw.multiline_text((10, y), text, fill="black", spacing=4)
            
        return final_img

    # --- Generate Top 10 Severe Swaps Grid ---
    if top_10_swaps:
        top_10_labels = [
            f"Rank {i+1}\nMSE Gap: {s['mse_gap']:.2f}\n" + 
            ("OVERLAP" if (s['mse_swapped'] and s['lpips_swapped']) else 
            ("MSE ONLY" if s['mse_swapped'] else "LPIPS ONLY"))
            for i, s in enumerate(top_10_swaps)
        ]
        top_10_grid = build_scientific_grid(top_10_swaps, top_10_labels, "Top 10 Severe Swaps")
        top_10_grid.save(os.path.join(save_dir, "top_10_severe_swaps.jpg"))

    # --- Generate Mismatch Examples Grid ---
    if mismatch_swaps:
        mismatch_labels = [
            f"Mismatch {i+1}\nMSE: {s['mse_swapped']}\nLPIPS: {s['lpips_swapped']}\nMSE Gap: {s['mse_gap']:.2f}\nLPIPS Gap: {s['lpips_gap']:.4f}"
            for i, s in enumerate(mismatch_swaps)
        ]
        mismatch_grid = build_scientific_grid(mismatch_swaps, mismatch_labels, "Metric Mismatches")
        mismatch_grid.save(os.path.join(save_dir, "metric_mismatch_examples.jpg"))

    print(f"Evaluation complete. Report saved to: {report_path}")
    print(f"Consolidated scientific grids saved to: {save_dir}\n")


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Path to the image dataset")
    parser.add_argument("--run_name", required=True, help="Name of the experiment folder")

    parser.add_argument("--test_images", default=1000, type=int, help="Number of distinct test images and test pairs")
    parser.add_argument("--train_samples_per_epoch", default=25000, type=int, help="Number of random train pairs per epoch")
    parser.add_argument("--num_workers", default=None, type=int, help="DataLoader worker count")

    parser.add_argument("--alpha_max", default=0.5, type=float, help="Maximum dark-image weight at the last timestep")
    parser.add_argument("--alpha_init", default=0.5, type=float, help="Dark-image weight used for previews and evaluation")
    parser.add_argument("--max_timesteps", default=250, type=int, help="Number of diffusion timesteps")
    parser.add_argument("--image_size", default=64, type=int, help="Square image size")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    
    parser.add_argument("--sample_every", default=25, type=int, help="Run iterative sampling preview every N epochs")
    parser.add_argument("--num_fixed_samples", default=10, type=int, help="Number of fixed validation pairs for previews")

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)

    #train(args)
    #eval(args)
    #one_shot_eval(args)
    #visualize_sampling_path(args)
    evaluate_full_validation_swaps(args)


if __name__ == "__main__":
    launch()

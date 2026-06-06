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
    def prep_lpips(tensor):
        return (tensor.unsqueeze(0).float() - 127.5) / 127.5

    with torch.no_grad():
        # Direct LPIPS mappings
        l_bb = lpips_model(prep_lpips(pred_bright_tensor), prep_lpips(bright_tensor)).item()
        l_dd = lpips_model(prep_lpips(pred_dark_tensor), prep_lpips(dark_tensor)).item()
        
        # Crossed LPIPS mappings
        l_bd = lpips_model(prep_lpips(pred_bright_tensor), prep_lpips(dark_tensor)).item()
        l_db = lpips_model(prep_lpips(pred_dark_tensor), prep_lpips(bright_tensor)).item()

    # Alignment Check: If the crossed mapping has a lower perceptual distance sum, a swap occurred
    is_swapped = (l_bd + l_db) < (l_bb + l_dd)

    if is_swapped:
        # Swap the numpy arrays to align identities before calculating standard metrics
        pred_bright, pred_dark = pred_dark, pred_bright
        
        # The final LPIPS values are the crossed ones
        final_lpips_bright = l_db  # Pred Dark is aligned with GT Bright
        final_lpips_dark = l_bd    # Pred Bright is aligned with GT Dark
    else:
        final_lpips_bright = l_bb
        final_lpips_dark = l_dd

    # Calculate SSIM and PSNR using the correctly aligned numpy arrays
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
    from PIL import Image
    
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

    # Ambiguity Tracking Variables
    mismatch_ambiguity_sum = 0.0
    mismatch_count = 0
    agreement_ambiguity_sum = 0.0
    agreement_count = 0

    detected_swaps = []
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

            # Metric: MSE
            mse_pb_gtb = F.mse_loss(p_b.float(), b_gt.float()).item()
            mse_pb_gtd = F.mse_loss(p_b.float(), d_gt.float()).item()
            mse_swapped = mse_pb_gtd < mse_pb_gtb

            # Metric: LPIPS
            def prep_lpips(tensor):
                return (tensor.unsqueeze(0).float() - 127.5) / 127.5

            with torch.no_grad():
                lpips_pb_gtb = lpips_model(prep_lpips(p_b), prep_lpips(b_gt)).item()
                lpips_pb_gtd = lpips_model(prep_lpips(p_b), prep_lpips(d_gt)).item()
            
            lpips_swapped = lpips_pb_gtd < lpips_pb_gtb

            # Compare Metrics
            is_mismatch = (mse_swapped != lpips_swapped)

            # Ambiguity Measurement (Distance from predictions to the original mixed state)
            ambiguity_b = F.mse_loss(p_b.float(), m_img.float()).item()
            ambiguity_d = F.mse_loss(p_d.float(), m_img.float()).item()
            avg_ambiguity = (ambiguity_b + ambiguity_d) / 2.0

            if is_mismatch:
                mismatch_ambiguity_sum += avg_ambiguity
                mismatch_count += 1
            else:
                agreement_ambiguity_sum += avg_ambiguity
                agreement_count += 1

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

                # Severity: How much closer is the prediction to the wrong target via MSE?
                severity = mse_pb_gtb - mse_pb_gtd
                
                detected_swaps.append({
                    "severity": severity,
                    "mse_gap": severity,
                    "tensors": (m_img, b_gt, p_b, d_gt, p_d)
                })

    # Sort swaps by severity (descending) and extract top 10
    detected_swaps.sort(key=lambda x: x["severity"], reverse=True)
    top_10_swaps = detected_swaps[:10]

    # --- Generate Ambiguity Stats ---
    avg_mismatch_ambiguity = (mismatch_ambiguity_sum / mismatch_count) if mismatch_count > 0 else 0
    avg_agreement_ambiguity = (agreement_ambiguity_sum / agreement_count) if agreement_count > 0 else 0

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
        f"--- Mismatch Ambiguity Analysis ---\n"
        f"Hypothesis: When metrics disagree, the model failed to confidently separate the identities, \n"
        f"leaving the predictions highly ambiguous and structurally similar to the initial mixed image.\n\n"
        f"Average MSE between Predictions and Mixed Input (When Metrics MISMATCH): {avg_mismatch_ambiguity:.2f}\n"
        f"Average MSE between Predictions and Mixed Input (When Metrics AGREE):    {avg_agreement_ambiguity:.2f}\n"
    )
    
    if mismatch_count > 0 and avg_mismatch_ambiguity < avg_agreement_ambiguity:
        report += f"\nConclusion: Mismatched pairs have a LOWER distance to the mixed input. The predictions are indeed more ambiguous.\n"
    elif mismatch_count > 0:
        report += f"\nConclusion: Mismatched pairs have a HIGHER/EQUAL distance to the mixed input. The ambiguity hypothesis is not fully supported.\n"

    with open(report_path, "w") as f:
        f.write(report)

    # --- Save Top 10 Severe Swaps (Unlabeled & Separate) ---
    for rank, swap in enumerate(top_10_swaps, start=1):
        m_img, b_gt, p_b, d_gt, p_d = swap["tensors"]
        
        # Layout: Mixed | GT Bright | Pred Bright | GT Dark | Pred Dark
        grid_tensor = torch.stack([m_img, b_gt, p_b, d_gt, p_d])
        grid = torchvision.utils.make_grid(grid_tensor, nrow=5, padding=2, pad_value=255)
        
        ndarr = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        save_file = os.path.join(save_dir, f"rank_{rank}_severity_{int(swap['severity'])}.jpg")
        Image.fromarray(ndarr).save(save_file)

    print(f"Evaluation complete. Report saved to: {report_path}")
    print(f"Top 10 severe swap images saved to: {save_dir}\n")

def evaluate_clipping_impact(args):
    import torchvision
    from PIL import Image
    
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)
    save_dir = os.path.join(base_dir, "samples", "clipping_test")
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

    # Define a custom internal sampling loop to inject the A/B testing
    def sample_custom(mixed_image, clamp_predictions=False):
        n = len(mixed_image)
        init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
        
        max_oob_val = 0.0
        oob_pixel_count = 0
        total_pixels = 0
        
        with torch.no_grad():
            x_t = mixed_image.clone().to(device)

            for i in reversed(range(1, init_timestep + 1)):
                t = torch.full((n,), i, device=device, dtype=torch.long)

                predicted_bright = model(x_t, t).sample
                predicted_dark = (mixed_image - (1.0 - args.alpha_init) * predicted_bright) / args.alpha_init

                # Track OOB severity (only meaningful on the unclamped run)
                if not clamp_predictions:
                    for p in [predicted_bright, predicted_dark]:
                        oob_mask = (p < -1.0) | (p > 1.0)
                        oob_pixel_count += oob_mask.sum().item()
                        total_pixels += p.numel()
                        
                        p_max = torch.max(torch.abs(p)).item()
                        if p_max > max_oob_val:
                            max_oob_val = p_max

                # Apply clamping if requested
                if clamp_predictions:
                    predicted_bright = torch.clamp(predicted_bright, -1.0, 1.0)
                    predicted_dark = torch.clamp(predicted_dark, -1.0, 1.0)

                x_t = x_t - diffusion.mix_images(predicted_bright, predicted_dark, t) + diffusion.mix_images(
                    predicted_bright, predicted_dark, t - 1
                )

        predicted_dark = (mixed_image - (1.0 - args.alpha_init) * x_t) / args.alpha_init
        
        # We always return the uint8 representations for metric calculation
        return to_uint8(x_t), to_uint8(predicted_dark), oob_pixel_count, total_pixels, max_oob_val

    print(f"\n--- Running OOB Clipping Impact Evaluation ---")
    
    # We will test on a fixed subset to save time (e.g., 50 images)
    subset_size = 50
    test_bright, test_dark = [], []
    for bright_images, dark_images in val_dataloader:
        test_bright.append(bright_images)
        test_dark.append(dark_images)
        if sum(b.shape[0] for b in test_bright) >= subset_size:
            break
            
    test_bright = torch.cat(test_bright)[:subset_size].to(device)
    test_dark = torch.cat(test_dark)[:subset_size].to(device)
    mixed_images = test_bright * (1.0 - args.alpha_init) + test_dark * args.alpha_init

    print(f"Testing on {subset_size} images across {math.ceil(args.alpha_init / diffusion.alteration_per_t)} timesteps...")

    # Run A/B Tests
    print("Running Baseline (Unclamped) Path...")
    base_pb_uint8, base_pd_uint8, base_oob_count, base_tot_pixels, base_max_oob = sample_custom(mixed_images, clamp_predictions=False)
    
    print("Running Experimental (Clamped) Path...")
    clamp_pb_uint8, clamp_pd_uint8, _, _, _ = sample_custom(mixed_images, clamp_predictions=True)

    bright_gt_uint8 = to_uint8(test_bright)
    dark_gt_uint8 = to_uint8(test_dark)
    
    base_metrics = {"ssim_b": [], "ssim_d": [], "psnr_b": [], "psnr_d": [], "lpips_b": [], "lpips_d": []}
    clamp_metrics = {"ssim_b": [], "ssim_d": [], "psnr_b": [], "psnr_d": [], "lpips_b": [], "lpips_d": []}

    bright_gt_np = bright_gt_uint8.cpu().permute(0, 2, 3, 1).numpy()
    dark_gt_np = dark_gt_uint8.cpu().permute(0, 2, 3, 1).numpy()
    
    base_pb_np = base_pb_uint8.cpu().permute(0, 2, 3, 1).numpy()
    base_pd_np = base_pd_uint8.cpu().permute(0, 2, 3, 1).numpy()
    clamp_pb_np = clamp_pb_uint8.cpu().permute(0, 2, 3, 1).numpy()
    clamp_pd_np = clamp_pd_uint8.cpu().permute(0, 2, 3, 1).numpy()

    # Calculate metrics for both runs
    with torch.no_grad():
        for k in range(subset_size):
            # Baseline
            sb, sd, pb, pd, lb, ld, _ = calculate_metrics(
                bright_gt_np[k], dark_gt_np[k], base_pb_np[k], base_pd_np[k],
                bright_gt_uint8[k], dark_gt_uint8[k], base_pb_uint8[k], base_pd_uint8[k], lpips_model
            )
            base_metrics["ssim_b"].append(sb); base_metrics["ssim_d"].append(sd)
            base_metrics["psnr_b"].append(pb); base_metrics["psnr_d"].append(pd)
            base_metrics["lpips_b"].append(lb); base_metrics["lpips_d"].append(ld)

            # Clamped
            sb, sd, pb, pd, lb, ld, _ = calculate_metrics(
                bright_gt_np[k], dark_gt_np[k], clamp_pb_np[k], clamp_pd_np[k],
                bright_gt_uint8[k], dark_gt_uint8[k], clamp_pb_uint8[k], clamp_pd_uint8[k], lpips_model
            )
            clamp_metrics["ssim_b"].append(sb); clamp_metrics["ssim_d"].append(sd)
            clamp_metrics["psnr_b"].append(pb); clamp_metrics["psnr_d"].append(pd)
            clamp_metrics["lpips_b"].append(lb); clamp_metrics["lpips_d"].append(ld)

    # Compile averages
    base_avg = {k: np.average(v) for k, v in base_metrics.items()}
    clamp_avg = {k: np.average(v) for k, v in clamp_metrics.items()}

    oob_percentage = (base_oob_count / base_tot_pixels) * 100

    report = (
        f"==================================================\n"
        f"        OOB PIXEL CLIPPING IMPACT REPORT          \n"
        f"==================================================\n\n"
        f"--- Out of Bounds (OOB) Severity ---\n"
        f"Does the model generate out of bounds pixels during inference?\n"
        f"Total Intermediate Pixels Evaluated: {base_tot_pixels}\n"
        f"Pixels falling outside [-1.0, 1.0]:  {base_oob_count} ({oob_percentage:.4f}%)\n"
        f"Absolute Maximum Tensor Magnitude:   {base_max_oob:.2f} (Expected max is 1.0)\n\n"
        f"--- Performance Comparison (Subset Size: {subset_size}) ---\n"
        f"             Baseline (Unclamped) | Clamped [-1, 1]  | Delta (Clamped - Base)\n"
        f"LPIPS Bright: {base_avg['lpips_b']:.4f}             | {clamp_avg['lpips_b']:.4f}           | {clamp_avg['lpips_b'] - base_avg['lpips_b']:+.4f} (Lower is better)\n"
        f"LPIPS Dark:   {base_avg['lpips_d']:.4f}             | {clamp_avg['lpips_d']:.4f}           | {clamp_avg['lpips_d'] - base_avg['lpips_d']:+.4f} (Lower is better)\n"
        f"SSIM Bright:  {base_avg['ssim_b']:.4f}             | {clamp_avg['ssim_b']:.4f}           | {clamp_avg['ssim_b'] - base_avg['ssim_b']:+.4f} (Higher is better)\n"
        f"SSIM Dark:    {base_avg['ssim_d']:.4f}             | {clamp_avg['ssim_d']:.4f}           | {clamp_avg['ssim_d'] - base_avg['ssim_d']:+.4f} (Higher is better)\n"
        f"PSNR Bright:  {base_avg['psnr_b']:.4f}             | {clamp_avg['psnr_b']:.4f}           | {clamp_avg['psnr_b'] - base_avg['psnr_b']:+.4f} (Higher is better)\n"
        f"PSNR Dark:    {base_avg['psnr_d']:.4f}             | {clamp_avg['psnr_d']:.4f}           | {clamp_avg['psnr_d'] - base_avg['psnr_d']:+.4f} (Higher is better)\n\n"
    )

    lpips_delta = (clamp_avg['lpips_b'] + clamp_avg['lpips_d']) - (base_avg['lpips_b'] + base_avg['lpips_d'])
    if lpips_delta < 0:
        report += "Conclusion: Clipping intermediate predictions consistently IMPROVED structural retrieval.\n"
    elif lpips_delta > 0:
        report += "Conclusion: Clipping intermediate predictions DAMAGED structural retrieval. The model relies on OOB vector magnitudes to traverse the latent space.\n"
    else:
        report += "Conclusion: Clipping had negligible or zero impact on final inference metrics.\n"

    report_path = os.path.join(base_dir, "results", "clipping_impact_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
        
    print(report)
    print(f"Saved report to {report_path}")

    # Generate a visual comparison grid for the first 10 pairs
    vis_count = min(10, subset_size)
    grid_tensor = torch.stack([
        to_uint8(mixed_images)[:vis_count],
        bright_gt_uint8[:vis_count],
        base_pb_uint8[:vis_count],
        clamp_pb_uint8[:vis_count],
        dark_gt_uint8[:vis_count],
        base_pd_uint8[:vis_count],
        clamp_pd_uint8[:vis_count]
    ], dim=1).view(-1, 3, args.image_size[0], args.image_size[1])

    grid = torchvision.utils.make_grid(grid_tensor, nrow=7, padding=2, pad_value=255)
    ndarr = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(ndarr).save(os.path.join(save_dir, "baseline_vs_clamped_visuals.jpg"))

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
    #evaluate_full_validation_swaps(args)
    evaluate_clipping_impact(args)


if __name__ == "__main__":
    launch()

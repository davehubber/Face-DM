import math
import os
from typing import Dict

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import optim

from utils_coupled import *

from accelerate import Accelerator
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class DifferenceColdDiffusion:
    def __init__(self, max_timesteps: int = 250, device: str = "cuda"):
        if max_timesteps <= 0:
            raise ValueError(f"max_timesteps must be positive, got {max_timesteps}")
        self.max_timesteps = int(max_timesteps)
        self.device = device

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(1, self.max_timesteps + 1, (batch_size,), device=self.device)

    def mix_ratio(self, t: torch.Tensor) -> torch.Tensor:
        return (t.float() / float(self.max_timesteps)).clamp(0.0, 1.0)[:, None, None, None]

    def average_image(self, brighter: torch.Tensor, darker: torch.Tensor) -> torch.Tensor:
        return 0.5 * (brighter + darker)

    def half_difference_image(self, brighter: torch.Tensor, darker: torch.Tensor) -> torch.Tensor:
        return 0.5 * (brighter - darker)

    def darker_from_average_and_half_difference(self, average: torch.Tensor, half_difference: torch.Tensor) -> torch.Tensor:
        return average - half_difference

    def degrade(self, half_difference: torch.Tensor, darker: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return half_difference + self.mix_ratio(t) * darker

    def degrade_from_average(self, half_difference: torch.Tensor, average: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        darker = self.darker_from_average_and_half_difference(average, half_difference)
        return self.degrade(half_difference, darker, t)

    def tacos_step(self, x_t: torch.Tensor, predicted_half_difference: torch.Tensor, average: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (
            x_t
            - self.degrade_from_average(predicted_half_difference, average, t)
            + self.degrade_from_average(predicted_half_difference, average, t - 1)
        )

    def sample(self, model, average: torch.Tensor) -> torch.Tensor:
        batch_size = average.shape[0]
        average = average.to(self.device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            x_t = average
            for i in range(self.max_timesteps, 0, -1):
                t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
                predicted_half_difference = model(x_t, t).sample
                x_t = self.tacos_step(x_t, predicted_half_difference, average, t)
        if was_training:
            model.train()
        return x_t

    def one_shot(self, model, average: torch.Tensor) -> torch.Tensor:
        average = average.to(self.device)
        t = torch.full((average.shape[0],), self.max_timesteps, device=self.device, dtype=torch.long)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            predicted_half_difference = model(average, t).sample
        if was_training:
            model.train()
        return predicted_half_difference


def get_unet(image_size):
    resolution = image_size[0] if isinstance(image_size, tuple) else image_size
    return UNet2DModel(
        sample_size=resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def ordered_l1_loss(predicted_half_difference: torch.Tensor, target_half_difference: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(predicted_half_difference, target_half_difference)


def build_fixed_validation_timesteps(batch_size: int, batch_idx: int, max_timesteps: int, device) -> torch.Tensor:
    base = torch.arange(batch_size, device=device, dtype=torch.long)
    return ((base + batch_idx * batch_size) % max_timesteps) + 1


def prepare_canonical_pair(diffusion: DifferenceColdDiffusion, image_1: torch.Tensor, image_2: torch.Tensor):
    brighter, darker, _ = order_pair_by_brightness(image_1, image_2)
    average = diffusion.average_image(brighter, darker)
    half_difference = diffusion.half_difference_image(brighter, darker)
    return brighter, darker, average, half_difference


def evaluate_validation_loss(model, dataloader, diffusion: DifferenceColdDiffusion, accelerator: Accelerator) -> float:
    model.eval()
    loss_sum = torch.zeros(1, device=accelerator.device)
    item_count = torch.zeros(1, device=accelerator.device)

    with torch.no_grad():
        for batch_idx, (images, images_add) in enumerate(dataloader):
            t = build_fixed_validation_timesteps(images.shape[0], batch_idx, diffusion.max_timesteps, accelerator.device)
            brighter, darker, _, half_difference = prepare_canonical_pair(diffusion, images, images_add)
            x_t = diffusion.degrade(half_difference, darker, t)
            predicted_half_difference = model(x_t, t).sample
            batch_loss = ordered_l1_loss(predicted_half_difference, half_difference)
            batch_size = torch.tensor([images.shape[0]], device=accelerator.device, dtype=torch.float32)
            loss_sum += batch_loss.detach() * batch_size
            item_count += batch_size

    avg_val_loss = (accelerator.gather(loss_sum).sum() / accelerator.gather(item_count).sum().clamp_min(1.0)).item()
    model.train()
    return avg_val_loss


def save_checkpoint_set(active_unet, ema_unet, base_dir: str, epoch: int, save_periodic_backup: bool):
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    active_latest_path = os.path.join(checkpoint_dir, "unet_active.pt")
    ema_latest_path = os.path.join(checkpoint_dir, "unet_ema.pt")
    torch.save(active_unet.state_dict(), active_latest_path)
    torch.save(ema_unet.state_dict(), ema_latest_path)
    if save_periodic_backup:
        torch.save(active_unet.state_dict(), os.path.join(checkpoint_dir, f"unet_active_epoch_{epoch:03d}.pt"))
        torch.save(ema_unet.state_dict(), os.path.join(checkpoint_dir, f"unet_ema_epoch_{epoch:03d}.pt"))
    return active_latest_path, ema_latest_path


def save_training_preview(unet, diffusion: DifferenceColdDiffusion, fixed_val_images, fixed_val_images_add, save_dir: str, epoch: int, save_history: bool):
    brighter, darker, average, _ = prepare_canonical_pair(diffusion, fixed_val_images, fixed_val_images_add)
    sampled_half_difference = diffusion.sample(unet, average)
    predicted_brighter, predicted_darker = reconstruct_images_from_average_and_half_difference(average, sampled_half_difference)

    preview = {
        "sampled_images": to_uint8(predicted_brighter),
        "other_images": to_uint8(predicted_darker),
        "original_images": to_uint8(brighter),
        "added_images": to_uint8(darker),
        "input_images": to_uint8(average),
    }

    latest_path = os.path.join(save_dir, "latest.jpg")
    save_images(
        preview["sampled_images"],
        preview["other_images"],
        preview["original_images"],
        preview["added_images"],
        latest_path,
        input_images=preview["input_images"],
    )

    history_path = None
    if save_history:
        history_path = os.path.join(save_dir, f"epoch_{epoch:03d}.jpg")
        save_images(
            preview["sampled_images"],
            preview["other_images"],
            preview["original_images"],
            preview["added_images"],
            history_path,
            input_images=preview["input_images"],
        )

    return latest_path, history_path, preview


def calculate_metrics(image, add_image, result_ori_image, result_add_image):
    ssim_original = structural_similarity(image, result_ori_image, data_range=255, channel_axis=-1)
    ssim_added = structural_similarity(add_image, result_add_image, data_range=255, channel_axis=-1)
    psnr_original = peak_signal_noise_ratio(image, result_ori_image, data_range=255)
    psnr_added = peak_signal_noise_ratio(add_image, result_add_image, data_range=255)
    return ssim_original, ssim_added, psnr_original, psnr_added


def evaluate_pair_outputs(
    brighter: torch.Tensor,
    darker: torch.Tensor,
    average: torch.Tensor,
    predicted_half_difference: torch.Tensor,
    lpips_model,
) -> Dict[str, object]:
    predicted_brighter, predicted_darker = reconstruct_images_from_average_and_half_difference(average, predicted_half_difference)
    predicted_brighter = predicted_brighter.clamp(-1, 1)
    predicted_darker = predicted_darker.clamp(-1, 1)

    brighter_uint8 = to_uint8(brighter)
    darker_uint8 = to_uint8(darker)
    predicted_brighter_uint8 = to_uint8(predicted_brighter)
    predicted_darker_uint8 = to_uint8(predicted_darker)
    average_uint8 = to_uint8(average)

    brighter_np = brighter_uint8.cpu().permute(0, 2, 3, 1).numpy()
    darker_np = darker_uint8.cpu().permute(0, 2, 3, 1).numpy()
    predicted_brighter_np = predicted_brighter_uint8.cpu().permute(0, 2, 3, 1).numpy()
    predicted_darker_np = predicted_darker_uint8.cpu().permute(0, 2, 3, 1).numpy()
    average_np = average_uint8.cpu().permute(0, 2, 3, 1).numpy()

    ssim_o, ssim_a, psnr_o, psnr_a, lpips_o, lpips_a = [], [], [], [], [], []
    success_count_target = 0
    success_count_deduced = 0

    with torch.no_grad():
        for k in range(len(brighter_np)):
            so, sa, po, pa = calculate_metrics(brighter_np[k], darker_np[k], predicted_brighter_np[k], predicted_darker_np[k])
            lo = lpips_model(brighter[k].unsqueeze(0), predicted_brighter[k].unsqueeze(0))
            la = lpips_model(darker[k].unsqueeze(0), predicted_darker[k].unsqueeze(0))

            ssim_avg_bright = structural_similarity(brighter_np[k], average_np[k], data_range=255, channel_axis=-1)
            ssim_avg_dark = structural_similarity(darker_np[k], average_np[k], data_range=255, channel_axis=-1)

            success_count_target += int(so > ssim_avg_bright)
            success_count_deduced += int(sa > ssim_avg_dark)

            ssim_o.append(so)
            ssim_a.append(sa)
            psnr_o.append(po)
            psnr_a.append(pa)
            lpips_o.append(lo.detach().cpu().item())
            lpips_a.append(la.detach().cpu().item())

    return {
        "images_uint8": brighter_uint8,
        "images_add_uint8": darker_uint8,
        "predicted_image_1_uint8": predicted_brighter_uint8,
        "predicted_image_2_uint8": predicted_darker_uint8,
        "average_uint8": average_uint8,
        "ssim_o": ssim_o,
        "ssim_a": ssim_a,
        "psnr_o": psnr_o,
        "psnr_a": psnr_a,
        "lpips_o": lpips_o,
        "lpips_a": lpips_a,
        "success_count_target": success_count_target,
        "success_count_deduced": success_count_deduced,
        "total_items": len(brighter_np),
    }


def run_reconstruction_eval(args, mode: str):
    if mode not in {"sample", "one_shot"}:
        raise ValueError(f"Unsupported eval mode: {mode}")

    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    test_dataloader = get_data(args, "test")
    model = get_unet(args.image_size)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = DifferenceColdDiffusion(max_timesteps=args.max_timesteps, device=device)
    lpips_model = lpips.LPIPS(net="alex").to(device)

    ssim_o, ssim_a, lpips_o, lpips_a, psnr_o, psnr_a = [], [], [], [], [], []
    success_count_target = 0
    success_count_deduced = 0
    total_items = 0
    grid_si, grid_soi, grid_i, grid_ia, grid_input = [], [], [], [], []
    collected_for_grid = 0

    for images, images_add in test_dataloader:
        brighter, darker, average, _ = prepare_canonical_pair(diffusion, images, images_add)
        if mode == "sample":
            predicted_half_difference = diffusion.sample(model, average)
        else:
            predicted_half_difference = diffusion.one_shot(model, average)

        batch_results = evaluate_pair_outputs(brighter, darker, average, predicted_half_difference, lpips_model)

        if collected_for_grid < 50:
            grid_si.append(batch_results["predicted_image_1_uint8"].cpu())
            grid_soi.append(batch_results["predicted_image_2_uint8"].cpu())
            grid_i.append(batch_results["images_uint8"].cpu())
            grid_ia.append(batch_results["images_add_uint8"].cpu())
            grid_input.append(batch_results["average_uint8"].cpu())
            collected_for_grid += len(images)

        ssim_o.extend(batch_results["ssim_o"])
        ssim_a.extend(batch_results["ssim_a"])
        psnr_o.extend(batch_results["psnr_o"])
        psnr_a.extend(batch_results["psnr_a"])
        lpips_o.extend(batch_results["lpips_o"])
        lpips_a.extend(batch_results["lpips_a"])
        success_count_target += batch_results["success_count_target"]
        success_count_deduced += batch_results["success_count_deduced"]
        total_items += batch_results["total_items"]

    sample_dir = "eval" if mode == "sample" else "one_shot"
    grid_name = "eval_grid_50.jpg" if mode == "sample" else "one_shot_grid_50.jpg"
    metrics_name = "final_metrics.txt" if mode == "sample" else "one_shot_metrics.txt"
    title = "TACoS Sampling" if mode == "sample" else "One-Shot"

    if collected_for_grid > 0:
        save_images(
            torch.cat(grid_si)[:50],
            torch.cat(grid_soi)[:50],
            torch.cat(grid_i)[:50],
            torch.cat(grid_ia)[:50],
            os.path.join(base_dir, "samples", sample_dir, grid_name),
            input_images=torch.cat(grid_input)[:50],
        )

    metrics_report = (
        f"--- {title} Evaluation Metrics (Entire Test Set) ---\n"
        f"SSIM Target: {np.average(ssim_o):.4f}\n"
        f"SSIM Deduced: {np.average(ssim_a):.4f}\n"
        f"PSNR Target: {np.average(psnr_o):.4f}\n"
        f"PSNR Deduced: {np.average(psnr_a):.4f}\n"
        f"LPIPS Target: {np.average(lpips_o):.4f}\n"
        f"LPIPS Deduced: {np.average(lpips_a):.4f}\n"
        f"Success Rate of Reversal Target (%S): {(success_count_target / total_items) * 100:.2f}%\n"
        f"Success Rate of Reversal Deduced (%S): {(success_count_deduced / total_items) * 100:.2f}%\n"
    )
    print(f"\n{metrics_report}")
    with open(os.path.join(base_dir, "results", metrics_name), "w") as f:
        f.write(metrics_report)


def train(args):
    base_dir = setup_logging(args.run_name)
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device

    train_dataloader = get_data(args, "train")
    test_dataloader = get_data(args, "test")
    model = get_unet(args.image_size)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=steps_per_epoch * args.epochs,
    )
    diffusion = DifferenceColdDiffusion(max_timesteps=args.max_timesteps, device=device)

    ema_model = EMAModel(model.parameters(), inv_gamma=1.0, power=0.75, max_value=0.9999)
    ema_model.to(device)

    if accelerator.is_main_process:
        wandb.init(project="Face-DM", name=args.run_name, config=vars(args))

    global_step = 0
    best_ema_val_loss = float("inf")
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    fixed_val_images, fixed_val_images_add = [], []
    for val_img, val_img_add in test_dataloader:
        fixed_val_images.append(val_img)
        fixed_val_images_add.append(val_img_add)
        if sum(batch.shape[0] for batch in fixed_val_images) >= args.num_fixed_samples:
            break
    fixed_val_images = torch.cat(fixed_val_images)[: args.num_fixed_samples].to(device)
    fixed_val_images_add = torch.cat(fixed_val_images_add)[: args.num_fixed_samples].to(device)

    train_loss_window_sum = 0.0
    train_loss_window_count = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss_sum = 0.0
        epoch_train_loss_count = 0

        for images, images_add in train_dataloader:
            t = diffusion.sample_timesteps(images.shape[0])
            with accelerator.accumulate(model):
                _, darker, _, half_difference = prepare_canonical_pair(diffusion, images, images_add)
                x_t = diffusion.degrade(half_difference, darker, t)
                predicted_half_difference = model(x_t, t).sample
                loss = ordered_l1_loss(predicted_half_difference, half_difference)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                ema_model.step(model.parameters())
                global_step += 1
                synced_loss = accelerator.gather(loss.detach()).mean().item()
                train_loss_window_sum += synced_loss
                train_loss_window_count += 1
                epoch_train_loss_sum += synced_loss
                epoch_train_loss_count += 1

                if global_step % args.train_log_every == 0 and accelerator.is_main_process:
                    wandb.log(
                        {
                            "train_loss": train_loss_window_sum / train_loss_window_count,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch + 1,
                            "step": global_step,
                        },
                        step=global_step,
                    )
                    train_loss_window_sum = 0.0
                    train_loss_window_count = 0

        accelerator.wait_for_everyone()

        if train_loss_window_count > 0 and accelerator.is_main_process:
            wandb.log(
                {
                    "train_loss_epoch_tail": train_loss_window_sum / train_loss_window_count,
                    "epoch": epoch + 1,
                    "step": global_step,
                },
                step=global_step,
            )
            train_loss_window_sum = 0.0
            train_loss_window_count = 0

        if (epoch + 1) % args.val_every != 0:
            continue

        unet = accelerator.unwrap_model(model)
        active_state_dict = None
        if accelerator.is_main_process:
            active_state_dict = {k: v.detach().cpu() for k, v in unet.state_dict().items()}

        accelerator.wait_for_everyone()
        ema_model.store(unet.parameters())
        ema_model.copy_to(unet.parameters())

        ema_val_loss = evaluate_validation_loss(model, test_dataloader, diffusion, accelerator)
        epoch_train_loss = epoch_train_loss_sum / max(epoch_train_loss_count, 1)
        is_best = ema_val_loss < best_ema_val_loss
        if is_best:
            best_ema_val_loss = ema_val_loss

        if accelerator.is_main_process:
            wandb.log(
                {
                    "train_loss_epoch": epoch_train_loss,
                    "ema_val_loss": ema_val_loss,
                    "best_ema_val_loss": best_ema_val_loss,
                    "epoch": epoch + 1,
                    "step": global_step,
                },
                step=global_step,
            )

            latest_preview_path, history_preview_path, preview_tensors = save_training_preview(
                unet=unet,
                diffusion=diffusion,
                fixed_val_images=fixed_val_images,
                fixed_val_images_add=fixed_val_images_add,
                save_dir=os.path.join(base_dir, "samples", "train_fixed"),
                epoch=epoch + 1,
                save_history=((epoch + 1) % args.preview_history_every == 0),
            )

            active_unet_for_save = get_unet(args.image_size)
            active_unet_for_save.load_state_dict(active_state_dict)
            active_checkpoint_path, ema_latest_path = save_checkpoint_set(
                active_unet=active_unet_for_save,
                ema_unet=unet,
                base_dir=base_dir,
                epoch=epoch + 1,
                save_periodic_backup=((epoch + 1) % args.checkpoint_every == 0),
            )

            if is_best:
                torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "unet_ema_best.pt"))
                save_images(
                    preview_tensors["sampled_images"],
                    preview_tensors["other_images"],
                    preview_tensors["original_images"],
                    preview_tensors["added_images"],
                    os.path.join(base_dir, "samples", "train_fixed", "best_ema.jpg"),
                    input_images=preview_tensors["input_images"],
                )

            path_log_payload = {
                "latest_preview_path": latest_preview_path,
                "ema_checkpoint_path": ema_latest_path,
                "active_checkpoint_path": active_checkpoint_path,
                "epoch": epoch + 1,
                "step": global_step,
            }
            if history_preview_path is not None:
                path_log_payload["history_preview_path"] = history_preview_path
            wandb.log(path_log_payload, step=global_step)

        ema_model.restore(unet.parameters())
        accelerator.wait_for_everyone()


def eval(args):
    run_reconstruction_eval(args, mode="sample")


def one_shot_eval(args):
    run_reconstruction_eval(args, mode="one_shot")


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Path to dataset")
    parser.add_argument("--run_name", required=True, help="Name of the experiment for saving models and results")
    parser.add_argument("--partition_file", default=None, help="Optional CSV file with predefined train/test pairs")
    parser.add_argument("--use_dynamic_pairs", action="store_true", help="Use on-the-fly random pair sampling")
    parser.add_argument("--train_fraction", default=0.8, type=float, help="Fraction of source images used for train")
    parser.add_argument("--split_seed", default=42, type=int, help="Seed for the image-level train/test split")
    parser.add_argument("--pair_seed", default=1234, type=int, help="Base seed for deterministic test pairs and worker init")
    parser.add_argument("--train_pairs_per_image", default=5, type=int, help="Default sampled train pairs per source image")
    parser.add_argument("--test_pairs_per_image", default=5, type=int, help="Default sampled test pairs per source image")
    parser.add_argument("--train_samples_per_epoch", default=None, type=int, help="Explicit number of train pairs per epoch")
    parser.add_argument("--test_samples", default=None, type=int, help="Explicit number of test pairs")
    parser.add_argument("--num_workers", default=None, type=int, help="DataLoader worker count")
    parser.add_argument("--max_timesteps", default=300, type=int, help="Number of diffusion timesteps")
    parser.add_argument("--image_size", default=64, type=int, help="Image size")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=3e-5, type=float, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--train_log_every", default=100, type=int, help="Log smoothed training loss every N optimizer steps")
    parser.add_argument("--val_every", default=1, type=int, help="Run EMA validation every N epochs")
    parser.add_argument("--checkpoint_every", default=25, type=int, help="Save periodic checkpoint history every N validation epochs")
    parser.add_argument("--preview_history_every", default=10, type=int, help="Save a numbered preview image every N validation epochs")
    parser.add_argument("--num_fixed_samples", default=10, type=int, help="Number of fixed validation pairs used for preview images")
    parser.add_argument("--device", default="cuda", help="Device, choose between [cuda, cpu]")

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)
    train(args)
    eval(args)
    one_shot_eval(args)


if __name__ == "__main__":
    launch()

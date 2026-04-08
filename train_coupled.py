import math
import os
from typing import Optional, Tuple

import lpips
import numpy as np
import torch
import wandb
import torch.nn.functional as F
from torch import optim

from utils_coupled import *

from accelerate import Accelerator
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class CoupledColdDiffusion:
    def __init__(self, max_timesteps=250, max_noise_std=1.0, img_size=256, device="cuda"):
        if max_timesteps <= 0:
            raise ValueError(f"max_timesteps must be positive, got {max_timesteps}")
        if max_noise_std < 0:
            raise ValueError(f"max_noise_std must be non-negative, got {max_noise_std}")

        self.max_timesteps = int(max_timesteps)
        self.max_noise_std = float(max_noise_std)
        self.img_size = img_size
        self.device = device

    def sample_timesteps(self, n: int) -> torch.Tensor:
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def mix_ratio(self, t: torch.Tensor) -> torch.Tensor:
        return (t.float() / float(self.max_timesteps))[:, None, None, None]

    def noise_std(self, t: torch.Tensor) -> torch.Tensor:
        return (self.max_noise_std * t.float() / float(self.max_timesteps))[:, None, None, None]

    def build_clean_state(self, image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
        return stack_pair_state(image_1, image_2)

    def average_image(self, image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
        return 0.5 * (image_1 + image_2)

    def duplicated_average_state(self, average_image: torch.Tensor) -> torch.Tensor:
        return torch.cat([average_image, average_image], dim=1)

    def degrade_state(
        self,
        clean_state: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if clean_state.shape[1] != 6:
            raise ValueError(f"Expected clean_state with 6 channels, got shape {tuple(clean_state.shape)}")

        duplicated_average = duplicate_average_from_state(clean_state)
        if noise is None:
            noise = torch.randn_like(clean_state)

        mix_ratio = self.mix_ratio(t)
        noise_std = self.noise_std(t)
        return (1.0 - mix_ratio) * clean_state + mix_ratio * duplicated_average + noise_std * noise

    def estimate_noise(self, x_t: torch.Tensor, predicted_clean_state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        duplicated_average = duplicate_average_from_state(predicted_clean_state)
        mix_ratio = self.mix_ratio(t)
        noise_std = self.noise_std(t)
        deterministic_part = (1.0 - mix_ratio) * predicted_clean_state + mix_ratio * duplicated_average
        return (x_t - deterministic_part) / noise_std.clamp_min(1e-8)

    def tacos_step(
        self,
        x_t: torch.Tensor,
        predicted_clean_state: torch.Tensor,
        t: torch.Tensor,
        estimated_noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            x_t
            - self.degrade_state(predicted_clean_state, t, noise=estimated_noise)
            + self.degrade_state(predicted_clean_state, t - 1, noise=estimated_noise)
        )

    def sample(
        self,
        model,
        average_image: torch.Tensor,
        start_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = len(average_image)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            duplicated_average = self.duplicated_average_state(average_image.to(self.device))
            if start_noise is None:
                start_noise = torch.randn_like(duplicated_average)
            else:
                start_noise = start_noise.to(self.device)

            final_timestep = torch.full((n,), self.max_timesteps, device=self.device, dtype=torch.long)
            x_t = duplicated_average + self.noise_std(final_timestep) * start_noise

            for i in reversed(range(1, self.max_timesteps + 1)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)
                predicted_clean_state = model(x_t, t).sample
                estimated_noise = self.estimate_noise(x_t, predicted_clean_state, t)
                x_t = self.tacos_step(x_t, predicted_clean_state, t, estimated_noise)

        if was_training:
            model.train()
        return x_t

    def one_shot(self, model, average_image: torch.Tensor, start_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = len(average_image)
        final_timestep = torch.full((n,), self.max_timesteps, device=self.device, dtype=torch.long)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            duplicated_average = self.duplicated_average_state(average_image.to(self.device))
            if start_noise is None:
                start_noise = torch.randn_like(duplicated_average)
            else:
                start_noise = start_noise.to(self.device)

            x_t = duplicated_average + self.noise_std(final_timestep) * start_noise
            predicted_clean_state = model(x_t, final_timestep).sample

        if was_training:
            model.train()
        return predicted_clean_state



def get_unet(image_size):
    resolution = image_size[0] if isinstance(image_size, tuple) else image_size
    return UNet2DModel(
        sample_size=resolution,
        in_channels=6,
        out_channels=6,
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



def build_fixed_validation_timesteps(batch_size, batch_idx, max_timesteps, device):
    base = torch.arange(batch_size, device=device, dtype=torch.long)
    return ((base + batch_idx * batch_size) % max_timesteps) + 1



def deterministic_noise(shape: Tuple[int, ...], seed: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(shape, generator=generator, dtype=dtype)
    return noise.to(device=device)



def permutation_invariant_l1_loss(predicted_state: torch.Tensor, target_image_1: torch.Tensor, target_image_2: torch.Tensor, beta: float = 1.0):
    predicted_image_1, predicted_image_2 = split_pair_state(predicted_state)

    def pair_loss(p1, t1, p2, t2):
        l1 = F.smooth_l1_loss(p1, t1, reduction='none', beta=beta).flatten(1).mean(1)
        l2 = F.smooth_l1_loss(p2, t2, reduction='none', beta=beta).flatten(1).mean(1)
        return l1 + l2

    loss_12 = pair_loss(predicted_image_1, target_image_1, predicted_image_2, target_image_2)
    loss_21 = pair_loss(predicted_image_1, target_image_2, predicted_image_2, target_image_1)

    per_sample_loss = torch.minimum(loss_12, loss_21)
    return per_sample_loss.mean()



def evaluate_validation_loss(model, dataloader, diffusion, accelerator, noise_seed: int):
    model.eval()
    loss_sum = torch.zeros(1, device=accelerator.device)
    loss_count = torch.zeros(1, device=accelerator.device)

    with torch.no_grad():
        for batch_idx, (val_images, val_images_add) in enumerate(dataloader):
            val_t = build_fixed_validation_timesteps(
                batch_size=val_images.shape[0],
                batch_idx=batch_idx,
                max_timesteps=diffusion.max_timesteps,
                device=accelerator.device,
            )
            clean_state = diffusion.build_clean_state(val_images, val_images_add)
            val_noise = deterministic_noise(
                tuple(clean_state.shape),
                seed=noise_seed + batch_idx,
                device=accelerator.device,
                dtype=clean_state.dtype,
            )
            val_x_t = diffusion.degrade_state(clean_state, val_t, noise=val_noise)
            val_pred = model(val_x_t, val_t).sample

            batch_loss = permutation_invariant_l1_loss(val_pred, val_images, val_images_add)
            batch_size = torch.tensor([val_images.shape[0]], device=accelerator.device, dtype=torch.float32)
            loss_sum += batch_loss.detach() * batch_size
            loss_count += batch_size

    gathered_loss_sum = accelerator.gather(loss_sum).sum()
    gathered_loss_count = accelerator.gather(loss_count).sum()
    avg_val_loss = (gathered_loss_sum / gathered_loss_count.clamp_min(1.0)).item()
    model.train()
    return avg_val_loss



def save_training_preview(
    unet,
    diffusion,
    fixed_val_images,
    fixed_val_images_add,
    fixed_preview_noise,
    save_dir,
    epoch,
    save_history,
):
    average_image = diffusion.average_image(fixed_val_images, fixed_val_images_add)
    sampled_state = diffusion.sample(unet, average_image, start_noise=fixed_preview_noise)
    aligned_image_1, aligned_image_2, _ = align_prediction_to_targets(sampled_state, fixed_val_images, fixed_val_images_add)

    f_images = to_uint8(fixed_val_images)
    f_images_add = to_uint8(fixed_val_images_add)
    fixed_average_uint8 = to_uint8(average_image)
    aligned_image_1_uint8 = to_uint8(aligned_image_1)
    aligned_image_2_uint8 = to_uint8(aligned_image_2)

    latest_path = os.path.join(save_dir, "latest.jpg")
    save_images(
        aligned_image_1_uint8,
        aligned_image_2_uint8,
        f_images,
        f_images_add,
        latest_path,
        input_images=fixed_average_uint8,
    )

    history_path = None
    if save_history:
        history_path = os.path.join(save_dir, f"epoch_{epoch:03d}.jpg")
        save_images(
            aligned_image_1_uint8,
            aligned_image_2_uint8,
            f_images,
            f_images_add,
            history_path,
            input_images=fixed_average_uint8,
        )

    preview_tensors = {
        "sampled_images": aligned_image_1_uint8,
        "other_images": aligned_image_2_uint8,
        "original_images": f_images,
        "added_images": f_images_add,
        "input_images": fixed_average_uint8,
    }
    return latest_path, history_path, preview_tensors



def save_checkpoint_set(active_unet, ema_unet, base_dir, epoch, save_periodic_backup):
    checkpoint_dir = os.path.join(base_dir, "checkpoints")

    active_latest_path = os.path.join(checkpoint_dir, "unet_active.pt")
    ema_latest_path = os.path.join(checkpoint_dir, "unet_ema.pt")

    torch.save(active_unet.state_dict(), active_latest_path)
    torch.save(ema_unet.state_dict(), ema_latest_path)

    if save_periodic_backup:
        torch.save(active_unet.state_dict(), os.path.join(checkpoint_dir, f"unet_active_epoch_{epoch:03d}.pt"))
        torch.save(ema_unet.state_dict(), os.path.join(checkpoint_dir, f"unet_ema_epoch_{epoch:03d}.pt"))

    return active_latest_path, ema_latest_path



def train(args):
    base_dir = setup_logging(args.run_name)

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
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

    diffusion = CoupledColdDiffusion(
        max_timesteps=args.max_timesteps,
        max_noise_std=args.max_noise_std,
        img_size=args.image_size,
        device=device,
    )

    ema_model = EMAModel(
        model.parameters(),
        inv_gamma=1.0,
        power=0.75,
        max_value=0.9999,
    )
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
        if sum(x.shape[0] for x in fixed_val_images) >= args.num_fixed_samples:
            break
    fixed_val_images = torch.cat(fixed_val_images)[: args.num_fixed_samples].to(device)
    fixed_val_images_add = torch.cat(fixed_val_images_add)[: args.num_fixed_samples].to(device)
    fixed_preview_noise = deterministic_noise(
        (fixed_val_images.shape[0], 6, fixed_val_images.shape[2], fixed_val_images.shape[3]),
        seed=args.preview_noise_seed,
        device=device,
        dtype=fixed_val_images.dtype,
    )

    train_loss_window_sum = 0.0
    train_loss_window_count = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss_sum = 0.0
        epoch_train_loss_count = 0

        for images, images_add in train_dataloader:
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            with accelerator.accumulate(model):
                clean_state = diffusion.build_clean_state(images, images_add)
                x_t = diffusion.degrade_state(clean_state, t)
                predicted_state = model(x_t, t).sample
                loss = permutation_invariant_l1_loss(predicted_state, images, images_add)

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

        ema_val_loss = evaluate_validation_loss(
            model=model,
            dataloader=test_dataloader,
            diffusion=diffusion,
            accelerator=accelerator,
            noise_seed=args.validation_noise_seed,
        )
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

            preview_saved_in_history = (epoch + 1) % args.preview_history_every == 0
            latest_preview_path, history_preview_path, preview_tensors = save_training_preview(
                unet=unet,
                diffusion=diffusion,
                fixed_val_images=fixed_val_images,
                fixed_val_images_add=fixed_val_images_add,
                fixed_preview_noise=fixed_preview_noise,
                save_dir=os.path.join(base_dir, "samples", "train_fixed"),
                epoch=epoch + 1,
                save_history=preview_saved_in_history,
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
                best_preview_path = os.path.join(base_dir, "samples", "train_fixed", "best_ema.jpg")
                save_images(
                    preview_tensors["sampled_images"],
                    preview_tensors["other_images"],
                    preview_tensors["original_images"],
                    preview_tensors["added_images"],
                    best_preview_path,
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



def calculate_metrics(image, add_image, result_ori_image, result_add_image):
    ssim_original = structural_similarity(image, result_ori_image, data_range=255, channel_axis=-1)
    ssim_added = structural_similarity(add_image, result_add_image, data_range=255, channel_axis=-1)
    psnr_original = peak_signal_noise_ratio(image, result_ori_image, data_range=255)
    psnr_added = peak_signal_noise_ratio(add_image, result_add_image, data_range=255)
    return ssim_original, ssim_added, psnr_original, psnr_added



def _evaluate_pair_outputs(
    images: torch.Tensor,
    images_add: torch.Tensor,
    predicted_state: torch.Tensor,
    average_image: torch.Tensor,
    lpips_model,
):
    aligned_image_1, aligned_image_2, _ = align_prediction_to_targets(predicted_state, images, images_add)

    images_uint8 = to_uint8(images)
    images_add_uint8 = to_uint8(images_add)
    aligned_image_1_uint8 = to_uint8(aligned_image_1)
    aligned_image_2_uint8 = to_uint8(aligned_image_2)
    average_uint8 = to_uint8(average_image)

    images_np = images_uint8.cpu().permute(0, 2, 3, 1).numpy()
    aligned_image_1_np = aligned_image_1_uint8.cpu().permute(0, 2, 3, 1).numpy()
    images_add_np = images_add_uint8.cpu().permute(0, 2, 3, 1).numpy()
    aligned_image_2_np = aligned_image_2_uint8.cpu().permute(0, 2, 3, 1).numpy()
    average_np = average_uint8.cpu().permute(0, 2, 3, 1).numpy()

    ssim_o, ssim_a, lpips_o, lpips_a, psnr_o, psnr_a = [], [], [], [], [], []
    success_count_target = 0
    success_count_deduced = 0

    with torch.no_grad():
        for k in range(len(images_np)):
            so, sa, po, pa = calculate_metrics(
                images_np[k],
                images_add_np[k],
                aligned_image_1_np[k],
                aligned_image_2_np[k],
            )
            lo = lpips_model(images[k].unsqueeze(0), aligned_image_1[k].unsqueeze(0))
            la = lpips_model(images_add[k].unsqueeze(0), aligned_image_2[k].unsqueeze(0))

            ssim_s_o = structural_similarity(images_np[k], average_np[k], data_range=255, channel_axis=-1)
            ssim_s_a = structural_similarity(images_add_np[k], average_np[k], data_range=255, channel_axis=-1)

            if so > ssim_s_o:
                success_count_target += 1
            if sa > ssim_s_a:
                success_count_deduced += 1

            ssim_o.append(so)
            ssim_a.append(sa)
            psnr_o.append(po)
            psnr_a.append(pa)
            lpips_o.append(lo.detach().cpu().item())
            lpips_a.append(la.detach().cpu().item())

    return {
        "images_uint8": images_uint8,
        "images_add_uint8": images_add_uint8,
        "predicted_image_1_uint8": aligned_image_1_uint8,
        "predicted_image_2_uint8": aligned_image_2_uint8,
        "average_uint8": average_uint8,
        "ssim_o": ssim_o,
        "ssim_a": ssim_a,
        "psnr_o": psnr_o,
        "psnr_a": psnr_a,
        "lpips_o": lpips_o,
        "lpips_a": lpips_a,
        "success_count_target": success_count_target,
        "success_count_deduced": success_count_deduced,
        "total_items": len(images_np),
    }



def eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    test_dataloader = get_data(args, "test")
    model = get_unet(args.image_size)

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = CoupledColdDiffusion(
        max_timesteps=args.max_timesteps,
        max_noise_std=args.max_noise_std,
        img_size=args.image_size,
        device=device,
    )
    lpips_model = lpips.LPIPS(net="alex").to(device)

    ssim_o, ssim_a, lpips_o, lpips_a, psnr_o, psnr_a = [], [], [], [], [], []
    success_count_target = 0
    success_count_deduced = 0
    total_items = 0

    grid_si, grid_soi, grid_i, grid_ia, grid_input = [], [], [], [], []
    collected_for_grid = 0

    for batch_idx, (images, images_add) in enumerate(test_dataloader):
        average_image = diffusion.average_image(images, images_add)
        start_noise = deterministic_noise(
            (images.shape[0], 6, images.shape[2], images.shape[3]),
            seed=args.eval_noise_seed + batch_idx,
            device=device,
            dtype=images.dtype,
        )
        sampled_state = diffusion.sample(model, average_image, start_noise=start_noise)

        batch_results = _evaluate_pair_outputs(
            images=images,
            images_add=images_add,
            predicted_state=sampled_state,
            average_image=average_image,
            lpips_model=lpips_model,
        )

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

    if collected_for_grid > 0:
        grid_si = torch.cat(grid_si)[:50]
        grid_soi = torch.cat(grid_soi)[:50]
        grid_i = torch.cat(grid_i)[:50]
        grid_ia = torch.cat(grid_ia)[:50]
        grid_input = torch.cat(grid_input)[:50]
        save_images(
            grid_si,
            grid_soi,
            grid_i,
            grid_ia,
            os.path.join(base_dir, "samples", "eval", "eval_grid_50.jpg"),
            input_images=grid_input,
        )

    avg_ssim_o = np.average(ssim_o)
    avg_ssim_a = np.average(ssim_a)
    avg_psnr_o = np.average(psnr_o)
    avg_psnr_a = np.average(psnr_a)
    avg_lpips_o = np.average(lpips_o)
    avg_lpips_a = np.average(lpips_a)
    success_rate_target = (success_count_target / total_items) * 100
    success_rate_deduced = (success_count_deduced / total_items) * 100

    metrics_report = (
        f"--- TACoS Sampling Evaluation Metrics (Entire Test Set) ---\n"
        f"SSIM Target: {avg_ssim_o:.4f}\n"
        f"SSIM Deduced: {avg_ssim_a:.4f}\n"
        f"PSNR Target: {avg_psnr_o:.4f}\n"
        f"PSNR Deduced: {avg_psnr_a:.4f}\n"
        f"LPIPS Target: {avg_lpips_o:.4f}\n"
        f"LPIPS Deduced: {avg_lpips_a:.4f}\n"
        f"Success Rate of Reversal Target (%S): {success_rate_target:.2f}%\n"
        f"Success Rate of Reversal Deduced (%S): {success_rate_deduced:.2f}%\n"
    )
    print(f"\n{metrics_report}")

    with open(os.path.join(base_dir, "results", "final_metrics.txt"), "w") as f:
        f.write(metrics_report)



def one_shot_eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    test_dataloader = get_data(args, "test")
    model = get_unet(args.image_size)

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = CoupledColdDiffusion(
        max_timesteps=args.max_timesteps,
        max_noise_std=args.max_noise_std,
        img_size=args.image_size,
        device=device,
    )
    lpips_model = lpips.LPIPS(net="alex").to(device)

    ssim_o, ssim_a, psnr_o, psnr_a, lpips_o, lpips_a = [], [], [], [], [], []
    success_count_target = 0
    success_count_deduced = 0
    total_items = 0

    grid_si, grid_soi, grid_i, grid_ia, grid_input = [], [], [], [], []
    collected_for_grid = 0

    for batch_idx, (images, images_add) in enumerate(test_dataloader):
        average_image = diffusion.average_image(images, images_add)
        start_noise = deterministic_noise(
            (images.shape[0], 6, images.shape[2], images.shape[3]),
            seed=args.one_shot_noise_seed + batch_idx,
            device=device,
            dtype=images.dtype,
        )
        predicted_state = diffusion.one_shot(model, average_image, start_noise=start_noise)

        batch_results = _evaluate_pair_outputs(
            images=images,
            images_add=images_add,
            predicted_state=predicted_state,
            average_image=average_image,
            lpips_model=lpips_model,
        )

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

    if collected_for_grid > 0:
        grid_si = torch.cat(grid_si)[:50]
        grid_soi = torch.cat(grid_soi)[:50]
        grid_i = torch.cat(grid_i)[:50]
        grid_ia = torch.cat(grid_ia)[:50]
        grid_input = torch.cat(grid_input)[:50]
        save_images(
            grid_si,
            grid_soi,
            grid_i,
            grid_ia,
            os.path.join(base_dir, "samples", "one_shot", "one_shot_grid_50.jpg"),
            input_images=grid_input,
        )

    avg_ssim_o, avg_ssim_a = np.average(ssim_o), np.average(ssim_a)
    avg_psnr_o, avg_psnr_a = np.average(psnr_o), np.average(psnr_a)
    avg_lpips_o, avg_lpips_a = np.average(lpips_o), np.average(lpips_a)
    success_rate_target = (success_count_target / total_items) * 100
    success_rate_deduced = (success_count_deduced / total_items) * 100

    metrics_report = (
        f"--- One-Shot Evaluation Metrics (Entire Test Set) ---\n"
        f"SSIM Target: {avg_ssim_o:.4f}\n"
        f"SSIM Deduced: {avg_ssim_a:.4f}\n"
        f"PSNR Target: {avg_psnr_o:.4f}\n"
        f"PSNR Deduced: {avg_psnr_a:.4f}\n"
        f"LPIPS Target: {avg_lpips_o:.4f}\n"
        f"LPIPS Deduced: {avg_lpips_a:.4f}\n"
        f"Success Rate of Reversal Target (%S): {success_rate_target:.2f}%\n"
        f"Success Rate of Reversal Deduced (%S): {success_rate_deduced:.2f}%\n"
    )

    print(f"\n{metrics_report}")

    with open(os.path.join(base_dir, "results", "one_shot_metrics.txt"), "w") as f:
        f.write(metrics_report)



def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="Path to dataset", required=True)
    parser.add_argument("--run_name", help="Name of the experiment for saving models and results", required=True)
    parser.add_argument("--partition_file", help="Optional CSV file with predefined train/test pairs", required=False, default=None)

    parser.add_argument(
        "--use_dynamic_pairs",
        action="store_true",
        help="Use on-the-fly random pair sampling from an image-level train/test split instead of a CSV pair file",
    )
    parser.add_argument(
        "--train_fraction",
        default=0.8,
        type=float,
        help="Fraction of source images assigned to the training split when using on-the-fly pairing",
    )
    parser.add_argument(
        "--split_seed",
        default=42,
        type=int,
        help="Seed used for the image-level 80/20 train/test split",
    )
    parser.add_argument(
        "--pair_seed",
        default=1234,
        type=int,
        help="Base seed used for deterministic test pairs and worker RNG initialization",
    )
    parser.add_argument(
        "--train_pairs_per_image",
        default=5,
        type=int,
        help="Default number of sampled train pairs per source image in each epoch when train_samples_per_epoch is not set",
    )
    parser.add_argument(
        "--test_pairs_per_image",
        default=5,
        type=int,
        help="Default number of sampled test/validation pairs per source image when test_samples is not set",
    )
    parser.add_argument(
        "--train_samples_per_epoch",
        default=None,
        type=int,
        help="Explicit number of random training pairs to draw per epoch when using on-the-fly pairing",
    )
    parser.add_argument(
        "--test_samples",
        default=None,
        type=int,
        help="Explicit number of deterministic test/validation pairs to evaluate when using on-the-fly pairing",
    )
    parser.add_argument(
        "--num_workers",
        default=None,
        type=int,
        help="DataLoader worker count; defaults to min(cpu_count, 8)",
    )

    parser.add_argument("--max_timesteps", default=300, type=int, help="Number of diffusion timesteps", required=False)
    parser.add_argument(
        "--max_noise_std",
        default=0.01,
        type=float,
        help="Maximum standard deviation reached by the linear Gaussian noise schedule over the full 6-channel state",
        required=False,
    )
    parser.add_argument("--image_size", default=64, type=int, help="Dimension of the images", required=False)
    parser.add_argument("--batch_size", default=16, help="Batch size", type=int, required=False)
    parser.add_argument("--epochs", default=1000, help="Number of epochs", type=int, required=False)
    parser.add_argument("--lr", default=3e-5, help="Learning rate", type=float, required=False)
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Number of gradient accumulation steps",
        required=False,
    )
    parser.add_argument(
        "--train_log_every",
        default=100,
        type=int,
        help="Log smoothed training loss every N optimizer steps",
        required=False,
    )
    parser.add_argument(
        "--val_every",
        default=1,
        type=int,
        help="Run EMA validation every N epochs",
        required=False,
    )
    parser.add_argument(
        "--checkpoint_every",
        default=25,
        type=int,
        help="Save periodic checkpoint history every N validation epochs",
        required=False,
    )
    parser.add_argument(
        "--preview_history_every",
        default=10,
        type=int,
        help="Save a numbered preview image every N validation epochs",
        required=False,
    )
    parser.add_argument(
        "--num_fixed_samples",
        default=10,
        type=int,
        help="Number of fixed validation pairs used for preview images",
        required=False,
    )
    parser.add_argument(
        "--preview_noise_seed",
        default=2024,
        type=int,
        help="Seed for the fixed preview start noise at the final diffusion timestep",
        required=False,
    )
    parser.add_argument(
        "--validation_noise_seed",
        default=31415,
        type=int,
        help="Base seed for deterministic validation noise when logging EMA validation loss",
        required=False,
    )
    parser.add_argument(
        "--eval_noise_seed",
        default=27182,
        type=int,
        help="Base seed for deterministic TACoS sampling noise during eval()",
        required=False,
    )
    parser.add_argument(
        "--one_shot_noise_seed",
        default=16180,
        type=int,
        help="Base seed for deterministic one-shot sampling noise during one_shot_eval()",
        required=False,
    )
    parser.add_argument("--device", default="cuda", help="Device, choose between [cuda, cpu]", required=False)

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)

    train(args)
    eval(args)
    one_shot_eval(args)


if __name__ == "__main__":
    launch()

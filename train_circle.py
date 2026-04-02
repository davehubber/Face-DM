
import os, torch, numpy as np, math
import torch.nn.functional as F
import wandb
import lpips
from torch import optim
from utils import *

from accelerate import Accelerator
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


class ColdDiffusion:
    """
    6-channel cold diffusion with an outside-in circular corruption mask and jittered center.

    State:
        s_t = concat(x1_t, x2_t), where an outer circular ring is progressively replaced
        by the average image in both branches.

    Forward process:
        - t = 0: no corruption
        - t = T: the entire image is corrupted to the duplicated average
        - intermediate t: pixels with distance >= r_t from the chosen center are averaged

    Radius schedule:
        r_t = (1 - t / T) * max_radius(center)

    Center handling:
        A center is sampled per sample. During training this is freshly sampled for each batch item.
        During iterative sampling the sampled centers remain fixed for the full reverse path.
    """

    def __init__(self, max_timesteps=64, img_size=64, center_jitter=6.0, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size[0] if isinstance(img_size, tuple) else img_size
        self.center_jitter = float(center_jitter)
        self.device = device
        self.yy, self.xx = self._build_coordinate_grids(self.img_size, self.device)

    def _build_coordinate_grids(self, img_size, device):
        coords = torch.arange(img_size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        return yy[None, None, :, :], xx[None, None, :, :]

    def _ensure_grids_device(self, device):
        if self.yy.device != device:
            self.yy = self.yy.to(device)
            self.xx = self.xx.to(device)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def sample_centers(self, n, device=None):
        device = device or self.device
        base = (self.img_size - 1) / 2.0
        if self.center_jitter <= 0:
            cy = torch.full((n,), base, device=device)
            cx = torch.full((n,), base, device=device)
        else:
            cy = base + (torch.rand(n, device=device) * 2.0 - 1.0) * self.center_jitter
            cx = base + (torch.rand(n, device=device) * 2.0 - 1.0) * self.center_jitter
        cy = cy.clamp(0.0, self.img_size - 1.0)
        cx = cx.clamp(0.0, self.img_size - 1.0)
        return torch.stack([cy, cx], dim=1)

    def _normalize_t(self, t, batch_size, device):
        if isinstance(t, int):
            t = torch.full((batch_size,), t, device=device, dtype=torch.long)
        elif isinstance(t, float):
            t = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
        else:
            t = t.to(device).long()
        return t.clamp_(0, self.max_timesteps)

    def _max_radius_from_centers(self, centers):
        cy = centers[:, 0]
        cx = centers[:, 1]
        h = float(self.img_size - 1)
        w = float(self.img_size - 1)
        d1 = torch.sqrt(cy ** 2 + cx ** 2)
        d2 = torch.sqrt(cy ** 2 + (w - cx) ** 2)
        d3 = torch.sqrt((h - cy) ** 2 + cx ** 2)
        d4 = torch.sqrt((h - cy) ** 2 + (w - cx) ** 2)
        return torch.maximum(torch.maximum(d1, d2), torch.maximum(d3, d4))

    def get_mask(self, centers, t, batch_size, device):
        self._ensure_grids_device(device)
        t = self._normalize_t(t, batch_size, device)
        ratio = (t.float() / float(self.max_timesteps)).view(-1, 1, 1, 1)

        cy = centers[:, 0].view(-1, 1, 1, 1)
        cx = centers[:, 1].view(-1, 1, 1, 1)
        dist = torch.sqrt((self.yy - cy) ** 2 + (self.xx - cx) ** 2)

        max_radius = self._max_radius_from_centers(centers).view(-1, 1, 1, 1)
        radius = (1.0 - ratio) * max_radius
        return (dist >= radius).float()

    def mix_images(self, image_1, image_2, t, centers):
        """
        Returns the 6-channel corrupted state:
            concat(x1_t, x2_t)
        where pixels outside the shrinking radius are replaced by the average.
        """
        batch_size = image_1.shape[0]
        device = image_1.device

        avg = 0.5 * (image_1 + image_2)
        mask = self.get_mask(centers, t, batch_size, device)

        x1_t = image_1 * (1.0 - mask) + avg * mask
        x2_t = image_2 * (1.0 - mask) + avg * mask
        return torch.cat([x1_t, x2_t], dim=1)

    def state_from_average(self, avg_image):
        return torch.cat([avg_image, avg_image], dim=1)

    def _resolve_order_from_current_state(self, current_state, pred_1, pred_2, t, centers):
        """
        Choose the branch ordering whose re-corruption best matches the current 6-channel state.
        """
        keep_state = self.mix_images(pred_1, pred_2, t, centers)
        swap_state = self.mix_images(pred_2, pred_1, t, centers)

        keep_dist = ((current_state - keep_state) ** 2).mean(dim=(1, 2, 3))
        swap_dist = ((current_state - swap_state) ** 2).mean(dim=(1, 2, 3))
        use_swap = swap_dist < keep_dist

        use_swap_expanded = use_swap[:, None, None, None]
        ordered_1 = torch.where(use_swap_expanded, pred_2, pred_1)
        ordered_2 = torch.where(use_swap_expanded, pred_1, pred_2)
        return ordered_1, ordered_2, use_swap

    def sample(self, model, average_image, track_order=True, centers=None):
        """
        Reverse process from s_T = concat(avg, avg) down to s_0, using TACOs update:
            s_{t-1} = s_t - F_t(pred) + F_{t-1}(pred)
        """
        n = len(average_image)
        model.eval()

        with torch.no_grad():
            x_t = self.state_from_average(average_image.to(self.device))
            centers = centers if centers is not None else self.sample_centers(n, device=self.device)

            for i in reversed(range(1, self.max_timesteps + 1)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)
                predicted = model(x_t, t).sample
                pred_1, pred_2 = predicted[:, :3], predicted[:, 3:]

                if track_order and i < self.max_timesteps:
                    pred_1, pred_2, _ = self._resolve_order_from_current_state(
                        x_t, pred_1, pred_2, t, centers
                    )

                x_t = (
                    x_t
                    - self.mix_images(pred_1, pred_2, t, centers)
                    + self.mix_images(pred_1, pred_2, t - 1, centers)
                )

        model.train()

        rec_1 = x_t[:, :3]
        rec_2 = x_t[:, 3:]

        rec_1 = (rec_1.clamp(-1, 1) + 1) / 2
        rec_1 = (rec_1 * 255).type(torch.uint8)

        rec_2 = (rec_2.clamp(-1, 1) + 1) / 2
        rec_2 = (rec_2 * 255).type(torch.uint8)

        return rec_1, rec_2


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


def permutation_invariant_mse(predicted, target_1, target_2):
    pred_1, pred_2 = predicted[:, :3], predicted[:, 3:]

    loss_keep = ((pred_1 - target_1) ** 2).mean(dim=(1, 2, 3)) + ((pred_2 - target_2) ** 2).mean(dim=(1, 2, 3))
    loss_swap = ((pred_1 - target_2) ** 2).mean(dim=(1, 2, 3)) + ((pred_2 - target_1) ** 2).mean(dim=(1, 2, 3))

    return torch.minimum(loss_keep, loss_swap).mean()


def reorder_to_match_targets(pred_1, pred_2, target_1, target_2):
    keep = ((pred_1 - target_1) ** 2).mean(dim=(1, 2, 3)) + ((pred_2 - target_2) ** 2).mean(dim=(1, 2, 3))
    swap = ((pred_1 - target_2) ** 2).mean(dim=(1, 2, 3)) + ((pred_2 - target_1) ** 2).mean(dim=(1, 2, 3))

    use_swap = swap < keep
    use_swap_expanded = use_swap[:, None, None, None]

    ordered_1 = torch.where(use_swap_expanded, pred_2, pred_1)
    ordered_2 = torch.where(use_swap_expanded, pred_1, pred_2)
    return ordered_1, ordered_2, use_swap


def calculate_metrics(image, add_image, result_ori_image, result_add_image):
    ssim_original = structural_similarity(image, result_ori_image, data_range=255, channel_axis=-1)
    ssim_added = structural_similarity(add_image, result_add_image, data_range=255, channel_axis=-1)
    psnr_original = peak_signal_noise_ratio(image, result_ori_image, data_range=255)
    psnr_added = peak_signal_noise_ratio(add_image, result_add_image, data_range=255)
    return ssim_original, ssim_added, psnr_original, psnr_added


def train(args):
    base_dir = setup_logging(args.run_name)

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
    )
    device = accelerator.device

    train_dataloader = get_data(args, 'train')
    test_dataloader = get_data(args, 'test')

    model = get_unet(args.image_size)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * args.epochs),
    )

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        img_size=args.image_size,
        center_jitter=args.center_jitter,
        device=device
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

    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    fixed_val_images, fixed_val_images_add = [], []
    for val_img, val_img_add in test_dataloader:
        fixed_val_images.append(val_img)
        fixed_val_images_add.append(val_img_add)
        if sum(x.shape[0] for x in fixed_val_images) >= 10:
            break

    fixed_val_images = torch.cat(fixed_val_images)[:10].to(device)
    fixed_val_images_add = torch.cat(fixed_val_images_add)[:10].to(device)
    fixed_val_average = 0.5 * (fixed_val_images + fixed_val_images_add)

    for epoch in range(args.epochs):
        model.train()
        for _, (images, images_add) in enumerate(train_dataloader):
            batch_size = images.shape[0]
            t = diffusion.sample_timesteps(batch_size).to(device)

            with accelerator.accumulate(model):
                x_t = diffusion.mix_images(images, images_add, t)
                predicted_images = model(x_t, t).sample

                loss = permutation_invariant_mse(predicted_images, images, images_add)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                ema_model.step(model.parameters())

            global_step += 1
            if global_step % 1000 == 0 and accelerator.is_main_process:
                wandb.log({
                    "train_loss": loss.item(),
                    "step": global_step,
                    "lr": lr_scheduler.get_last_lr()[0]
                })

        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for val_images, val_images_add in test_dataloader:
                    batch_size = val_images.shape[0]
                    val_t = diffusion.sample_timesteps(batch_size).to(device)
                    val_x_t = diffusion.mix_images(val_images, val_images_add, val_t)

                    val_pred = model(val_x_t, val_t).sample
                    v_loss = permutation_invariant_mse(val_pred, val_images, val_images_add)
                    v_loss = accelerator.gather(v_loss).mean()

                    val_loss += v_loss.detach()
                    val_steps += 1

            avg_val_loss = val_loss.item() / val_steps

            if accelerator.is_main_process:
                wandb.log({
                    "val_loss": avg_val_loss,
                    "epoch": epoch
                })

        if (epoch + 1) % 50 == 0 and accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)

            torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "unet_active.pt"))

            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())

            sampled_images_1, sampled_images_2 = diffusion.sample(
                unet,
                fixed_val_average,
                track_order=args.track_sampling_order
            )

            gt_1 = fixed_val_images
            gt_2 = fixed_val_images_add

            pred_1_f = sampled_images_1.float() / 127.5 - 1.0
            pred_2_f = sampled_images_2.float() / 127.5 - 1.0
            pred_1_f, pred_2_f, _ = reorder_to_match_targets(pred_1_f, pred_2_f, gt_1, gt_2)

            pred_1 = ((pred_1_f.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
            pred_2 = ((pred_2_f.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

            f_images = (gt_1.clamp(-1, 1) + 1) / 2
            f_images = (f_images * 255).type(torch.uint8)

            f_images_add = (gt_2.clamp(-1, 1) + 1) / 2
            f_images_add = (f_images_add * 255).type(torch.uint8)

            save_path = os.path.join(base_dir, "samples", "train_fixed", f"epoch_{epoch+1}.jpg")
            save_images(pred_1, pred_2, f_images, f_images_add, save_path)

            torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "unet_ema.pt"))

            ema_model.restore(unet.parameters())


def eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    test_dataloader = get_data(args, 'test')
    model = get_unet(args.image_size)

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        img_size=args.image_size,
        center_jitter=args.center_jitter,
        device=device
    )
    lpips_model = lpips.LPIPS(net='alex').to(device)

    ssim_o, ssim_a, lpips_o, lpips_a, psnr_o, psnr_a = [], [], [], [], [], []
    success_count_target = 0
    success_count_deduced = 0
    total_items = 0

    grid_si, grid_soi, grid_i, grid_ia = [], [], [], []
    collected_for_grid = 0

    for images, images_add in test_dataloader:
        avg_images = 0.5 * (images + images_add)
        sampled_images_1, sampled_images_2 = diffusion.sample(
            model,
            avg_images,
            track_order=args.track_sampling_order
        )

        pred_1 = sampled_images_1.float() / 127.5 - 1.0
        pred_2 = sampled_images_2.float() / 127.5 - 1.0
        pred_1, pred_2, _ = reorder_to_match_targets(pred_1, pred_2, images, images_add)

        sampled_images = ((pred_1.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        sampled_other_image = ((pred_2.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

        images_u8 = ((images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        images_add_u8 = ((images_add.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        avg_u8 = ((avg_images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

        if collected_for_grid < 50:
            grid_si.append(sampled_images.cpu())
            grid_soi.append(sampled_other_image.cpu())
            grid_i.append(images_u8.cpu())
            grid_ia.append(images_add_u8.cpu())
            collected_for_grid += len(images_u8)

        images_np = images_u8.cpu().permute(0, 2, 3, 1).numpy()
        sampled_images_np = sampled_images.cpu().permute(0, 2, 3, 1).numpy()
        images_add_np = images_add_u8.cpu().permute(0, 2, 3, 1).numpy()
        sampled_other_image_np = sampled_other_image.cpu().permute(0, 2, 3, 1).numpy()
        avg_np = avg_u8.cpu().permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            for k in range(len(images_np)):
                so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_images_np[k], sampled_other_image_np[k])
                lo = lpips_model((images_u8[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_images[k].unsqueeze(0).float() - 127.5) / 127.5)
                la = lpips_model((images_add_u8[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_other_image[k].unsqueeze(0).float() - 127.5) / 127.5)

                ssim_s_o = structural_similarity(images_np[k], avg_np[k], data_range=255, channel_axis=-1)
                ssim_s_a = structural_similarity(images_add_np[k], avg_np[k], data_range=255, channel_axis=-1)

                if so > ssim_s_o:
                    success_count_target += 1
                if sa > ssim_s_a:
                    success_count_deduced += 1
                total_items += 1

                ssim_o.append(so)
                ssim_a.append(sa)
                psnr_o.append(po)
                psnr_a.append(pa)
                lpips_o.append(lo.detach().cpu().numpy())
                lpips_a.append(la.detach().cpu().numpy())

    if collected_for_grid > 0:
        grid_si = torch.cat(grid_si)[:50]
        grid_soi = torch.cat(grid_soi)[:50]
        grid_i = torch.cat(grid_i)[:50]
        grid_ia = torch.cat(grid_ia)[:50]
        save_images(grid_si, grid_soi, grid_i, grid_ia, os.path.join(base_dir, "samples", "eval", "eval_grid_50.jpg"))

    avg_ssim_o = np.average(ssim_o)
    avg_ssim_a = np.average(ssim_a)
    avg_psnr_o = np.average(psnr_o)
    avg_psnr_a = np.average(psnr_a)
    avg_lpips_o = np.average(lpips_o)
    avg_lpips_a = np.average(lpips_a)
    success_rate_target = (success_count_target / total_items) * 100
    success_rate_deduced = (success_count_deduced / total_items) * 100

    metrics_report = (
        f"--- Iterative Evaluation Metrics (Entire Test Set) ---\n"
        f"SSIM Target: {avg_ssim_o:.4f}\n"
        f"SSIM Deduced: {avg_ssim_a:.4f}\n"
        f"PSNR Target: {avg_psnr_o:.4f}\n"
        f"PSNR Deduced: {avg_psnr_a:.4f}\n"
        f"LPIPS Target: {avg_lpips_o:.4f}\n"
        f"LPIPS Deduced: {avg_lpips_a:.4f}\n"
        f"Success Rate of Reversal Target (%S): {success_rate_target:.2f}%\n"
        f"Success Rate of Reversal Deduced (%S): {success_rate_deduced:.2f}%\n"
    )
    print(f'\n{metrics_report}')

    with open(os.path.join(base_dir, "results", "final_metrics.txt"), "w") as f:
        f.write(metrics_report)


def one_shot_eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    test_dataloader = get_data(args, 'test')
    model = get_unet(args.image_size)

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model_path = os.path.join(base_dir, "checkpoints", "unet_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = ColdDiffusion(
        max_timesteps=args.max_timesteps,
        img_size=args.image_size,
        center_jitter=args.center_jitter,
        device=device
    )
    lpips_model = lpips.LPIPS(net='alex').to(device)

    ssim_o, ssim_a, psnr_o, psnr_a, lpips_o, lpips_a = [], [], [], [], [], []
    success_count_target = 0
    success_count_deduced = 0
    total_items = 0

    grid_si, grid_soi, grid_i, grid_ia = [], [], [], []
    collected_for_grid = 0

    for images, images_add in test_dataloader:
        n = len(images)
        avg_images = 0.5 * (images + images_add)
        state_T = diffusion.state_from_average(avg_images)
        t = torch.full((n,), diffusion.max_timesteps, device=device, dtype=torch.long)

        with torch.no_grad():
            predicted = model(state_T, t).sample
            pred_1, pred_2 = predicted[:, :3], predicted[:, 3:]

        pred_1, pred_2, _ = reorder_to_match_targets(pred_1, pred_2, images, images_add)

        sampled_images = ((pred_1.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        sampled_other_image = ((pred_2.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

        images_u8 = ((images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        images_add_u8 = ((images_add.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        avg_u8 = ((avg_images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

        if collected_for_grid < 50:
            grid_si.append(sampled_images.cpu())
            grid_soi.append(sampled_other_image.cpu())
            grid_i.append(images_u8.cpu())
            grid_ia.append(images_add_u8.cpu())
            collected_for_grid += n

        images_np = images_u8.cpu().permute(0, 2, 3, 1).numpy()
        sampled_images_np = sampled_images.cpu().permute(0, 2, 3, 1).numpy()
        images_add_np = images_add_u8.cpu().permute(0, 2, 3, 1).numpy()
        sampled_other_image_np = sampled_other_image.cpu().permute(0, 2, 3, 1).numpy()
        avg_np = avg_u8.cpu().permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            for k in range(n):
                so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_images_np[k], sampled_other_image_np[k])
                lo = lpips_model((images_u8[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_images[k].unsqueeze(0).float() - 127.5) / 127.5)
                la = lpips_model((images_add_u8[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_other_image[k].unsqueeze(0).float() - 127.5) / 127.5)

                ssim_s_o = structural_similarity(images_np[k], avg_np[k], data_range=255, channel_axis=-1)
                ssim_s_a = structural_similarity(images_add_np[k], avg_np[k], data_range=255, channel_axis=-1)

                if so > ssim_s_o:
                    success_count_target += 1
                if sa > ssim_s_a:
                    success_count_deduced += 1
                total_items += 1

                ssim_o.append(so)
                ssim_a.append(sa)
                psnr_o.append(po)
                psnr_a.append(pa)
                lpips_o.append(lo.detach().cpu().numpy())
                lpips_a.append(la.detach().cpu().numpy())

    if collected_for_grid > 0:
        grid_si = torch.cat(grid_si)[:50]
        grid_soi = torch.cat(grid_soi)[:50]
        grid_i = torch.cat(grid_i)[:50]
        grid_ia = torch.cat(grid_ia)[:50]
        save_images(grid_si, grid_soi, grid_i, grid_ia, os.path.join(base_dir, "samples", "one_shot", "one_shot_grid_50.jpg"))

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
    parser.add_argument('--dataset_path', help='Path to dataset', required=True)
    parser.add_argument('--run_name', help='Name of the experiment for saving models and results', required=True)
    parser.add_argument('--partition_file', help='CSV file with test indexes', required=True)

    parser.add_argument('--image_size', default=64, type=int, help='Dimension of the images', required=False)
    parser.add_argument('--max_timesteps', default=64, type=int, help='Number of cold diffusion timesteps for outside-in circular corruption', required=False)
    parser.add_argument('--center_jitter', default=6.0, type=float, help='Maximum absolute center offset in pixels from the image center. Use 0 for a fixed center.', required=False)
    parser.add_argument('--track_sampling_order', action='store_true', help='Resolve branch ordering during iterative TACOs sampling', required=False)

    parser.add_argument('--batch_size', default=16, help='Batch size', type=int, required=False)
    parser.add_argument('--epochs', default=1000, help='Number of epochs', type=int, required=False)
    parser.add_argument('--lr', default=3e-4, help='Learning rate', type=float, required=False)
    parser.add_argument('--device', default='cuda', help='Device, choose between [cuda, cpu]', required=False)

    # Kept only so old command lines do not immediately break.
    parser.add_argument('--alpha_max', default=0.5, type=float, help='Unused in outside-in jittered-circle experiment', required=False)
    parser.add_argument('--alpha_init', default=0.5, type=float, help='Unused in outside-in jittered-circle experiment', required=False)

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)

    train(args)
    eval(args)
    one_shot_eval(args)


if __name__ == '__main__':
    launch()

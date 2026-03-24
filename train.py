import os, torch, numpy as np, math
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
    def __init__(self, max_timesteps=250, alpha_start=0., alpha_max=0.5, img_size=256, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps
        self.lpips_model = None

    def mix_images(self, image_1, image_2, t):
        return image_1 * (1. - self.alteration_per_t * t)[:, None, None, None] + image_2 * (self.alteration_per_t * t)[:, None, None, None]

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def _get_lpips_model(self):
        if self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()
            for p in self.lpips_model.parameters():
                p.requires_grad = False
        return self.lpips_model

    def _lpips_dist_batch(self, ref, cand):
        lpips_model = self._get_lpips_model()
        return lpips_model(
            ref.float().clamp(-1, 1),
            cand.float().clamp(-1, 1)
        ).view(-1)

    def _choose_target_by_continuity(self, preds, prev_target):
        cand_a = preds[:, :3]
        cand_b = preds[:, 3:]

        dist_a = self._lpips_dist_batch(prev_target, cand_a)
        dist_b = self._lpips_dist_batch(prev_target, cand_b)

        choose_a = dist_a <= dist_b

        curr_target = cand_b.clone()
        curr_target[choose_a] = cand_a[choose_a]

        curr_other = cand_a.clone()
        curr_other[choose_a] = cand_b[choose_a]

        return curr_target, curr_other

    def sample(self, model, superimposed_image, alpha_init=0.5):
        n = len(superimposed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)

        was_training = model.training
        model.eval()

        with torch.no_grad():
            self._get_lpips_model()

            x_t = superimposed_image.to(self.device)
            superimposed = x_t

            t_init = (torch.ones(n, device=self.device) * init_timestep).long()
            init_preds = model(x_t, t_init).sample

            p_1 = init_preds[:, :3]
            p_2 = init_preds[:, 3:]

            o_1 = (superimposed - (1. - alpha_init) * p_1) / alpha_init
            o_2 = (superimposed - alpha_init * p_2) / (1. - alpha_init)

            x_t_1 = x_t - self.mix_images(p_1, o_1, t_init) + self.mix_images(p_1, o_1, t_init - 1)
            x_t_2 = x_t - self.mix_images(p_2, o_2, t_init) + self.mix_images(p_2, o_2, t_init - 1)

            prev_target_1 = p_1.detach().clone()
            prev_target_2 = p_2.detach().clone()

            for i in reversed(range(1, init_timestep)):
                t = (torch.ones(n, device=self.device) * i).long()
                t_next = (torch.ones(n, device=self.device) * (i - 1)).long()

                preds_1 = model(x_t_1, t).sample
                p_1, _ = self._choose_target_by_continuity(preds_1, prev_target_1)
                o_1 = (superimposed - (1. - alpha_init) * p_1) / alpha_init
                x_t_1 = x_t_1 - self.mix_images(p_1, o_1, t) + self.mix_images(p_1, o_1, t_next)
                prev_target_1 = p_1.detach().clone()

                preds_2 = model(x_t_2, t).sample
                p_2, _ = self._choose_target_by_continuity(preds_2, prev_target_2)
                o_2 = (superimposed - alpha_init * p_2) / (1. - alpha_init)
                x_t_2 = x_t_2 - self.mix_images(p_2, o_2, t) + self.mix_images(p_2, o_2, t_next)
                prev_target_2 = p_2.detach().clone()

        if was_training:
            model.train()

        out_img_1 = (x_t_1.clamp(-1, 1) + 1) / 2
        out_img_1 = (out_img_1 * 255).type(torch.uint8)

        out_img_2 = (x_t_2.clamp(-1, 1) + 1) / 2
        out_img_2 = (out_img_2 * 255).type(torch.uint8)

        return out_img_1, out_img_2

def get_unet(image_size):
    resolution = image_size if isinstance(image_size, tuple) else image_size
    return UNet2DModel(
        sample_size=resolution,
        in_channels=3,
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

def permutation_invariant_mse_loss(predicted_image, image_1, image_2):
    pred_1 = predicted_image[:, :3]
    pred_2 = predicted_image[:, 3:]

    loss_direct = (
        ((pred_1 - image_1) ** 2).mean(dim=(1, 2, 3)) +
        ((pred_2 - image_2) ** 2).mean(dim=(1, 2, 3))
    )

    loss_swapped = (
        ((pred_1 - image_2) ** 2).mean(dim=(1, 2, 3)) +
        ((pred_2 - image_1) ** 2).mean(dim=(1, 2, 3))
    )

    return torch.minimum(loss_direct, loss_swapped).mean()

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
    
    diffusion = ColdDiffusion(img_size=args.image_size, device=device)

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

    for epoch in range(args.epochs):
        model.train()
        for _, (images, images_add) in enumerate(train_dataloader):
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            with accelerator.accumulate(model):
                x_t = diffusion.mix_images(images, images_add, t)
                predicted_image = model(x_t, t).sample

                loss = permutation_invariant_mse_loss(predicted_image, images, images_add)

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
                    val_t = diffusion.sample_timesteps(val_images.shape[0]).to(device)
                    val_x_t = diffusion.mix_images(val_images, val_images_add, val_t)
                    
                    val_pred = model(val_x_t, val_t).sample
                    v_loss = permutation_invariant_mse_loss(val_pred, val_images, val_images_add)
                    v_loss = accelerator.gather(v_loss.unsqueeze(0)).mean()
                    
                    val_loss += v_loss.detach()
                    val_steps += 1
                    
            avg_val_loss = val_loss.item() / val_steps
            
            if accelerator.is_main_process:
                wandb.log({
                    "val_loss": avg_val_loss,
                    "epoch": epoch + 1
                })

        if (epoch + 1) % 50 == 0 and accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)

            torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "unet_active.pt"))

            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())

            sampled_images, other_images = diffusion.sample(
                unet,
                (fixed_val_images + fixed_val_images_add) / 2.
            )

            # Convert sampled uint8 outputs back to [-1, 1] for permutation-invariant matching
            sampled_images_f = (sampled_images.float() / 255.0) * 2 - 1
            other_images_f = (other_images.float() / 255.0) * 2 - 1

            # Match predictions to the GT pair in the best order
            matched_pred_1, matched_pred_2, _ = match_by_best_mse_assignment(
                sampled_images_f,
                other_images_f,
                fixed_val_images,
                fixed_val_images_add
            )

            # Convert matched predictions back to uint8 for saving
            matched_pred_1 = ((matched_pred_1.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
            matched_pred_2 = ((matched_pred_2.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

            f_images = ((fixed_val_images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
            f_images_add = ((fixed_val_images_add.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

            save_path = os.path.join(base_dir, "samples", "train_fixed", f"epoch_{epoch+1}.jpg")
            save_images(matched_pred_1, matched_pred_2, f_images, f_images_add, save_path)

            torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "unet_ema.pt"))

            ema_model.restore(unet.parameters())

def match_by_best_mse_assignment(pred_1, pred_2, gt_1, gt_2):
    mse_direct = (
        ((pred_1 - gt_1) ** 2).mean(dim=(1, 2, 3)) +
        ((pred_2 - gt_2) ** 2).mean(dim=(1, 2, 3))
    )
    mse_swapped = (
        ((pred_1 - gt_2) ** 2).mean(dim=(1, 2, 3)) +
        ((pred_2 - gt_1) ** 2).mean(dim=(1, 2, 3))
    )

    swap_mask = mse_swapped < mse_direct
    matched_pred_1 = pred_1.clone()
    matched_pred_2 = pred_2.clone()

    matched_pred_1[swap_mask] = pred_2[swap_mask]
    matched_pred_2[swap_mask] = pred_1[swap_mask]

    return matched_pred_1, matched_pred_2, swap_mask

def calculate_metrics(image, add_image, result_ori_image, result_add_image):
    ssim_original = structural_similarity(image, result_ori_image, data_range=255, channel_axis=-1)
    ssim_added = structural_similarity(add_image, result_add_image, data_range=255, channel_axis=-1)
    psnr_original = peak_signal_noise_ratio(image, result_ori_image, data_range=255)
    psnr_added = peak_signal_noise_ratio(add_image, result_add_image, data_range=255)
    return ssim_original, ssim_added, psnr_original, psnr_added

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

    diffusion = ColdDiffusion(img_size=args.image_size, device=device)
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    ssim_1, ssim_2, lpips_1, lpips_2, psnr_1, psnr_2 = [], [], [], [], [], []
    success_count_1 = 0
    success_count_2 = 0
    total_items = 0

    grid_si, grid_soi, grid_i, grid_ia = [], [], [], []
    collected_for_grid = 0

    for images, images_add in test_dataloader:
        superimposed = images * (1 - args.alpha_init) + images_add * args.alpha_init

        with torch.no_grad():
            pred_1_uint8, pred_2_uint8 = diffusion.sample(model, superimposed, args.alpha_init)

        # Convert predictions back to [-1, 1] only for matching
        pred_1 = (pred_1_uint8.float() / 255.0) * 2 - 1
        pred_2 = (pred_2_uint8.float() / 255.0) * 2 - 1

        # Match predictions to GT with permutation-invariant assignment
        matched_pred_1, matched_pred_2, _ = match_by_best_mse_assignment(pred_1, pred_2, images, images_add)

        # Convert everything to uint8 for metrics / saving
        images_u8 = ((images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        images_add_u8 = ((images_add.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        matched_pred_1_u8 = ((matched_pred_1.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        matched_pred_2_u8 = ((matched_pred_2.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        superimposed_u8 = ((superimposed.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

        if collected_for_grid < 50:
            grid_si.append(matched_pred_1_u8.cpu())
            grid_soi.append(matched_pred_2_u8.cpu())
            grid_i.append(images_u8.cpu())
            grid_ia.append(images_add_u8.cpu())
            collected_for_grid += len(images_u8)

        images_np = images_u8.cpu().permute(0, 2, 3, 1).numpy()
        images_add_np = images_add_u8.cpu().permute(0, 2, 3, 1).numpy()
        matched_pred_1_np = matched_pred_1_u8.cpu().permute(0, 2, 3, 1).numpy()
        matched_pred_2_np = matched_pred_2_u8.cpu().permute(0, 2, 3, 1).numpy()
        superimposed_np = superimposed_u8.cpu().permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            for k in range(len(images_np)):
                s1, s2, p1, p2 = calculate_metrics(
                    images_np[k],
                    images_add_np[k],
                    matched_pred_1_np[k],
                    matched_pred_2_np[k]
                )

                l1 = lpips_model(
                    (images_u8[k].unsqueeze(0).float() - 127.5) / 127.5,
                    (matched_pred_1_u8[k].unsqueeze(0).float() - 127.5) / 127.5
                )
                l2 = lpips_model(
                    (images_add_u8[k].unsqueeze(0).float() - 127.5) / 127.5,
                    (matched_pred_2_u8[k].unsqueeze(0).float() - 127.5) / 127.5
                )

                ssim_s_1 = structural_similarity(images_np[k], superimposed_np[k], data_range=255, channel_axis=-1)
                ssim_s_2 = structural_similarity(images_add_np[k], superimposed_np[k], data_range=255, channel_axis=-1)

                if s1 > ssim_s_1:
                    success_count_1 += 1
                if s2 > ssim_s_2:
                    success_count_2 += 1
                total_items += 1

                ssim_1.append(s1)
                ssim_2.append(s2)
                psnr_1.append(p1)
                psnr_2.append(p2)
                lpips_1.append(l1.detach().cpu().item())
                lpips_2.append(l2.detach().cpu().item())

    if collected_for_grid > 0:
        grid_si = torch.cat(grid_si)[:50]
        grid_soi = torch.cat(grid_soi)[:50]
        grid_i = torch.cat(grid_i)[:50]
        grid_ia = torch.cat(grid_ia)[:50]
        save_images(grid_si, grid_soi, grid_i, grid_ia, os.path.join(base_dir, "samples", "eval", "eval_grid_50.jpg"))

    avg_ssim_1 = np.average(ssim_1)
    avg_ssim_2 = np.average(ssim_2)
    avg_psnr_1 = np.average(psnr_1)
    avg_psnr_2 = np.average(psnr_2)
    avg_lpips_1 = np.average(lpips_1)
    avg_lpips_2 = np.average(lpips_2)
    success_rate_1 = (success_count_1 / total_items) * 100
    success_rate_2 = (success_count_2 / total_items) * 100

    metrics_report = (
        f"--- Iterative Evaluation Metrics (Permutation-Invariant Matching) ---\n"
        f"SSIM Image 1: {avg_ssim_1:.4f}\n"
        f"SSIM Image 2: {avg_ssim_2:.4f}\n"
        f"PSNR Image 1: {avg_psnr_1:.4f}\n"
        f"PSNR Image 2: {avg_psnr_2:.4f}\n"
        f"LPIPS Image 1: {avg_lpips_1:.4f}\n"
        f"LPIPS Image 2: {avg_lpips_2:.4f}\n"
        f"Success Rate Image 1 (%S): {success_rate_1:.2f}%\n"
        f"Success Rate Image 2 (%S): {success_rate_2:.2f}%\n"
    )

    print(f"\n{metrics_report}")

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

    diffusion = ColdDiffusion(img_size=args.image_size, device=device)
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    ssim_1, ssim_2, psnr_1, psnr_2, lpips_1, lpips_2 = [], [], [], [], [], []
    success_count_1 = 0
    success_count_2 = 0
    total_items = 0

    grid_si, grid_soi, grid_i, grid_ia = [], [], [], []
    collected_for_grid = 0

    for images, images_add in test_dataloader:
        n = len(images)
        S = images * (1 - args.alpha_init) + images_add * args.alpha_init
        init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
        t = (torch.ones(n, device=device) * init_timestep).long()

        with torch.no_grad():
            predicted_image = model(S, t).sample
            pred_1 = predicted_image[:, :3]
            pred_2 = predicted_image[:, 3:]

        # Match predictions to GT with permutation-invariant assignment
        matched_pred_1, matched_pred_2, _ = match_by_best_mse_assignment(pred_1, pred_2, images, images_add)

        # Convert to uint8
        images_u8 = ((images.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        images_add_u8 = ((images_add.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        matched_pred_1_u8 = ((matched_pred_1.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        matched_pred_2_u8 = ((matched_pred_2.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        S_u8 = ((S.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)

        if collected_for_grid < 50:
            grid_si.append(matched_pred_1_u8.cpu())
            grid_soi.append(matched_pred_2_u8.cpu())
            grid_i.append(images_u8.cpu())
            grid_ia.append(images_add_u8.cpu())
            collected_for_grid += n

        images_np = images_u8.cpu().permute(0, 2, 3, 1).numpy()
        images_add_np = images_add_u8.cpu().permute(0, 2, 3, 1).numpy()
        matched_pred_1_np = matched_pred_1_u8.cpu().permute(0, 2, 3, 1).numpy()
        matched_pred_2_np = matched_pred_2_u8.cpu().permute(0, 2, 3, 1).numpy()
        S_np = S_u8.cpu().permute(0, 2, 3, 1).numpy()

        with torch.no_grad():
            for k in range(n):
                s1, s2, p1, p2 = calculate_metrics(
                    images_np[k],
                    images_add_np[k],
                    matched_pred_1_np[k],
                    matched_pred_2_np[k]
                )

                l1 = lpips_model(
                    (images_u8[k].unsqueeze(0).float() - 127.5) / 127.5,
                    (matched_pred_1_u8[k].unsqueeze(0).float() - 127.5) / 127.5
                )
                l2 = lpips_model(
                    (images_add_u8[k].unsqueeze(0).float() - 127.5) / 127.5,
                    (matched_pred_2_u8[k].unsqueeze(0).float() - 127.5) / 127.5
                )

                ssim_s_1 = structural_similarity(images_np[k], S_np[k], data_range=255, channel_axis=-1)
                ssim_s_2 = structural_similarity(images_add_np[k], S_np[k], data_range=255, channel_axis=-1)

                if s1 > ssim_s_1:
                    success_count_1 += 1
                if s2 > ssim_s_2:
                    success_count_2 += 1
                total_items += 1

                ssim_1.append(s1)
                ssim_2.append(s2)
                psnr_1.append(p1)
                psnr_2.append(p2)
                lpips_1.append(l1.detach().cpu().item())
                lpips_2.append(l2.detach().cpu().item())

    if collected_for_grid > 0:
        grid_si = torch.cat(grid_si)[:50]
        grid_soi = torch.cat(grid_soi)[:50]
        grid_i = torch.cat(grid_i)[:50]
        grid_ia = torch.cat(grid_ia)[:50]
        save_images(grid_si, grid_soi, grid_i, grid_ia, os.path.join(base_dir, "samples", "one_shot", "one_shot_grid_50.jpg"))

    avg_ssim_1 = np.average(ssim_1)
    avg_ssim_2 = np.average(ssim_2)
    avg_psnr_1 = np.average(psnr_1)
    avg_psnr_2 = np.average(psnr_2)
    avg_lpips_1 = np.average(lpips_1)
    avg_lpips_2 = np.average(lpips_2)
    success_rate_1 = (success_count_1 / total_items) * 100
    success_rate_2 = (success_count_2 / total_items) * 100

    metrics_report = (
        f"--- One-Shot Evaluation Metrics (Permutation-Invariant Matching) ---\n"
        f"SSIM Image 1: {avg_ssim_1:.4f}\n"
        f"SSIM Image 2: {avg_ssim_2:.4f}\n"
        f"PSNR Image 1: {avg_psnr_1:.4f}\n"
        f"PSNR Image 2: {avg_psnr_2:.4f}\n"
        f"LPIPS Image 1: {avg_lpips_1:.4f}\n"
        f"LPIPS Image 2: {avg_lpips_2:.4f}\n"
        f"Success Rate Image 1 (%S): {success_rate_1:.2f}%\n"
        f"Success Rate Image 2 (%S): {success_rate_2:.2f}%\n"
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
    parser.add_argument('--alpha_max', default=0.5, type=float, help='Maximum weight of the added image at the last time step of the forward diffusion process: alpha_max', required=False)
    parser.add_argument('--alpha_init', default=0.5, type=float, help='Weight of the added image: alpha_init', required=False)
    parser.add_argument('--image_size', default=64, type=int, help='Dimension of the images', required=False)
    parser.add_argument('--batch_size', default=16, help='Batch size', type=int, required=False)
    parser.add_argument('--epochs', default=1000, help='Number of epochs', type=int, required=False)
    parser.add_argument('--lr', default=3e-4, help='Learning rate', type=float, required=False)
    parser.add_argument('--device', default='cuda', help='Device, choose between [cuda, cpu]', required=False)

    args = parser.parse_args()
    args.image_size = (args.image_size, args.image_size)

    #train(args)
    eval(args)
    #one_shot_eval(args)

if __name__ == '__main__':
    launch()

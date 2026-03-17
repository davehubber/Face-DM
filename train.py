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
    def __init__(self, max_timesteps=250, alpha_start=0., alpha_max=0.5, img_size=256, device="cuda"):
        self.max_timesteps = max_timesteps
        self.img_size = img_size
        self.device = device
        self.alteration_per_t = (alpha_max - alpha_start) / max_timesteps

    def mix_images(self, image_1, image_2, t):
        return image_1 * (1. - self.alteration_per_t * t)[:, None, None, None] + image_2 * (self.alteration_per_t * t)[:, None, None, None]

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.max_timesteps + 1, size=(n,), device=self.device)

    def sample(self, model, superimposed_image, alpha_init=0.5):
        n = len(superimposed_image)
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        model.eval()
        with torch.no_grad():
            x_t = superimposed_image.to(self.device)

            for i in reversed(range(1, init_timestep + 1)):
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_image = model(x_t, t).sample
                
                other_image = (superimposed_image - (1. - alpha_init) * predicted_image) / alpha_init

                x_t = x_t - self.mix_images(predicted_image, other_image, t) + self.mix_images(predicted_image, other_image, t-1)
        
        model.train()

        other_image = (superimposed_image - (1 - alpha_init) * x_t) / alpha_init
        other_image = (other_image.clamp(-1, 1) + 1) / 2
        other_image = (other_image * 255).type(torch.uint8)

        x_t = (x_t.clamp(-1, 1) + 1) / 2
        x_t = (x_t * 255).type(torch.uint8)

        return x_t, other_image

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

                loss = F.mse_loss(predicted_image, images)

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
                    v_loss = F.mse_loss(val_pred, val_images)
                    
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

            sampled_images, other_images = diffusion.sample(unet, (fixed_val_images + fixed_val_images_add) / 2.)

            f_images = (fixed_val_images.clamp(-1, 1) + 1) / 2
            f_images = (f_images * 255).type(torch.uint8)
            f_images_add = (fixed_val_images_add.clamp(-1, 1) + 1) / 2
            f_images_add = (f_images_add * 255).type(torch.uint8)
            
            save_path = os.path.join(base_dir, "samples", "train_fixed", f"epoch_{epoch+1}.jpg")
            save_images(sampled_images, other_images, f_images, f_images_add, save_path)
            
            torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "unet_ema.pt"))
            
            ema_model.restore(unet.parameters())


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
    lpips_model = lpips.LPIPS(net='alex').to(device)

    ssim_o, ssim_a, lpips_o, lpips_a, psnr_o, psnr_a = [], [], [], [], [], []
    success_count = 0
    total_count = 0
    
    grid_si, grid_soi, grid_i, grid_ia = [], [], [], []
    collected_for_grid = 0

    for i, (images, images_add) in enumerate(test_dataloader):
        superimposed = images * (1 - args.alpha_init) + images_add * args.alpha_init
        sampled_images, sampled_other_image = diffusion.sample(model, superimposed, args.alpha_init)
        
        images = (images.clamp(-1, 1) + 1) / 2
        images = (images * 255).type(torch.uint8)
        
        images_add = (images_add.clamp(-1, 1) + 1) / 2
        images_add = (images_add * 255).type(torch.uint8)
        
        superimposed = (superimposed.clamp(-1, 1) + 1) / 2
        superimposed = (superimposed * 255).type(torch.uint8)

        if collected_for_grid < 50:
            grid_si.append(sampled_images.cpu())
            grid_soi.append(sampled_other_image.cpu())
            grid_i.append(images.cpu())
            grid_ia.append(images_add.cpu())
            collected_for_grid += len(images)

        images_np = images.cpu().permute(0, 2, 3, 1).numpy()
        sampled_images_np = sampled_images.cpu().permute(0, 2, 3, 1).numpy()
        images_add_np = images_add.cpu().permute(0, 2, 3, 1).numpy()
        sampled_other_image_np = sampled_other_image.cpu().permute(0, 2, 3, 1).numpy()
        superimposed_np = superimposed.cpu().permute(0, 2, 3, 1).numpy()
        
        with torch.no_grad():
            for k in range(len(images_np)):
                so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_images_np[k], sampled_other_image_np[k])
                lo = lpips_model((images[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_images[k].unsqueeze(0).float() - 127.5) / 127.5)
                la = lpips_model((images_add[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_other_image[k].unsqueeze(0).float() - 127.5) / 127.5)
                
                ssim_s_o = structural_similarity(images_np[k], superimposed_np[k], data_range=255, channel_axis=-1)
                ssim_s_a = structural_similarity(images_add_np[k], superimposed_np[k], data_range=255, channel_axis=-1)
                
                if so > ssim_s_o:
                    success_count += 1
                if sa > ssim_s_a:
                    success_count += 1
                total_count += 2

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
    success_rate = (success_count / total_count) * 100

    metrics_report = (
        f"Metrics organized by original images:\n"
        f"SSIM Original: {avg_ssim_o}\n"
        f"SSIM Added: {avg_ssim_a}\n"
        f"PSNR Original: {avg_psnr_o}\n"
        f"PSNR Added: {avg_psnr_a}\n"
        f"LPIPS Original: {avg_lpips_o}\n"
        f"LPIPS Added: {avg_lpips_a}\n"
        f"Success Rate of Reversal (%S): {success_rate:.2f}%\n"
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
    
    diffusion = ColdDiffusion(img_size=args.image_size, device=device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    ssim_o, ssim_a, psnr_o, psnr_a, lpips_o, lpips_a = [], [], [], [], [], []
    success_count = 0
    total_count = 0

    grid_si, grid_soi, grid_i, grid_ia = [], [], [], []
    collected_for_grid = 0
    
    for images, images_add in test_dataloader:
        n = len(images)
        S = images * (1 - args.alpha_init) + images_add * args.alpha_init
        init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
        t = (torch.ones(n) * init_timestep).long().to(device)
        
        with torch.no_grad():
            predicted_image = model(S, t).sample
            sampled_images = predicted_image
            sampled_other_image = (S - (1 - args.alpha_init) * sampled_images) / args.alpha_init

        images = (images.clamp(-1, 1) + 1) / 2
        images = (images * 255).type(torch.uint8)
        images_add = (images_add.clamp(-1, 1) + 1) / 2
        images_add = (images_add * 255).type(torch.uint8)
        sampled_images = (sampled_images.clamp(-1, 1) + 1) / 2
        sampled_images = (sampled_images * 255).type(torch.uint8)
        sampled_other_image = (sampled_other_image.clamp(-1, 1) + 1) / 2
        sampled_other_image = (sampled_other_image * 255).type(torch.uint8)
        
        S_formatted = (S.clamp(-1, 1) + 1) / 2
        S_formatted = (S_formatted * 255).type(torch.uint8)

        if collected_for_grid < 50:
            grid_si.append(sampled_images.cpu())
            grid_soi.append(sampled_other_image.cpu())
            grid_i.append(images.cpu())
            grid_ia.append(images_add.cpu())
            collected_for_grid += n

        images_np = images.cpu().permute(0, 2, 3, 1).numpy()
        sampled_images_np = sampled_images.cpu().permute(0, 2, 3, 1).numpy()
        images_add_np = images_add.cpu().permute(0, 2, 3, 1).numpy()
        sampled_other_image_np = sampled_other_image.cpu().permute(0, 2, 3, 1).numpy()
        S_np = S_formatted.cpu().permute(0, 2, 3, 1).numpy()
        
        with torch.no_grad():
            for k in range(n):
                so, sa, po, pa = calculate_metrics(images_np[k], images_add_np[k], sampled_images_np[k], sampled_other_image_np[k])
                lo = lpips_model((images[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_images[k].unsqueeze(0).float() - 127.5) / 127.5)
                la = lpips_model((images_add[k].unsqueeze(0).float() - 127.5) / 127.5, (sampled_other_image[k].unsqueeze(0).float() - 127.5) / 127.5)
                
                ssim_s_o = structural_similarity(images_np[k], S_np[k], data_range=255, channel_axis=-1)
                ssim_s_a = structural_similarity(images_add_np[k], S_np[k], data_range=255, channel_axis=-1)
                
                if so > ssim_s_o:
                    success_count += 1
                if sa > ssim_s_a:
                    success_count += 1
                total_count += 2
                
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
    success_rate = (success_count / total_count) * 100

    metrics_report = (
        f"--- One-Shot Evaluation Metrics (Entire Test Set) ---\n"
        f"SSIM Original: {avg_ssim_o:.4f}\n"
        f"SSIM Added: {avg_ssim_a:.4f}\n"
        f"PSNR Original: {avg_psnr_o:.4f}\n"
        f"PSNR Added: {avg_psnr_a:.4f}\n"
        f"LPIPS Original: {avg_lpips_o:.4f}\n"
        f"LPIPS Added: {avg_lpips_a:.4f}\n"
        f"Success Rate of Reversal (%S): {success_rate:.2f}%\n"
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

    train(args)
    eval(args)
    one_shot_eval(args)

if __name__ == '__main__':
    launch()
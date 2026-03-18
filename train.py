import os
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import KandinskyV22DecoderPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
import wandb
import lpips
from skimage.metrics import structural_similarity

from utils import (
    setup_logging, save_images, EmbeddingDataset, ColdDiffusionEmbeds, 
    get_prior_model, calculate_metrics, decode_batch_to_images
)

def train(args):
    base_dir = setup_logging(args.run_name)
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=1)
    device = accelerator.device

    train_dataloader = DataLoader(EmbeddingDataset(args.data_dir, 'train'), batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(EmbeddingDataset(args.data_dir, 'test'), batch_size=args.batch_size, num_workers=4)

    model = get_prior_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * args.epochs),
    )
    
    diffusion = ColdDiffusionEmbeds(device=device)

    ema_model = EMAModel(model.parameters(), inv_gamma=1.0, power=0.75, max_value=0.9999)
    ema_model.to(device)

    if accelerator.is_main_process:
        wandb.init(project="Face-DM-Embeds", name=args.run_name, config=vars(args))
    
    global_step = 0
    model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    for epoch in range(args.epochs):
        model.train()
        for e1, e2 in train_dataloader:
            t = diffusion.sample_timesteps(e1.shape[0]).to(device)

            with accelerator.accumulate(model):
                x_t = diffusion.mix_embeds(e1, e2, t)
                superimposed = (e1 + e2) / 2.0
                predicted_emb = model(hidden_states=x_t, timestep=t, proj_embedding=superimposed).predicted_image_embedding

                loss = F.mse_loss(predicted_emb, e1)

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
                wandb.log({"train_loss": loss.item(), "step": global_step, "lr": lr_scheduler.get_last_lr()[0]})

        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for v_e1, v_e2 in test_dataloader:
                    val_t = diffusion.sample_timesteps(v_e1.shape[0]).to(device)
                    val_x_t = diffusion.mix_embeds(v_e1, v_e2, val_t)
                    val_sup = (v_e1 + v_e2) / 2.0
                    
                    val_pred = model(hidden_states=val_x_t, timestep=val_t, proj_embedding=val_sup).predicted_image_embedding
                    v_loss = F.mse_loss(val_pred, v_e1)
                    v_loss = accelerator.gather(v_loss).mean()
                    val_loss += v_loss.detach() 
                    val_steps += 1
                    
            avg_val_loss = val_loss.item() / val_steps
            if accelerator.is_main_process:
                wandb.log({"val_loss": avg_val_loss, "epoch": epoch})

        if (epoch + 1) % 50 == 0 and accelerator.is_main_process:
            unet = accelerator.unwrap_model(model)
            torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "prior_active.pt"))
            ema_model.store(unet.parameters())
            ema_model.copy_to(unet.parameters())
            torch.save(unet.state_dict(), os.path.join(base_dir, "checkpoints", "prior_ema.pt"))
            ema_model.restore(unet.parameters())

def eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    dataset = EmbeddingDataset(args.data_dir, 'test')
    scale_factor = dataset.scale_factor
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    model = get_prior_model()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.load_state_dict(torch.load(os.path.join(base_dir, "checkpoints", "prior_ema.pt"), map_location=device))
    model.eval()

    diffusion = ColdDiffusionEmbeds(device=device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    decoder = KandinskyV22DecoderPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16).to(device)
    decoder.set_progress_bar_config(disable=True)

    ssim_t, ssim_d, lpips_t, lpips_d, psnr_t, psnr_d = [], [], [], [], [], []
    latent_mse_t, latent_mse_d = [], []
    success_count_target = 0
    success_count_deduced = 0
    total_count = 0
    
    grid_si, grid_soi, grid_i, grid_ia = [], [], [], []
    collected_for_grid = 0

    for e1, e2 in test_dataloader:
        superimposed = e1 * (1 - args.alpha_init) + e2 * args.alpha_init
        pred_e1, pred_e2 = diffusion.sample(model, superimposed, args.alpha_init)
        
        latent_mse_t.append(F.mse_loss(pred_e1, e1).item())
        latent_mse_d.append(F.mse_loss(pred_e2, e2).item())

        with torch.no_grad():
            i1_lpips, i1_u8 = decode_batch_to_images(e1, decoder, scale_factor)
            i2_lpips, i2_u8 = decode_batch_to_images(e2, decoder, scale_factor)
            sup_lpips, sup_u8 = decode_batch_to_images(superimposed, decoder, scale_factor)
            p1_lpips, p1_u8 = decode_batch_to_images(pred_e1, decoder, scale_factor)
            p2_lpips, p2_u8 = decode_batch_to_images(pred_e2, decoder, scale_factor)

        if collected_for_grid < 50:
            grid_si.append(p1_u8.cpu())
            grid_soi.append(p2_u8.cpu())
            grid_i.append(i1_u8.cpu())
            grid_ia.append(i2_u8.cpu())
            collected_for_grid += len(e1)

        i1_np = i1_u8.cpu().permute(0, 2, 3, 1).numpy()
        i2_np = i2_u8.cpu().permute(0, 2, 3, 1).numpy()
        sup_np = sup_u8.cpu().permute(0, 2, 3, 1).numpy()
        p1_np = p1_u8.cpu().permute(0, 2, 3, 1).numpy()
        p2_np = p2_u8.cpu().permute(0, 2, 3, 1).numpy()
        
        with torch.no_grad():
            for k in range(len(i1_np)):
                st, sd, pt, pd = calculate_metrics(i1_np[k], i2_np[k], p1_np[k], p2_np[k])
                lt = lpips_model(i1_lpips[k].unsqueeze(0), p1_lpips[k].unsqueeze(0))
                ld = lpips_model(i2_lpips[k].unsqueeze(0), p2_lpips[k].unsqueeze(0))
                
                ssim_s_t = structural_similarity(i1_np[k], sup_np[k], data_range=255, channel_axis=-1)
                ssim_s_d = structural_similarity(i2_np[k], sup_np[k], data_range=255, channel_axis=-1)
                
                if st > ssim_s_t: success_count_target += 1
                if sd > ssim_s_d: success_count_deduced += 1
                total_count += 1

                ssim_t.append(st); ssim_d.append(sd)
                psnr_t.append(pt); psnr_d.append(pd)
                lpips_t.append(lt.item()); lpips_d.append(ld.item())

    if collected_for_grid > 0:
        save_images(
            torch.cat(grid_si)[:50], torch.cat(grid_soi)[:50], 
            torch.cat(grid_i)[:50], torch.cat(grid_ia)[:50], 
            os.path.join(base_dir, "samples", "eval", "eval_grid_50.jpg")
        )

    metrics_report = (
        f"--- Regular Evaluation Metrics ---\n"
        f"Latent MSE Target: {np.mean(latent_mse_t):.4f}\n"
        f"Latent MSE Deduced: {np.mean(latent_mse_d):.4f}\n"
        f"SSIM Target: {np.mean(ssim_t):.4f} | SSIM Deduced: {np.mean(ssim_d):.4f}\n"
        f"PSNR Target: {np.mean(psnr_t):.4f} | PSNR Deduced: {np.mean(psnr_d):.4f}\n"
        f"LPIPS Target: {np.mean(lpips_t):.4f} | LPIPS Deduced: {np.mean(lpips_d):.4f}\n"
        f"Success Rate Target (%S): {(success_count_target / total_count) * 100:.2f}%\n"
        f"Success Rate Deduced (%S): {(success_count_deduced / total_count) * 100:.2f}%\n"
    )
    print(f'\n{metrics_report}')
    with open(os.path.join(base_dir, "results", "final_metrics.txt"), "w") as f:
        f.write(metrics_report)


def one_shot_eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    dataset = EmbeddingDataset(args.data_dir, 'test')
    scale_factor = dataset.scale_factor
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    model = get_prior_model()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.load_state_dict(torch.load(os.path.join(base_dir, "checkpoints", "prior_ema.pt"), map_location=device))
    model.eval()
    
    diffusion = ColdDiffusionEmbeds(device=device)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    decoder = KandinskyV22DecoderPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16).to(device)
    decoder.set_progress_bar_config(disable=True)
    
    ssim_t, ssim_d, psnr_t, psnr_d, lpips_t, lpips_d = [], [], [], [], [], []
    latent_mse_t, latent_mse_d = [], []
    success_count_target = 0
    success_count_deduced = 0
    total_count = 0

    grid_si, grid_soi, grid_i, grid_ia = [], [], [], []
    collected_for_grid = 0
    
    for e1, e2 in test_dataloader:
        n = len(e1)
        S = e1 * (1 - args.alpha_init) + e2 * args.alpha_init
        init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
        t = (torch.ones(n) * init_timestep).long().to(device)
        
        with torch.no_grad():
            pred_e1 = model(hidden_states=S, timestep=t, proj_embedding=S).predicted_image_embedding
            pred_e2 = (S - (1 - args.alpha_init) * pred_e1) / args.alpha_init

        latent_mse_t.append(F.mse_loss(pred_e1, e1).item())
        latent_mse_d.append(F.mse_loss(pred_e2, e2).item())

        with torch.no_grad():
            i1_lpips, i1_u8 = decode_batch_to_images(e1, decoder, scale_factor)
            i2_lpips, i2_u8 = decode_batch_to_images(e2, decoder, scale_factor)
            sup_lpips, sup_u8 = decode_batch_to_images(S, decoder, scale_factor)
            p1_lpips, p1_u8 = decode_batch_to_images(pred_e1, decoder, scale_factor)
            p2_lpips, p2_u8 = decode_batch_to_images(pred_e2, decoder, scale_factor)

        if collected_for_grid < 50:
            grid_si.append(p1_u8.cpu()); grid_soi.append(p2_u8.cpu())
            grid_i.append(i1_u8.cpu()); grid_ia.append(i2_u8.cpu())
            collected_for_grid += n

        i1_np = i1_u8.cpu().permute(0, 2, 3, 1).numpy()
        i2_np = i2_u8.cpu().permute(0, 2, 3, 1).numpy()
        sup_np = sup_u8.cpu().permute(0, 2, 3, 1).numpy()
        p1_np = p1_u8.cpu().permute(0, 2, 3, 1).numpy()
        p2_np = p2_u8.cpu().permute(0, 2, 3, 1).numpy()
        
        with torch.no_grad():
            for k in range(n):
                st, sd, pt, pd = calculate_metrics(i1_np[k], i2_np[k], p1_np[k], p2_np[k])
                lt = lpips_model(i1_lpips[k].unsqueeze(0), p1_lpips[k].unsqueeze(0))
                ld = lpips_model(i2_lpips[k].unsqueeze(0), p2_lpips[k].unsqueeze(0))
                
                ssim_s_t = structural_similarity(i1_np[k], sup_np[k], data_range=255, channel_axis=-1)
                ssim_s_d = structural_similarity(i2_np[k], sup_np[k], data_range=255, channel_axis=-1)
                
                if st > ssim_s_t: success_count_target += 1
                if sd > ssim_s_d: success_count_deduced += 1
                total_count += 1
                
                ssim_t.append(st); ssim_d.append(sd)
                psnr_t.append(pt); psnr_d.append(pd)
                lpips_t.append(lt.item()); lpips_d.append(ld.item())

    if collected_for_grid > 0:
        save_images(
            torch.cat(grid_si)[:50], torch.cat(grid_soi)[:50], 
            torch.cat(grid_i)[:50], torch.cat(grid_ia)[:50], 
            os.path.join(base_dir, "samples", "one_shot", "one_shot_grid_50.jpg")
        )

    metrics_report = (
        f"--- One-Shot Evaluation Metrics ---\n"
        f"Latent MSE Target: {np.mean(latent_mse_t):.4f}\n"
        f"Latent MSE Deduced: {np.mean(latent_mse_d):.4f}\n"
        f"SSIM Target: {np.mean(ssim_t):.4f} | SSIM Deduced: {np.mean(ssim_d):.4f}\n"
        f"PSNR Target: {np.mean(psnr_t):.4f} | PSNR Deduced: {np.mean(psnr_d):.4f}\n"
        f"LPIPS Target: {np.mean(lpips_t):.4f} | LPIPS Deduced: {np.mean(lpips_d):.4f}\n"
        f"Success Rate Target (%S): {(success_count_target / total_count) * 100:.2f}%\n"
        f"Success Rate Deduced (%S): {(success_count_deduced / total_count) * 100:.2f}%\n"
    )
    
    print(f"\n{metrics_report}")
    with open(os.path.join(base_dir, "results", "one_shot_metrics.txt"), "w") as f:
        f.write(metrics_report)

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Path to precomputed embeddings directory', required=True)
    parser.add_argument('--run_name', help='Name of the experiment', required=True)
    parser.add_argument('--alpha_max', default=0.5, type=float, help='Max alpha weight', required=False)
    parser.add_argument('--alpha_init', default=0.5, type=float, help='Init alpha weight', required=False)
    parser.add_argument('--batch_size', default=1024, help='Batch size', type=int, required=False)
    parser.add_argument('--epochs', default=1000, help='Number of epochs', type=int, required=False)
    parser.add_argument('--lr', default=1e-4, help='Learning rate', type=float, required=False)
    
    args = parser.parse_args()
    train(args)
    eval(args)
    one_shot_eval(args)

if __name__ == '__main__':
    launch()
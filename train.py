import os
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
import wandb

from utils import setup_logging, EmbeddingDataset, ColdDiffusionEmbeds, get_prior_model


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

    ema_model = EMAModel(model.parameters(), inv_gamma=1.0, power=0.75, decay=0.9999)
    ema_model.to(device)

    if accelerator.is_main_process:
        wandb.init(project="Face-DM", name=args.run_name, config=vars(args))

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
                predicted_emb = model(hidden_states=x_t, timestep=t, proj_embedding=superimposed).predicted_image_embedding.squeeze(1)

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

            if global_step % 100 == 0: 
                model.eval()
                val_loss = 0.0
                val_steps = 0

                with torch.no_grad():
                    for v_e1, v_e2 in test_dataloader:
                        val_t = diffusion.sample_timesteps(v_e1.shape[0]).to(device)
                        val_x_t = diffusion.mix_embeds(v_e1, v_e2, val_t)
                        val_sup = (v_e1 + v_e2) / 2.0

                        val_pred = model(hidden_states=val_x_t, timestep=val_t, proj_embedding=val_sup).predicted_image_embedding.squeeze(1)
                        v_loss = F.mse_loss(val_pred, v_e1)
                        v_loss = accelerator.gather(v_loss).mean()
                        val_loss += v_loss.item()
                        val_steps += 1

                avg_val_loss = val_loss / val_steps

                if accelerator.is_main_process:
                    wandb.log({
                        "train_loss": loss.item(), 
                        "val_loss": avg_val_loss,
                        "step": global_step, 
                        "epoch": epoch,
                        "lr": optimizer.param_groups['lr']
                    })
                
                model.train()

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
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model = get_prior_model()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.load_state_dict(torch.load(os.path.join(base_dir, "checkpoints", "prior_ema.pt"), map_location=device, weights_only=True))
    model.eval()

    diffusion = ColdDiffusionEmbeds(device=device)
    cos_sim_t_list, cos_sim_d_list = [], []
    top1_acc_t_list, top1_acc_d_list = [], []

    for e1, e2 in test_dataloader:
        superimposed = e1 * (1 - args.alpha_init) + e2 * args.alpha_init
        pred_e1, pred_e2 = diffusion.sample(model, superimposed, args.alpha_init)

        cos_sim_t_list.append(F.cosine_similarity(pred_e1, e1, dim=-1).mean().item())
        cos_sim_d_list.append(F.cosine_similarity(pred_e2, e2, dim=-1).mean().item())

        pred_e1_norm = F.normalize(pred_e1, dim=-1)
        e1_norm = F.normalize(e1, dim=-1)
        sim_matrix_t = pred_e1_norm @ e1_norm.T
        top1_t = (sim_matrix_t.argmax(dim=1) == torch.arange(e1.shape[0], device=device)).float().mean().item()
        top1_acc_t_list.append(top1_t)

        pred_e2_norm = F.normalize(pred_e2, dim=-1)
        e2_norm = F.normalize(e2, dim=-1)
        sim_matrix_d = pred_e2_norm @ e2_norm.T
        top1_d = (sim_matrix_d.argmax(dim=1) == torch.arange(e2.shape[0], device=device)).float().mean().item()
        top1_acc_d_list.append(top1_d)

    metrics_report = (
        f"--- Regular Evaluation Metrics ---\n"
        f"Cosine Sim Target: {np.mean(cos_sim_t_list):.4f}\n"
        f"Cosine Sim Deduced: {np.mean(cos_sim_d_list):.4f}\n"
        f"Top-1 Acc Target: {np.mean(top1_acc_t_list):.4f}\n"
        f"Top-1 Acc Deduced: {np.mean(top1_acc_d_list):.4f}\n"
    )
    print(f'\n{metrics_report}')
    with open(os.path.join(base_dir, "results", "final_metrics.txt"), "w") as f:
        f.write(metrics_report)

def one_shot_eval(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    dataset = EmbeddingDataset(args.data_dir, 'test')
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size)

    model = get_prior_model()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    model.load_state_dict(torch.load(os.path.join(base_dir, "checkpoints", "prior_ema.pt"), map_location=device, weights_only=True))
    model.eval()

    diffusion = ColdDiffusionEmbeds(device=device)
    cos_sim_t_list, cos_sim_d_list = [], []
    top1_acc_t_list, top1_acc_d_list = [], []

    for e1, e2 in test_dataloader:
        n = len(e1)
        S = e1 * (1 - args.alpha_init) + e2 * args.alpha_init
        init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
        t = (torch.ones(n) * init_timestep).long().to(device)

        with torch.no_grad():
            pred_e1 = model(hidden_states=S, timestep=t, proj_embedding=S).predicted_image_embedding.squeeze(1)
            pred_e2 = (S - (1 - args.alpha_init) * pred_e1) / args.alpha_init

        cos_sim_t_list.append(F.cosine_similarity(pred_e1, e1, dim=-1).mean().item())
        cos_sim_d_list.append(F.cosine_similarity(pred_e2, e2, dim=-1).mean().item())

        pred_e1_norm = F.normalize(pred_e1, dim=-1)
        e1_norm = F.normalize(e1, dim=-1)
        sim_matrix_t = pred_e1_norm @ e1_norm.T
        top1_t = (sim_matrix_t.argmax(dim=1) == torch.arange(n, device=device)).float().mean().item()
        top1_acc_t_list.append(top1_t)

        pred_e2_norm = F.normalize(pred_e2, dim=-1)
        e2_norm = F.normalize(e2, dim=-1)
        sim_matrix_d = pred_e2_norm @ e2_norm.T
        top1_d = (sim_matrix_d.argmax(dim=1) == torch.arange(n, device=device)).float().mean().item()
        top1_acc_d_list.append(top1_d)

    metrics_report = (
        f"--- One-Shot Evaluation Metrics ---\n"
        f"Cosine Sim Target: {np.mean(cos_sim_t_list):.4f}\n"
        f"Cosine Sim Deduced: {np.mean(cos_sim_d_list):.4f}\n"
        f"Top-1 Acc Target: {np.mean(top1_acc_t_list):.4f}\n"
        f"Top-1 Acc Deduced: {np.mean(top1_acc_d_list):.4f}\n"
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

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch import optim

from utils_semantic import get_data, setup_logging


class TimeEmbedding(nn.Module):
    def __init__(self, num_time_emb_channels: int, out_channels: int):
        super().__init__()
        self.num_time_emb_channels = num_time_emb_channels
        self.net = nn.Sequential(
            nn.Linear(num_time_emb_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.num_time_emb_channels // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.num_time_emb_channels % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.net(emb)


class MLPSkipBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_channels: int, use_condition: bool, use_activation: bool):
        super().__init__()
        self.use_condition = use_condition
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels) if use_activation else nn.Identity()
        self.act = nn.SiLU() if use_activation else nn.Identity()
        self.time_scale = nn.Sequential(nn.SiLU(), nn.Linear(condition_channels, out_channels)) if use_condition else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.use_condition:
            x = x * (1.0 + self.time_scale(cond))
        x = self.norm(x)
        return self.act(x)


class MLPSkipNet(nn.Module):
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 2048, num_layers: int = 10, num_time_emb_channels: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.skip_layers = set(range(1, num_layers))
        self.time_embed = TimeEmbedding(num_time_emb_channels=num_time_emb_channels, out_channels=embedding_dim)

        layers = []
        for i in range(num_layers):
            if i == 0:
                in_dim = embedding_dim
                out_dim = hidden_dim
                use_condition = True
                use_activation = True
            elif i == num_layers - 1:
                in_dim = hidden_dim + embedding_dim
                out_dim = embedding_dim
                use_condition = False
                use_activation = False
            else:
                in_dim = hidden_dim + embedding_dim
                out_dim = hidden_dim
                use_condition = True
                use_activation = True

            layers.append(
                MLPSkipBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    condition_channels=embedding_dim,
                    use_condition=use_condition,
                    use_activation=use_activation,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = self.time_embed(t)
        h = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_layers:
                h = torch.cat([h, x], dim=1)
            h = layer(h, cond)
        return h


class ColdDiffusionEmbeddings:
    def __init__(self, max_timesteps: int = 300, device: str = "cuda"):
        self.max_timesteps = int(max_timesteps)
        self.device = device

    def degrade(self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha = (t.float() / self.max_timesteps).unsqueeze(1)
        return (1.0 - alpha) * x_0 + alpha * x_T

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(1, self.max_timesteps + 1, (batch_size,), device=self.device, dtype=torch.long)

    def sample(self, model: nn.Module, x_T: torch.Tensor) -> torch.Tensor:
        initial_noise_std = 1e-4

        batch_size = x_T.shape[0]
        model.eval()

        with torch.no_grad():
            x_T = x_T.to(self.device)

            if initial_noise_std > 0:
                x_t = x_T + initial_noise_std * torch.randn_like(x_T)
            else:
                x_t = x_T

            for i in reversed(range(1, self.max_timesteps + 1)):
                t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)

                # Predict clean x_0
                pred_x0 = model(x_t, t)

                # Deterministic reverse step formula.
                # Important: keep using the original clean x_T here,
                # not the noise-perturbed x_t.
                D_t = self.degrade(pred_x0, x_T, t)
                D_t_prev = self.degrade(pred_x0, x_T, t - 1)
                x_t = x_t - D_t + D_t_prev

        model.train()
        return pred_x0


def permutation_invariant_loss(pred_x0: torch.Tensor, target_x0: torch.Tensor) -> torch.Tensor:
    # x0 shapes are [Batch, 512]. We split them into the two 256-dimensional PCA chunks
    x0_A, x0_B = target_x0.chunk(2, dim=-1)

    # Create the alternative flipped target
    target_x0_flipped = torch.cat([x0_B, x0_A], dim=-1)

    # Compute L1 Loss for both valid permutations
    loss_standard = F.l1_loss(pred_x0, target_x0, reduction="none").mean(dim=-1)
    loss_flipped = F.l1_loss(pred_x0, target_x0_flipped, reduction="none").mean(dim=-1)

    # Take the minimum error per sample in the batch
    min_loss = torch.min(loss_standard, loss_flipped)
    return min_loss.mean()


@torch.no_grad()
def evaluate_validation_loss(model: nn.Module, dataloader, diffusion: ColdDiffusionEmbeddings, accelerator: Accelerator):
    model.eval()
    loss_sum = torch.zeros(1, device=accelerator.device)
    loss_count = torch.zeros(1, device=accelerator.device)

    for batch_idx, batch in enumerate(dataloader):
        x_0 = batch["x_0"]
        x_T = batch["x_T"]

        t = ((torch.arange(x_0.shape[0], device=accelerator.device) + batch_idx * x_0.shape[0]) % diffusion.max_timesteps) + 1
        x_t = diffusion.degrade(x_0, x_T, t)

        pred_x0 = model(x_t, t)

        batch_loss = permutation_invariant_loss(pred_x0, x_0)
        loss_sum += batch_loss * x_0.shape[0]
        loss_count += x_0.shape[0]

    avg_val_loss = (accelerator.gather(loss_sum).sum() / accelerator.gather(loss_count).sum()).item()
    model.train()
    return avg_val_loss


@torch.no_grad()
def evaluate_embedding_l1(model: nn.Module, dataloader, diffusion: ColdDiffusionEmbeddings, accelerator: Accelerator, one_shot: bool = False):
    model.eval()
    total_sum = torch.zeros(1, device=accelerator.device)
    total_count = torch.zeros(1, device=accelerator.device)

    first_item = None

    for batch in dataloader:
        x_0 = batch["x_0"]
        x_T = batch["x_T"]

        if one_shot:
            t = torch.full((x_0.shape[0],), diffusion.max_timesteps, device=accelerator.device, dtype=torch.long)
            pred_x0 = model(x_T, t)
        else:
            pred_x0 = diffusion.sample(model, x_T)

        batch_loss = permutation_invariant_loss(pred_x0, x_0)
        total_sum += batch_loss * x_0.shape[0]
        total_count += x_0.shape[0]

        if first_item is None:
            first_item = {
                "x_T_input": x_T[0].detach().cpu(),
                "predicted_x0": pred_x0[0].detach().cpu(),
                "target_x0": x_0[0].detach().cpu(),
                "source_path_A": batch["source_path_A"][0],
                "source_path_B": batch["source_path_B"][0],
                "sample_id_A": batch["sample_id_A"][0],
                "sample_id_B": batch["sample_id_B"][0],
            }

    total_l1 = (accelerator.gather(total_sum).sum() / accelerator.gather(total_count).sum()).item()
    model.train()
    return {
        "total_l1": total_l1,
        "first_item": first_item,
    }


def train(args):
    base_dir = setup_logging(args.run_name)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device

    train_dataloader = get_data(args, "train")
    val_dataloader = get_data(args, "val")

    model = MLPSkipNet(
        embedding_dim=512,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_time_emb_channels=args.num_time_emb_channels,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=steps_per_epoch * args.epochs,
    )

    diffusion = ColdDiffusionEmbeddings(
        max_timesteps=args.max_timesteps,
        device=device,
    )

    ema_model = EMAModel(model.parameters(), inv_gamma=1.0, power=0.75, max_value=0.9999)
    ema_model.to(device)

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss_sum = 0.0
        epoch_train_loss_count = 0

        for batch in train_dataloader:
            x_0 = batch["x_0"]
            x_T = batch["x_T"]
            t = diffusion.sample_timesteps(x_0.shape[0])

            with accelerator.accumulate(model):
                x_t = diffusion.degrade(x_0, x_T, t)
                pred_x0 = model(x_t, t)

                loss = permutation_invariant_loss(pred_x0, x_0)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                ema_model.step(model.parameters())
                epoch_train_loss_sum += accelerator.gather(loss.detach()).mean().item()
                epoch_train_loss_count += 1

        accelerator.wait_for_everyone()

        if (epoch + 1) % args.val_every != 0:
            continue

        epoch_train_loss = epoch_train_loss_sum / max(epoch_train_loss_count, 1)
        unwrapped_model = accelerator.unwrap_model(model)

        ema_model.store(unwrapped_model.parameters())
        ema_model.copy_to(unwrapped_model.parameters())
        val_loss = evaluate_validation_loss(model, val_dataloader, diffusion, accelerator)
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        if accelerator.is_main_process:
            wandb.log({"train_l1": epoch_train_loss, "val_l1": val_loss}, step=epoch + 1)
            torch.save(unwrapped_model.state_dict(), os.path.join(base_dir, "checkpoints", "mlp_ema.pt"))
            if is_best:
                torch.save(unwrapped_model.state_dict(), os.path.join(base_dir, "checkpoints", "mlp_ema_best.pt"))

        ema_model.restore(unwrapped_model.parameters())
        accelerator.wait_for_everyone()


def eval_model(args, one_shot: bool = False):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    val_dataloader = get_data(args, "val")

    model = MLPSkipNet(
        embedding_dim=512,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_time_emb_channels=args.num_time_emb_channels,
    )

    model, val_dataloader = accelerator.prepare(model, val_dataloader)
    model_path = os.path.join(base_dir, "checkpoints", "mlp_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = ColdDiffusionEmbeddings(
        max_timesteps=args.max_timesteps,
        device=device,
    )

    metrics = evaluate_embedding_l1(
        model=model,
        dataloader=val_dataloader,
        diffusion=diffusion,
        accelerator=accelerator,
        one_shot=one_shot,
    )

    if accelerator.is_main_process:
        label = "One-Shot" if one_shot else "Iterative"
        report = (
            f"--- {label} Evaluation (Validation Set) ---\n"
            f"Permutation-Invariant Min L1 Error: {metrics['total_l1']:.8f}\n"
        )
        print(f"\n{report}")

        out_name = "one_shot_metrics.txt" if one_shot else "final_metrics.txt"
        with open(os.path.join(base_dir, "results", out_name), "w", encoding="utf-8") as f:
            f.write(report)

        if not one_shot and metrics["first_item"] is not None:
            save_path = os.path.join(base_dir, "results", "decode_pair_data.pt")
            torch.save(metrics["first_item"], save_path)
            print(f"Saved evaluation embeddings for visual decoding to: {save_path}")


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="encoded_ffhq256_semantic_split", help="Folder containing semantic/train_zsem.pt and semantic/val_zsem.pt")
    parser.add_argument("--run_name", required=True, help="Name of the experiment folder")

    parser.add_argument("--n_components", default=256, type=int, help="Number of PCA components to form half of x_0")
    parser.add_argument("--train_samples_per_epoch", default=1000000, type=int, help="Number of random train pairs per epoch")
    parser.add_argument("--val_samples", default=100000, type=int, help="Number of deterministic validation pairs")
    parser.add_argument("--num_workers", default=4, type=int, help="DataLoader worker count")

    parser.add_argument("--max_timesteps", default=300, type=int, help="Number of diffusion timesteps")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--epochs", default=150, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="AdamW weight decay")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--val_every", default=1, type=int, help="Run validation every N epochs")
    parser.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"], help="Accelerate mixed precision mode")
    parser.add_argument("--num_warmup_steps", default=500, type=int, help="Scheduler warmup steps")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Gradient clipping norm")
    parser.add_argument("--wandb_project", default="Face-DM", help="Weights & Biases project name")

    parser.add_argument("--hidden_dim", default=2048, type=int, help="Hidden width of the latent MLP")
    parser.add_argument("--num_layers", default=10, type=int, help="Number of MLP layers")
    parser.add_argument("--num_time_emb_channels", default=64, type=int, help="Sinusoidal timestep embedding width")

    args = parser.parse_args()

    #train(args)
    #eval_model(args, one_shot=False)
    eval_model(args, one_shot=True)


if __name__ == "__main__":
    launch()
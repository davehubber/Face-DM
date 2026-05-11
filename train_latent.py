import argparse
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
from torch import optim

from utils_latent import get_data, setup_logging


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
            -math.log(10000)
            * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.num_time_emb_channels % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.net(emb)


class MLPSkipBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_channels: int,
        use_condition: bool,
        use_activation: bool,
    ):
        super().__init__()
        self.use_condition = use_condition
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels) if use_activation else nn.Identity()
        self.act = nn.SiLU() if use_activation else nn.Identity()
        self.time_scale = (
            nn.Sequential(nn.SiLU(), nn.Linear(condition_channels, out_channels))
            if use_condition
            else None
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.use_condition:
            x = x * (1.0 + self.time_scale(cond))
        x = self.norm(x)
        return self.act(x)


class MLPSkipNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 2048,
        num_layers: int = 10,
        num_time_emb_channels: int = 64,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.in_features = embedding_dim * 2
        self.skip_layers = set(range(1, num_layers))
        
        self.time_embed = TimeEmbedding(
            num_time_emb_channels=num_time_emb_channels,
            out_channels=embedding_dim,
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        layers = []
        for i in range(num_layers):
            if i == 0:
                in_dim = self.in_features
                out_dim = hidden_dim
                use_condition = True
                use_activation = True
            elif i == num_layers - 1:
                in_dim = hidden_dim + self.in_features
                out_dim = self.in_features
                use_condition = False
                use_activation = False
            else:
                in_dim = hidden_dim + self.in_features
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

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c)
        cond = t_emb + c_emb  # Morph guidance injected into timestep embedding
        h = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_layers:
                h = torch.cat([h, x], dim=1)
            h = layer(h, cond)
        return h


class GaussianDiffusion:
    def __init__(self, num_timesteps: int = 1000, device: str = "cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Standard linear schedule
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t - 1])[:, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t - 1])[:, None]
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

    @torch.no_grad()
    def p_sample_ddim(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        alpha_bar_t = self.alphas_cumprod[t - 1][:, None]
        alpha_bar_t_prev = torch.where(
            t > 1, 
            self.alphas_cumprod[t - 2], 
            torch.ones_like(self.alphas_cumprod[t - 1])
        )[:, None]

        pred_noise = model(x_t, t, c)
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * pred_noise
        x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
        return x_prev

    @torch.no_grad()
    def sample(self, model: nn.Module, c: torch.Tensor) -> torch.Tensor:
        batch_size = c.shape[0]
        embedding_dim = c.shape[1]
        x_t = torch.randn((batch_size, embedding_dim * 2), device=self.device)
        
        for i in reversed(range(1, self.num_timesteps + 1)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x_t = self.p_sample_ddim(model, x_t, t, c)
            
        return x_t


def _l1_mean_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.abs(x - y).reshape(x.shape[0], -1).mean(dim=1)


def permutation_invariant_mse_loss(
    predicted_noise: torch.Tensor,
    true_noise: torch.Tensor,
) -> torch.Tensor:
    dim = predicted_noise.shape[1] // 2

    pred_1, pred_2 = predicted_noise[:, :dim], predicted_noise[:, dim:]
    true_1, true_2 = true_noise[:, :dim], true_noise[:, dim:]

    mse_11 = F.mse_loss(pred_1, true_1, reduction='none').mean(dim=1)
    mse_22 = F.mse_loss(pred_2, true_2, reduction='none').mean(dim=1)
    loss_A = mse_11 + mse_22

    mse_12 = F.mse_loss(pred_1, true_2, reduction='none').mean(dim=1)
    mse_21 = F.mse_loss(pred_2, true_1, reduction='none').mean(dim=1)
    loss_B = mse_12 + mse_21

    return torch.minimum(loss_A, loss_B).mean()


def permutation_invariant_pair_l1_cosine_sums(
    predicted_1: torch.Tensor,
    predicted_2: torch.Tensor,
    clean_1: torch.Tensor,
    clean_2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    p1_to_c1 = _l1_mean_per_sample(predicted_1, clean_1)
    p2_to_c2 = _l1_mean_per_sample(predicted_2, clean_2)

    p1_to_c2 = _l1_mean_per_sample(predicted_1, clean_2)
    p2_to_c1 = _l1_mean_per_sample(predicted_2, clean_1)

    config_1_total_l1 = p1_to_c1 + p2_to_c2
    config_2_total_l1 = p1_to_c2 + p2_to_c1

    use_config_1 = config_1_total_l1 <= config_2_total_l1

    total_l1 = torch.where(use_config_1, config_1_total_l1, config_2_total_l1)

    p1_to_c1_cos = F.cosine_similarity(predicted_1, clean_1, dim=-1)
    p2_to_c2_cos = F.cosine_similarity(predicted_2, clean_2, dim=-1)
    p1_to_c2_cos = F.cosine_similarity(predicted_1, clean_2, dim=-1)
    p2_to_c1_cos = F.cosine_similarity(predicted_2, clean_1, dim=-1)

    total_cosine = torch.where(
        use_config_1,
        p1_to_c1_cos + p2_to_c2_cos,
        p1_to_c2_cos + p2_to_c1_cos,
    )

    return total_l1.sum(), total_cosine.sum()


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    dataloader,
    diffusion: GaussianDiffusion,
    accelerator: Accelerator,
):
    model.eval()

    loss_sum = torch.zeros(1, device=accelerator.device)
    loss_count = torch.zeros(1, device=accelerator.device)

    for batch_idx, batch in enumerate(dataloader):
        clean_embeddings_1 = batch["clean_embedding_1"]
        clean_embeddings_2 = batch["clean_embedding_2"]
        batch_size = clean_embeddings_1.shape[0]

        x_0 = torch.cat([clean_embeddings_1, clean_embeddings_2], dim=1)
        c = 0.5 * (clean_embeddings_1 + clean_embeddings_2)
        
        t = torch.randint(1, diffusion.num_timesteps + 1, (batch_size,), device=accelerator.device)
        noise = torch.randn_like(x_0)
        
        x_t = diffusion.q_sample(x_0, t, noise)
        predicted_noise = model(x_t, t, c)

        loss_sum += permutation_invariant_mse_loss(
            predicted_noise=predicted_noise,
            true_noise=noise
        ).detach() * batch_size

        loss_count += batch_size

    avg_val_loss = (
        accelerator.gather(loss_sum).sum()
        / accelerator.gather(loss_count).sum()
    ).item()

    model.train()
    return avg_val_loss


@torch.no_grad()
def evaluate_embedding_metrics(
    model: nn.Module,
    dataloader,
    diffusion: GaussianDiffusion,
    accelerator: Accelerator,
):
    model.eval()

    total_l1_sum = torch.zeros(1, device=accelerator.device)
    total_cosine_sum = torch.zeros(1, device=accelerator.device)
    total_count = torch.zeros(1, device=accelerator.device)

    first_item = None
    embedding_dim = model.module.embedding_dim if hasattr(model, 'module') else model.embedding_dim

    for batch in dataloader:
        clean_1 = batch["clean_embedding_1"]
        clean_2 = batch["clean_embedding_2"]
        batch_size = clean_1.shape[0]

        c = 0.5 * (clean_1 + clean_2)
        predicted_x0 = diffusion.sample(model, c)
        
        predicted_1 = predicted_x0[:, :embedding_dim]
        predicted_2 = predicted_x0[:, embedding_dim:]

        total_l1_batch_sum, total_cosine_batch_sum = permutation_invariant_pair_l1_cosine_sums(
            predicted_1, predicted_2, clean_1, clean_2
        )

        total_l1_sum += total_l1_batch_sum
        total_cosine_sum += total_cosine_batch_sum
        total_count += batch_size

        if first_item is None:
            first_item = {
                "mixed_embedding": c.detach().cpu(),
                "predicted_embedding_1": predicted_1.detach().cpu(),
                "predicted_embedding_2": predicted_2.detach().cpu(),
                "clean_embedding_1": clean_1.detach().cpu(),
                "clean_embedding_2": clean_2.detach().cpu(),
                "clean_source_path_1": batch["clean_source_path_1"],
                "clean_source_path_2": batch["clean_source_path_2"],
            }

    total_count_val = accelerator.gather(total_count).sum()

    total_l1 = (accelerator.gather(total_l1_sum).sum() / (2.0 * total_count_val)).item()
    total_cosine = (accelerator.gather(total_cosine_sum).sum() / (2.0 * total_count_val)).item()

    model.train()

    return {
        "total_l1": total_l1,
        "total_cosine": total_cosine,
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

    sample_pack = torch.load(
        os.path.join(args.dataset_root, "semantic", "train_zsem.pt"),
        map_location="cpu",
    )
    embedding_dim = sample_pack["z_sem"].shape[1]

    model = MLPSkipNet(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_time_emb_channels=args.num_time_emb_channels,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    diffusion = GaussianDiffusion(
        num_timesteps=args.max_timesteps,
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
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()

        epoch_train_loss_sum = 0.0
        epoch_train_loss_count = 0

        for batch in train_dataloader:
            clean_1 = batch["clean_embedding_1"]
            clean_2 = batch["clean_embedding_2"]
            batch_size = clean_1.shape[0]

            x_0 = torch.cat([clean_1, clean_2], dim=1)
            c = 0.5 * (clean_1 + clean_2)
            
            t = torch.randint(1, diffusion.num_timesteps + 1, (batch_size,), device=device)
            noise = torch.randn_like(x_0)

            with accelerator.accumulate(model):
                x_t = diffusion.q_sample(x_0, t, noise)
                predicted_noise = model(x_t, t, c)

                loss = permutation_invariant_mse_loss(
                    predicted_noise=predicted_noise,
                    true_noise=noise
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

                optimizer.step()
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

        val_loss = evaluate_validation_loss(
            model=model,
            dataloader=val_dataloader,
            diffusion=diffusion,
            accelerator=accelerator,
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if accelerator.is_main_process:
            wandb.log(
                {
                    "train_loss": epoch_train_loss,
                    "val_loss": val_loss,
                },
                step=epoch + 1,
            )

            torch.save(
                unwrapped_model.state_dict(),
                os.path.join(base_dir, "checkpoints", "mlp_ema.pt"),
            )

            if is_best:
                torch.save(
                    unwrapped_model.state_dict(),
                    os.path.join(base_dir, "checkpoints", "mlp_ema_best.pt"),
                )

        ema_model.restore(unwrapped_model.parameters())
        accelerator.wait_for_everyone()


def eval_model(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    val_dataloader = get_data(args, "val")

    sample_pack = torch.load(
        os.path.join(args.dataset_root, "semantic", "train_zsem.pt"),
        map_location="cpu",
    )
    embedding_dim = sample_pack["z_sem"].shape[1]

    model = MLPSkipNet(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_time_emb_channels=args.num_time_emb_channels,
    )

    model, val_dataloader = accelerator.prepare(model, val_dataloader)

    model_path = os.path.join(base_dir, "checkpoints", "mlp_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    diffusion = GaussianDiffusion(
        num_timesteps=args.max_timesteps,
        device=device,
    )

    metrics = evaluate_embedding_metrics(
        model=model,
        dataloader=val_dataloader,
        diffusion=diffusion,
        accelerator=accelerator,
    )

    if accelerator.is_main_process:
        report = (
            f"--- DDIM Sampling Evaluation (Validation Set) ---\n"
            f"Permutation-Invariant Pair L1: {metrics['total_l1']:.8f}\n"
            f"Permutation-Invariant Pair Cosine Similarity: {metrics['total_cosine']:.8f}\n"
        )

        print(f"\n{report}")

        out_name = "final_metrics.txt"
        with open(
            os.path.join(base_dir, "results", out_name),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report)

        if metrics["first_item"] is not None:
            save_path = os.path.join(base_dir, "results", "decode_pair_data.pt")
            torch.save(metrics["first_item"], save_path)
            print(f"Saved evaluation embeddings for visual decoding to: {save_path}")


def launch():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        default="encoded_ffhq256_semantic_split",
        help="Folder containing semantic/train_zsem.pt and semantic/val_zsem.pt",
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="Name of the experiment folder",
    )
    parser.add_argument(
        "--train_samples_per_epoch",
        default=1000000,
        type=int,
        help="Number of random train pairs per epoch",
    )
    parser.add_argument(
        "--val_samples",
        default=100000,
        type=int,
        help="Number of deterministic validation pairs",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--max_timesteps",
        default=1000,
        type=int,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        default=150,
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--val_every",
        default=1,
        type=int,
        help="Run validation every N epochs",
    )
    parser.add_argument(
        "--mixed_precision",
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Accelerate mixed precision mode",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--wandb_project",
        default="Face-DM",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--hidden_dim",
        default=2048,
        type=int,
        help="Hidden width of the latent MLP",
    )
    parser.add_argument(
        "--num_layers",
        default=10,
        type=int,
        help="Number of MLP layers",
    )
    parser.add_argument(
        "--num_time_emb_channels",
        default=64,
        type=int,
        help="Sinusoidal timestep embedding width",
    )

    args = parser.parse_args()

    train(args)
    eval_model(args)


if __name__ == "__main__":
    launch()
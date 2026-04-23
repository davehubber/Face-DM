import argparse
import math
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from torch import optim

from utils_semantic_standard_diffusion import get_data, setup_logging


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
        self.condition_scale = nn.Sequential(nn.SiLU(), nn.Linear(condition_channels, out_channels)) if use_condition else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.use_condition:
            x = x * (1.0 + self.condition_scale(cond))
        x = self.norm(x)
        return self.act(x)


class ConditionalMLPSkipNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 2048,
        num_layers: int = 10,
        num_time_emb_channels: int = 64,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = embedding_dim * 2
        self.condition_dim = embedding_dim
        self.skip_layers = set(range(1, num_layers))

        self.time_embed = TimeEmbedding(
            num_time_emb_channels=num_time_emb_channels,
            out_channels=self.condition_dim,
        )
        self.avg_embed = nn.Sequential(
            nn.Linear(embedding_dim, self.condition_dim),
            nn.SiLU(),
            nn.Linear(self.condition_dim, self.condition_dim),
        )

        layers = []
        for i in range(num_layers):
            if i == 0:
                in_dim = self.input_dim
                out_dim = hidden_dim
                use_condition = True
                use_activation = True
            elif i == num_layers - 1:
                in_dim = hidden_dim + self.input_dim
                out_dim = self.input_dim
                use_condition = False
                use_activation = False
            else:
                in_dim = hidden_dim + self.input_dim
                out_dim = hidden_dim
                use_condition = True
                use_activation = True

            layers.append(
                MLPSkipBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    condition_channels=self.condition_dim,
                    use_condition=use_condition,
                    use_activation=use_activation,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, avg_embedding: torch.Tensor) -> torch.Tensor:
        cond = self.time_embed(t) + self.avg_embed(avg_embedding)
        h = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_layers:
                h = torch.cat([h, x], dim=1)
            h = layer(h, cond)
        return h


class StandardDiffusionEmbeddings:
    def __init__(
        self,
        max_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str = "cuda",
    ):
        self.max_timesteps = int(max_timesteps)
        self.device = device

        betas = torch.linspace(beta_start, beta_end, self.max_timesteps, device=device, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def _extract(self, values: torch.Tensor, t: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        out = values.gather(0, t.long() - 1)
        return out.view(-1, *([1] * (reference.ndim - 1)))

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(1, self.max_timesteps + 1, (batch_size,), device=self.device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x0)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bars, t, x0)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return x_t, noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x_t)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bars, t, x_t)
        return (x_t - sqrt_1mab * pred_noise) / sqrt_ab.clamp_min(1e-8)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        avg_embedding: torch.Tensor,
        sampling_stride: int = 10,
    ) -> torch.Tensor:
        sampling_stride = int(sampling_stride)
        if sampling_stride <= 0:
            raise ValueError(f"sampling_stride must be >= 1, got {sampling_stride}")

        batch_size, embedding_dim = avg_embedding.shape
        x_t = torch.randn(batch_size, embedding_dim * 2, device=self.device)
        source_timesteps = list(range(self.max_timesteps, 0, -sampling_stride))

        was_training = model.training
        model.eval()

        for step_idx, t_value in enumerate(source_timesteps):
            t = torch.full((batch_size,), t_value, device=self.device, dtype=torch.long)
            pred_noise = model(x_t, t, avg_embedding)
            x0_pred = self.predict_x0(x_t, t, pred_noise)

            if step_idx == len(source_timesteps) - 1:
                x_t = x0_pred
                break

            prev_t_value = source_timesteps[step_idx + 1]
            prev_t = torch.full((batch_size,), prev_t_value, device=self.device, dtype=torch.long)
            alpha_bar_prev = self._extract(self.alpha_bars, prev_t, x_t)
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

            # DDIM-style deterministic update (eta = 0), matching the standard diffusion objective.
            x_t = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * pred_noise

        if was_training:
            model.train()

        return x_t


def diffusion_loss(prediction: torch.Tensor, target: torch.Tensor, loss_type: str, reduction: str = "mean") -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(prediction, target, reduction=reduction)
    if loss_type == "mse":
        return F.mse_loss(prediction, target, reduction=reduction)
    raise ValueError(f"Unsupported prediction loss: {loss_type}")


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    dataloader,
    diffusion: StandardDiffusionEmbeddings,
    accelerator: Accelerator,
    prediction_loss_type: str,
):
    model.eval()
    loss_sum = torch.zeros(1, device=accelerator.device)
    loss_count = torch.zeros(1, device=accelerator.device)

    for batch_idx, batch in enumerate(dataloader):
        pair_embedding = batch["pair_embedding"]
        avg_embedding = batch["avg_embedding"]
        t = ((torch.arange(pair_embedding.shape[0], device=accelerator.device) + batch_idx * pair_embedding.shape[0]) % diffusion.max_timesteps) + 1

        x_t, noise = diffusion.q_sample(pair_embedding, t)
        pred_noise = model(x_t, t, avg_embedding)

        loss_sum += diffusion_loss(pred_noise, noise, prediction_loss_type, reduction="sum").detach()
        loss_count += torch.tensor(pair_embedding.numel(), device=accelerator.device, dtype=torch.float32)

    avg_val_loss = (accelerator.gather(loss_sum).sum() / accelerator.gather(loss_count).sum()).item()
    model.train()
    return avg_val_loss


@torch.no_grad()
def evaluate_pair_mse(
    model: nn.Module,
    dataloader,
    diffusion: StandardDiffusionEmbeddings,
    accelerator: Accelerator,
    sampling_stride: int,
):
    model.eval()
    embedding_1_sum = torch.zeros(1, device=accelerator.device)
    embedding_2_sum = torch.zeros(1, device=accelerator.device)
    total_sum = torch.zeros(1, device=accelerator.device)
    embedding_1_count = torch.zeros(1, device=accelerator.device)
    embedding_2_count = torch.zeros(1, device=accelerator.device)

    first_item: Dict | None = None

    for batch in dataloader:
        embedding_1 = batch["embedding_1"]
        embedding_2 = batch["embedding_2"]
        avg_embedding = batch["avg_embedding"]

        predicted_pair = diffusion.sample(
            model=model,
            avg_embedding=avg_embedding,
            sampling_stride=sampling_stride,
        )
        predicted_embedding_1, predicted_embedding_2 = predicted_pair.chunk(2, dim=1)

        embedding_1_loss = F.mse_loss(predicted_embedding_1, embedding_1, reduction="sum")
        embedding_2_loss = F.mse_loss(predicted_embedding_2, embedding_2, reduction="sum")

        embedding_1_sum += embedding_1_loss
        embedding_2_sum += embedding_2_loss
        total_sum += embedding_1_loss + embedding_2_loss
        embedding_1_count += embedding_1.numel()
        embedding_2_count += embedding_2.numel()

        if first_item is None:
            first_item = {
                "avg_embedding": avg_embedding[0].detach().cpu(),
                "predicted_embedding_1": predicted_embedding_1[0].detach().cpu(),
                "predicted_embedding_2": predicted_embedding_2[0].detach().cpu(),
                "target_embedding_1": embedding_1[0].detach().cpu(),
                "target_embedding_2": embedding_2[0].detach().cpu(),
                "embedding_1_source_path": batch["embedding_1_source_path"][0],
                "embedding_2_source_path": batch["embedding_2_source_path"][0],
                "embedding_1_sample_id": batch["embedding_1_sample_id"][0],
                "embedding_2_sample_id": batch["embedding_2_sample_id"][0],
            }

    embedding_1_mse = (accelerator.gather(embedding_1_sum).sum() / accelerator.gather(embedding_1_count).sum()).item()
    embedding_2_mse = (accelerator.gather(embedding_2_sum).sum() / accelerator.gather(embedding_2_count).sum()).item()
    total_mse = (
        accelerator.gather(total_sum).sum()
        / (accelerator.gather(embedding_1_count).sum() + accelerator.gather(embedding_2_count).sum())
    ).item()

    model.train()
    return {
        "embedding_1_mse": embedding_1_mse,
        "embedding_2_mse": embedding_2_mse,
        "total_mse": total_mse,
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

    sample_pack = torch.load(os.path.join(args.dataset_root, "semantic", "train_zsem.pt"), map_location="cpu")
    embedding_dim = sample_pack["z_sem"].shape[1]

    model = ConditionalMLPSkipNet(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_time_emb_channels=args.num_time_emb_channels,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=steps_per_epoch * args.epochs,
    )

    diffusion = StandardDiffusionEmbeddings(
        max_timesteps=args.max_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
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
            pair_embedding = batch["pair_embedding"]
            avg_embedding = batch["avg_embedding"]
            t = diffusion.sample_timesteps(pair_embedding.shape[0])

            with accelerator.accumulate(model):
                x_t, noise = diffusion.q_sample(pair_embedding, t)
                pred_noise = model(x_t, t, avg_embedding)
                loss = diffusion_loss(pred_noise, noise, args.prediction_loss, reduction="mean")

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
        val_loss = evaluate_validation_loss(
            model=model,
            dataloader=val_dataloader,
            diffusion=diffusion,
            accelerator=accelerator,
            prediction_loss_type=args.prediction_loss,
        )
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss

        if accelerator.is_main_process:
            wandb.log({"train_loss": epoch_train_loss, "val_loss": val_loss}, step=epoch + 1)
            torch.save(unwrapped_model.state_dict(), os.path.join(base_dir, "checkpoints", "mlp_ema.pt"))
            if is_best:
                torch.save(unwrapped_model.state_dict(), os.path.join(base_dir, "checkpoints", "mlp_ema_best.pt"))

        ema_model.restore(unwrapped_model.parameters())
        accelerator.wait_for_everyone()


def eval_model(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    val_dataloader = get_data(args, "val")
    sample_pack = torch.load(os.path.join(args.dataset_root, "semantic", "train_zsem.pt"), map_location="cpu")
    embedding_dim = sample_pack["z_sem"].shape[1]

    model = ConditionalMLPSkipNet(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_time_emb_channels=args.num_time_emb_channels,
    )

    model, val_dataloader = accelerator.prepare(model, val_dataloader)
    model_path = os.path.join(base_dir, "checkpoints", "mlp_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = StandardDiffusionEmbeddings(
        max_timesteps=args.max_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    metrics = evaluate_pair_mse(
        model=model,
        dataloader=val_dataloader,
        diffusion=diffusion,
        accelerator=accelerator,
        sampling_stride=args.eval_sampling_stride,
    )

    if accelerator.is_main_process:
        report = (
            "--- Standard Diffusion Pair Evaluation (Validation Set) ---\n"
            f"Embedding 1 MSE: {metrics['embedding_1_mse']:.8f}\n"
            f"Embedding 2 MSE: {metrics['embedding_2_mse']:.8f}\n"
            f"Total MSE: {metrics['total_mse']:.8f}\n"
            f"Sampling timestep stride: {args.eval_sampling_stride}\n"
        )
        print(f"\n{report}")

        with open(os.path.join(base_dir, "results", "final_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(report)

        if metrics["first_item"] is not None:
            save_path = os.path.join(base_dir, "results", "decode_pair_data.pt")
            torch.save(metrics["first_item"], save_path)
            print(f"Saved evaluation embeddings for visual decoding to: {save_path}")


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="encoded_ffhq256_semantic_split", help="Folder containing semantic/train_zsem.pt and semantic/val_zsem.pt")
    parser.add_argument("--run_name", required=True, help="Name of the experiment folder")

    parser.add_argument("--train_samples_per_epoch", default=1000000, type=int, help="Number of random train pairs per epoch")
    parser.add_argument("--val_samples", default=100000, type=int, help="Number of deterministic validation pairs")
    parser.add_argument("--num_workers", default=4, type=int, help="DataLoader worker count")

    parser.add_argument("--max_timesteps", default=1000, type=int, help="Number of diffusion timesteps")
    parser.add_argument("--beta_start", default=1e-4, type=float, help="Initial beta in the linear schedule")
    parser.add_argument("--beta_end", default=2e-2, type=float, help="Final beta in the linear schedule")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--epochs", default=150, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient accumulation steps")
    parser.add_argument("--val_every", default=1, type=int, help="Run validation every N epochs")
    parser.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"], help="Accelerate mixed precision mode")
    parser.add_argument("--num_warmup_steps", default=500, type=int, help="Scheduler warmup steps")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Gradient clipping norm")
    parser.add_argument("--wandb_project", default="Face-DM", help="Weights & Biases project name")

    parser.add_argument("--hidden_dim", default=2048, type=int, help="Hidden width of the latent MLP")
    parser.add_argument("--num_layers", default=10, type=int, help="Number of MLP layers")
    parser.add_argument("--num_time_emb_channels", default=64, type=int, help="Sinusoidal timestep embedding width")
    parser.add_argument("--prediction_loss", default="l1", choices=["l1", "mse"], help="Noise-prediction loss for the standard diffusion objective")

    parser.add_argument("--eval_sampling_stride", default=10, type=int, help="Reverse-sampling timestep stride for iterative eval")

    args = parser.parse_args()

    train(args)
    eval_model(args)


if __name__ == "__main__":
    launch()

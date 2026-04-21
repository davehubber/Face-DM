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
    def __init__(self, max_timesteps: int = 300, alpha_max: float = 0.5, device: str = "cuda"):
        self.max_timesteps = int(max_timesteps)
        self.alpha_max = float(alpha_max)
        self.device = device
        self.alteration_per_t = self.alpha_max / self.max_timesteps

    def mix_embeddings(self, dominant_embedding: torch.Tensor, recessive_embedding: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        weight = (self.alteration_per_t * t.float()).unsqueeze(1)
        return dominant_embedding * (1.0 - weight) + recessive_embedding * weight

    def extract_recessive(self, mixed_embedding: torch.Tensor, dominant_embedding: torch.Tensor, alpha: float) -> torch.Tensor:
        alpha = float(alpha)
        if alpha <= 0:
            raise ValueError("alpha must be > 0 to extract the second embedding")
        return (mixed_embedding - (1.0 - alpha) * dominant_embedding) / alpha

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(1, self.max_timesteps + 1, (batch_size,), device=self.device, dtype=torch.long)

    def sample(self, model: nn.Module, mixed_embedding: torch.Tensor, alpha_init: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = mixed_embedding.shape[0]
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        init_timestep = max(1, min(init_timestep, self.max_timesteps))

        model.eval()
        with torch.no_grad():
            x_t = mixed_embedding.to(self.device)
            for i in reversed(range(1, init_timestep + 1)):
                t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
                predicted_dominant = model(x_t, t)
                extracted_recessive = self.extract_recessive(mixed_embedding, predicted_dominant, alpha_init)
                x_t = x_t - self.mix_embeddings(predicted_dominant, extracted_recessive, t) + self.mix_embeddings(
                    predicted_dominant, extracted_recessive, t - 1
                )

        model.train()
        predicted_recessive = self.extract_recessive(mixed_embedding, x_t, alpha_init)
        return x_t, predicted_recessive


@torch.no_grad()
def evaluate_validation_loss(model: nn.Module, dataloader, diffusion: ColdDiffusionEmbeddings, accelerator: Accelerator):
    model.eval()
    loss_sum = torch.zeros(1, device=accelerator.device)
    loss_count = torch.zeros(1, device=accelerator.device)

    for batch_idx, batch in enumerate(dataloader):
        dominant_embeddings = batch["dominant_embedding"]
        recessive_embeddings = batch["recessive_embedding"]
        t = ((torch.arange(dominant_embeddings.shape[0], device=accelerator.device) + batch_idx * dominant_embeddings.shape[0]) % diffusion.max_timesteps) + 1

        x_t = diffusion.mix_embeddings(dominant_embeddings, recessive_embeddings, t)
        predicted_dominant = model(x_t, t)

        loss_sum += F.mse_loss(predicted_dominant, dominant_embeddings, reduction="sum").detach()
        loss_count += dominant_embeddings.numel()

    avg_val_loss = (accelerator.gather(loss_sum).sum() / accelerator.gather(loss_count).sum()).item()
    model.train()
    return avg_val_loss


@torch.no_grad()
def evaluate_embedding_mse(model: nn.Module, dataloader, diffusion: ColdDiffusionEmbeddings, accelerator: Accelerator, alpha_init: float, one_shot: bool = False):
    model.eval()
    dominant_sum = torch.zeros(1, device=accelerator.device)
    recessive_sum = torch.zeros(1, device=accelerator.device)
    total_sum = torch.zeros(1, device=accelerator.device)
    dominant_count = torch.zeros(1, device=accelerator.device)
    recessive_count = torch.zeros(1, device=accelerator.device)

    init_timestep = math.ceil(alpha_init / diffusion.alteration_per_t)
    init_timestep = max(1, min(init_timestep, diffusion.max_timesteps))

    first_item = None

    for batch in dataloader:
        dominant_embeddings = batch["dominant_embedding"]
        recessive_embeddings = batch["recessive_embedding"]
        mixed_embeddings = dominant_embeddings * (1.0 - alpha_init) + recessive_embeddings * alpha_init

        if one_shot:
            t = torch.full((dominant_embeddings.shape[0],), init_timestep, device=accelerator.device, dtype=torch.long)
            predicted_dominant = model(mixed_embeddings, t)
            predicted_recessive = diffusion.extract_recessive(mixed_embeddings, predicted_dominant, alpha_init)
        else:
            predicted_dominant, predicted_recessive = diffusion.sample(model, mixed_embeddings, alpha_init=alpha_init)

        dominant_sum += F.mse_loss(predicted_dominant, dominant_embeddings, reduction="sum")
        recessive_sum += F.mse_loss(predicted_recessive, recessive_embeddings, reduction="sum")
        total_sum += (
            F.mse_loss(predicted_dominant, dominant_embeddings, reduction="sum")
            + F.mse_loss(predicted_recessive, recessive_embeddings, reduction="sum")
        )
        dominant_count += dominant_embeddings.numel()
        recessive_count += recessive_embeddings.numel()

        if first_item is None:
            first_item = {
                "mixed_embedding": mixed_embeddings[0].detach().cpu(),
                "predicted_dominant": predicted_dominant[0].detach().cpu(),
                "predicted_recessive": predicted_recessive[0].detach().cpu(),
                "dominant_source_path": batch["dominant_source_path"][0],
                "recessive_source_path": batch["recessive_source_path"][0],
                "dominant_sample_id": batch["dominant_sample_id"][0],
                "recessive_sample_id": batch["recessive_sample_id"][0],
            }

    dominant_mse = (accelerator.gather(dominant_sum).sum() / accelerator.gather(dominant_count).sum()).item()
    recessive_mse = (accelerator.gather(recessive_sum).sum() / accelerator.gather(recessive_count).sum()).item()
    total_mse = (
        accelerator.gather(total_sum).sum() / (accelerator.gather(dominant_count).sum() + accelerator.gather(recessive_count).sum())
    ).item()
    model.train()
    return {
        "dominant_mse": dominant_mse,
        "recessive_mse": recessive_mse,
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

    model = MLPSkipNet(
        embedding_dim=embedding_dim,
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
        alpha_max=args.alpha_max,
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
            dominant_embeddings = batch["dominant_embedding"]
            recessive_embeddings = batch["recessive_embedding"]
            t = diffusion.sample_timesteps(dominant_embeddings.shape[0])

            with accelerator.accumulate(model):
                x_t = diffusion.mix_embeddings(dominant_embeddings, recessive_embeddings, t)
                predicted_dominant = model(x_t, t)
                loss = F.mse_loss(predicted_dominant, dominant_embeddings)

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
            wandb.log({"train_mse": epoch_train_loss, "val_mse": val_loss}, step=epoch + 1)
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
    sample_pack = torch.load(os.path.join(args.dataset_root, "semantic", "train_zsem.pt"), map_location="cpu")
    embedding_dim = sample_pack["z_sem"].shape[1]

    model = MLPSkipNet(
        embedding_dim=embedding_dim,
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
        alpha_max=args.alpha_max,
        device=device,
    )

    metrics = evaluate_embedding_mse(
        model=model,
        dataloader=val_dataloader,
        diffusion=diffusion,
        accelerator=accelerator,
        alpha_init=args.alpha_init,
        one_shot=one_shot,
    )

    if accelerator.is_main_process:
        label = "One-Shot" if one_shot else "Iterative"
        report = (
            f"--- {label} Evaluation (Validation Set) ---\n"
            f"Dominant MSE: {metrics['dominant_mse']:.8f}\n"
            f"Recovered Recessive MSE: {metrics['recessive_mse']:.8f}\n"
            f"Total MSE: {metrics['total_mse']:.8f}\n"
        )
        print(f"\n{report}")

        out_name = "one_shot_metrics.txt" if one_shot else "final_metrics.txt"
        with open(os.path.join(base_dir, "results", out_name), "w", encoding="utf-8") as f:
            f.write(report)

        if not one_shot and metrics["first_item"] is not None:
            save_path = os.path.join(base_dir, "results", "decode_pair_data.pt")
            torch.save(metrics["first_item"], save_path)
            print(f"Saved evaluation embeddings for visual decoding to: {save_path}")

@torch.no_grad()
def eval_simulated_error_sampling(args):
    accelerator = Accelerator()
    device = accelerator.device
    base_dir = os.path.join("experiments", args.run_name)

    # 1. Setup Data and Model
    val_dataloader = get_data(args, "val")
    sample_pack = torch.load(os.path.join(args.dataset_root, "semantic", "train_zsem.pt"), map_location="cpu")
    embedding_dim = sample_pack["z_sem"].shape[1]

    model = MLPSkipNet(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_time_emb_channels=args.num_time_emb_channels,
    )

    model, val_dataloader = accelerator.prepare(model, val_dataloader)
    model_path = os.path.join(base_dir, "checkpoints", "mlp_ema.pt")

    if not os.path.exists(model_path):
        print(f"Checkpoint not found at {model_path}. Please train the model first.")
        return

    accelerator.unwrap_model(model).load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    diffusion = ColdDiffusionEmbeddings(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )

    total_mse_sum = torch.zeros(1, device=device)
    total_count = torch.zeros(1, device=device)

    # Determine the initial timestep based on alpha_init
    init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
    init_timestep = max(1, min(init_timestep, diffusion.max_timesteps))

    # Build the coarse reverse-time schedule
    requested_steps = args.simulated_sampling_steps
    if requested_steps is None or requested_steps <= 0:
        step_stride = 1
        source_timesteps = list(range(init_timestep, 0, -1))
    else:
        if requested_steps > init_timestep:
            raise ValueError(
                f"simulated_sampling_steps ({requested_steps}) cannot be larger than init_timestep ({init_timestep})."
            )
        if init_timestep % requested_steps != 0:
            raise ValueError(
                f"init_timestep={init_timestep} is not divisible by simulated_sampling_steps={requested_steps}. "
                f"For exact jumps, choose a divisor of {init_timestep}."
            )
        step_stride = init_timestep // requested_steps
        source_timesteps = list(range(init_timestep, 0, -step_stride))

    effective_steps = len(source_timesteps)

    # Calculate standard deviation for the noise to achieve the target MSE
    std_dev = math.sqrt(args.injected_mse)

    for batch in val_dataloader:
        dominant_embeddings = batch["dominant_embedding"]
        recessive_embeddings = batch["recessive_embedding"]
        batch_size = dominant_embeddings.shape[0]

        # Standard cold diffusion starting point
        mixed_embeddings = dominant_embeddings * (1.0 - args.alpha_init) + recessive_embeddings * args.alpha_init
        x_t = mixed_embeddings.to(device)

        final_pred_dominant = None

        # Reverse process using either full sampling or coarse jumps
        for step_idx, t_value in enumerate(source_timesteps):
            t = torch.full((batch_size,), t_value, device=device, dtype=torch.long)

            if step_idx == 0:
                # Inject Gaussian noise into the clean dominant embedding only at the top step
                noise = torch.randn_like(dominant_embeddings) * std_dev
                final_pred_dominant = dominant_embeddings + noise
            else:
                # Use the trained model for all remaining reverse steps
                final_pred_dominant = model(x_t, t)

            extracted_recessive = diffusion.extract_recessive(
                mixed_embeddings, final_pred_dominant, args.alpha_init
            )

            # Jump directly to the next coarse timestep, or to t=0 at the end
            if step_idx == effective_steps - 1:
                prev_t_value = 0
            else:
                prev_t_value = source_timesteps[step_idx + 1]

            prev_t = torch.full((batch_size,), prev_t_value, device=device, dtype=torch.long)

            x_t = x_t - diffusion.mix_embeddings(final_pred_dominant, extracted_recessive, t) + \
                  diffusion.mix_embeddings(final_pred_dominant, extracted_recessive, prev_t)

        # Measure the last dominant prediction produced by the schedule
        total_mse_sum += F.mse_loss(final_pred_dominant, dominant_embeddings, reduction="sum")
        total_count += dominant_embeddings.numel()

    # Aggregate across distributed processes (if any)
    avg_mse = (accelerator.gather(total_mse_sum).sum() / accelerator.gather(total_count).sum()).item()

    if accelerator.is_main_process:
        report = (
            f"--- Simulated Error Injection Evaluation ---\n"
            f"Injected MSE at T={init_timestep}: {args.injected_mse:.8f}\n"
            f"Sampling steps used: {effective_steps}\n"
            f"Timestep stride: {step_stride}\n"
            f"Final Recovered Dominant MSE: {avg_mse:.8f}\n"
        )
        print(f"\n{report}")

        with open(os.path.join(base_dir, "results", "simulated_error_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(report)

def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="encoded_ffhq256_semantic_split", help="Folder containing semantic/train_zsem.pt and semantic/val_zsem.pt")
    parser.add_argument("--run_name", required=True, help="Name of the experiment folder")

    parser.add_argument("--train_samples_per_epoch", default=1000000, type=int, help="Number of random train pairs per epoch")
    parser.add_argument("--val_samples", default=100000, type=int, help="Number of deterministic validation pairs")
    parser.add_argument("--num_workers", default=4, type=int, help="DataLoader worker count")

    parser.add_argument("--alpha_max", default=0.5, type=float, help="Maximum recessive weight at the last timestep")
    parser.add_argument("--alpha_init", default=0.5, type=float, help="Recessive weight used for evaluation sampling")
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

    parser.add_argument("--injected_mse", default=0.5, type=float, help="Amount of MSE error to inject at the initial timestep")

    parser.add_argument("--simulated_sampling_steps", default=None, type=int, help="Number of reverse sampling steps for eval_simulated_error_sampling.")

    args = parser.parse_args()

    train(args)
    eval_model(args, one_shot=False)
    eval_model(args, one_shot=True)
    #eval_simulated_error_sampling(args)


if __name__ == "__main__":
    launch()

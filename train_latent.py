import argparse
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
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
        self.skip_layers = set(range(1, num_layers))
        self.time_embed = TimeEmbedding(
            num_time_emb_channels=num_time_emb_channels,
            out_channels=embedding_dim,
        )

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
    def __init__(
        self,
        max_timesteps: int = 300,
        alpha_max: float = 0.5,
        device: str = "cuda",
    ):
        self.max_timesteps = int(max_timesteps)
        self.alpha_max = float(alpha_max)
        self.device = device
        self.alteration_per_t = self.alpha_max / self.max_timesteps

    def mix_embeddings(
        self,
        clean_embedding_1: torch.Tensor,
        clean_embedding_2: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        weight = (self.alteration_per_t * t.float()).unsqueeze(1)
        return clean_embedding_1 * (1.0 - weight) + clean_embedding_2 * weight

    def extract_other(
        self,
        mixed_embedding: torch.Tensor,
        predicted_embedding: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        alpha = float(alpha)
        if alpha <= 0:
            raise ValueError("alpha must be > 0 to extract the second embedding")
        return (mixed_embedding - (1.0 - alpha) * predicted_embedding) / alpha

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(
            1,
            self.max_timesteps + 1,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def sample(
        self,
        model: nn.Module,
        mixed_embedding: torch.Tensor,
        alpha_init: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = mixed_embedding.shape[0]
        init_timestep = math.ceil(alpha_init / self.alteration_per_t)
        init_timestep = max(1, min(init_timestep, self.max_timesteps))

        model.eval()
        with torch.no_grad():
            x_t = mixed_embedding.to(self.device)
            for i in reversed(range(1, init_timestep + 1)):
                t = torch.full(
                    (batch_size,),
                    i,
                    device=self.device,
                    dtype=torch.long,
                )

                predicted_embedding = model(x_t, t)
                extracted_embedding = self.extract_other(
                    mixed_embedding,
                    predicted_embedding,
                    alpha_init,
                )

                x_t = (
                    x_t
                    - self.mix_embeddings(predicted_embedding, extracted_embedding, t)
                    + self.mix_embeddings(predicted_embedding, extracted_embedding, t - 1)
                )

        model.train()
        extracted_embedding = self.extract_other(
            mixed_embedding,
            x_t,
            alpha_init,
        )
        return x_t, extracted_embedding


def _l1_sum_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.abs(x - y).reshape(x.shape[0], -1).sum(dim=1)


def permutation_invariant_single_prediction_l1(
    predicted_embedding: torch.Tensor,
    clean_embedding_1: torch.Tensor,
    clean_embedding_2: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    loss_to_1 = _l1_sum_per_sample(predicted_embedding, clean_embedding_1)
    loss_to_2 = _l1_sum_per_sample(predicted_embedding, clean_embedding_2)
    min_loss = torch.minimum(loss_to_1, loss_to_2)

    if reduction == "sum":
        return min_loss.sum()

    if reduction == "mean":
        elements_per_sample = predicted_embedding[0].numel()
        return min_loss.mean() / elements_per_sample

    raise ValueError(f"Unsupported reduction: {reduction}")


def permutation_invariant_pair_l1_sums(
    predicted_embedding: torch.Tensor,
    extracted_embedding: torch.Tensor,
    clean_embedding_1: torch.Tensor,
    clean_embedding_2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_to_1 = _l1_sum_per_sample(predicted_embedding, clean_embedding_1)
    extracted_to_2 = _l1_sum_per_sample(extracted_embedding, clean_embedding_2)

    predicted_to_2 = _l1_sum_per_sample(predicted_embedding, clean_embedding_2)
    extracted_to_1 = _l1_sum_per_sample(extracted_embedding, clean_embedding_1)

    config_1_total = predicted_to_1 + extracted_to_2
    config_2_total = predicted_to_2 + extracted_to_1

    use_config_1 = config_1_total <= config_2_total

    predicted_loss = torch.where(use_config_1, predicted_to_1, predicted_to_2)
    extracted_loss = torch.where(use_config_1, extracted_to_2, extracted_to_1)
    total_loss = torch.minimum(config_1_total, config_2_total)

    return predicted_loss.sum(), extracted_loss.sum(), total_loss.sum()


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    dataloader,
    diffusion: ColdDiffusionEmbeddings,
    accelerator: Accelerator,
    fixed_timestep: Optional[int] = None,
):
    model.eval()

    loss_sum = torch.zeros(1, device=accelerator.device)
    loss_count = torch.zeros(1, device=accelerator.device)

    for batch_idx, batch in enumerate(dataloader):
        clean_embeddings_1 = batch["clean_embedding_1"]
        clean_embeddings_2 = batch["clean_embedding_2"]

        if fixed_timestep is None:
            t = (
                (
                    torch.arange(
                        clean_embeddings_1.shape[0],
                        device=accelerator.device,
                    )
                    + batch_idx * clean_embeddings_1.shape[0]
                )
                % diffusion.max_timesteps
            ) + 1
        else:
            t = torch.full(
                (clean_embeddings_1.shape[0],),
                fixed_timestep,
                device=accelerator.device,
                dtype=torch.long,
            )

        x_t = diffusion.mix_embeddings(clean_embeddings_1, clean_embeddings_2, t)
        predicted_embedding = model(x_t, t)

        loss_sum += permutation_invariant_single_prediction_l1(
            predicted_embedding=predicted_embedding,
            clean_embedding_1=clean_embeddings_1,
            clean_embedding_2=clean_embeddings_2,
            reduction="sum",
        ).detach()

        loss_count += clean_embeddings_1.numel()

    avg_val_loss = (
        accelerator.gather(loss_sum).sum()
        / accelerator.gather(loss_count).sum()
    ).item()

    model.train()
    return avg_val_loss


@torch.no_grad()
def evaluate_embedding_l1(
    model: nn.Module,
    dataloader,
    diffusion: ColdDiffusionEmbeddings,
    accelerator: Accelerator,
    alpha_init: float,
    one_shot: bool = False,
):
    model.eval()

    predicted_sum = torch.zeros(1, device=accelerator.device)
    extracted_sum = torch.zeros(1, device=accelerator.device)
    total_sum = torch.zeros(1, device=accelerator.device)

    predicted_count = torch.zeros(1, device=accelerator.device)
    extracted_count = torch.zeros(1, device=accelerator.device)

    init_timestep = math.ceil(alpha_init / diffusion.alteration_per_t)
    init_timestep = max(1, min(init_timestep, diffusion.max_timesteps))

    first_item = None

    for batch in dataloader:
        clean_embeddings_1 = batch["clean_embedding_1"]
        clean_embeddings_2 = batch["clean_embedding_2"]

        mixed_embeddings = (
            clean_embeddings_1 * (1.0 - alpha_init)
            + clean_embeddings_2 * alpha_init
        )

        if one_shot:
            t = torch.full(
                (clean_embeddings_1.shape[0],),
                init_timestep,
                device=accelerator.device,
                dtype=torch.long,
            )

            predicted_embedding = model(mixed_embeddings, t)
            extracted_embedding = diffusion.extract_other(
                mixed_embeddings,
                predicted_embedding,
                alpha_init,
            )
        else:
            predicted_embedding, extracted_embedding = diffusion.sample(
                model,
                mixed_embeddings,
                alpha_init=alpha_init,
            )

        predicted_loss_sum, extracted_loss_sum, total_loss_sum = (
            permutation_invariant_pair_l1_sums(
                predicted_embedding=predicted_embedding,
                extracted_embedding=extracted_embedding,
                clean_embedding_1=clean_embeddings_1,
                clean_embedding_2=clean_embeddings_2,
            )
        )

        predicted_sum += predicted_loss_sum
        extracted_sum += extracted_loss_sum
        total_sum += total_loss_sum

        predicted_count += clean_embeddings_1.numel()
        extracted_count += clean_embeddings_2.numel()

        if first_item is None:
            first_item = {
                "mixed_embedding": mixed_embeddings[0].detach().cpu(),
                "predicted_embedding": predicted_embedding[0].detach().cpu(),
                "extracted_embedding": extracted_embedding[0].detach().cpu(),
                "clean_embedding_1": clean_embeddings_1[0].detach().cpu(),
                "clean_embedding_2": clean_embeddings_2[0].detach().cpu(),
                "clean_source_path_1": batch["clean_source_path_1"][0],
                "clean_source_path_2": batch["clean_source_path_2"][0],
                "clean_sample_id_1": batch["clean_sample_id_1"][0],
                "clean_sample_id_2": batch["clean_sample_id_2"][0],
            }

    predicted_l1 = (
        accelerator.gather(predicted_sum).sum()
        / accelerator.gather(predicted_count).sum()
    ).item()

    extracted_l1 = (
        accelerator.gather(extracted_sum).sum()
        / accelerator.gather(extracted_count).sum()
    ).item()

    total_l1 = (
        accelerator.gather(total_sum).sum()
        / (
            accelerator.gather(predicted_count).sum()
            + accelerator.gather(extracted_count).sum()
        )
    ).item()

    model.train()

    return {
        "predicted_l1": predicted_l1,
        "extracted_l1": extracted_l1,
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

    steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

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

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()

        epoch_train_loss_sum = 0.0
        epoch_train_loss_count = 0

        epoch_train_final_loss_sum = 0.0
        epoch_train_final_loss_count = 0

        for batch in train_dataloader:
            clean_embeddings_1 = batch["clean_embedding_1"]
            clean_embeddings_2 = batch["clean_embedding_2"]

            t = diffusion.sample_timesteps(clean_embeddings_1.shape[0])

            with accelerator.accumulate(model):
                x_t = diffusion.mix_embeddings(
                    clean_embeddings_1,
                    clean_embeddings_2,
                    t,
                )

                predicted_embedding = model(x_t, t)

                loss = permutation_invariant_single_prediction_l1(
                    predicted_embedding=predicted_embedding,
                    clean_embedding_1=clean_embeddings_1,
                    clean_embedding_2=clean_embeddings_2,
                    reduction="mean",
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                ema_model.step(model.parameters())

                epoch_train_loss_sum += accelerator.gather(loss.detach()).mean().item()
                epoch_train_loss_count += 1

                with torch.no_grad():
                    t_final = torch.full(
                        (clean_embeddings_1.shape[0],),
                        diffusion.max_timesteps,
                        device=accelerator.device,
                        dtype=torch.long,
                    )

                    x_final = diffusion.mix_embeddings(
                        clean_embeddings_1,
                        clean_embeddings_2,
                        t_final,
                    )

                    predicted_final = model(x_final, t_final)

                    final_loss = permutation_invariant_single_prediction_l1(
                        predicted_embedding=predicted_final,
                        clean_embedding_1=clean_embeddings_1,
                        clean_embedding_2=clean_embeddings_2,
                        reduction="mean",
                    )

                epoch_train_final_loss_sum += (
                    accelerator.gather(final_loss.detach()).mean().item()
                )
                epoch_train_final_loss_count += 1

        accelerator.wait_for_everyone()

        if (epoch + 1) % args.val_every != 0:
            continue

        epoch_train_loss = epoch_train_loss_sum / max(epoch_train_loss_count, 1)

        epoch_train_final_loss = epoch_train_final_loss_sum / max(
            epoch_train_final_loss_count,
            1,
        )

        unwrapped_model = accelerator.unwrap_model(model)

        ema_model.store(unwrapped_model.parameters())
        ema_model.copy_to(unwrapped_model.parameters())

        val_loss = evaluate_validation_loss(
            model=model,
            dataloader=val_dataloader,
            diffusion=diffusion,
            accelerator=accelerator,
            fixed_timestep=None,
        )

        val_final_loss = evaluate_validation_loss(
            model=model,
            dataloader=val_dataloader,
            diffusion=diffusion,
            accelerator=accelerator,
            fixed_timestep=diffusion.max_timesteps,
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if accelerator.is_main_process:
            wandb.log(
                {
                    "train_loss": epoch_train_loss,
                    "val_loss": val_loss,
                    "train_loss_final_timestep": epoch_train_final_loss,
                    "val_loss_final_timestep": val_final_loss,
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


def eval_model(args, one_shot: bool = False):
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

    diffusion = ColdDiffusionEmbeddings(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )

    metrics = evaluate_embedding_l1(
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
            f"Predicted Clean L1: {metrics['predicted_l1']:.8f}\n"
            f"Extracted Other L1: {metrics['extracted_l1']:.8f}\n"
            f"Permutation-Invariant Pair L1: {metrics['total_l1']:.8f}\n"
        )

        print(f"\n{report}")

        out_name = "one_shot_metrics.txt" if one_shot else "final_metrics.txt"
        with open(
            os.path.join(base_dir, "results", out_name),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report)

        if not one_shot and metrics["first_item"] is not None:
            save_path = os.path.join(base_dir, "results", "decode_pair_data.pt")
            torch.save(metrics["first_item"], save_path)
            print(f"Saved evaluation embeddings for visual decoding to: {save_path}")

@torch.no_grad()
def check_sampling_swaps(args):
    """
    Diagnostic function.

    Runs the same iterative sampling path used in full normal evaluation, but
    checks at every reverse step whether the model's predicted clean embedding
    changes identity assignment relative to the previous timestep.

    The assignment is computed using the true clean embeddings:
        predicted closer to clean_embedding_1 -> assignment 0
        predicted closer to clean_embedding_2 -> assignment 1

    A swap is counted when this assignment changes after the first sampled
    timestep.

    Writes only a txt report under:
        experiments/<run_name>/results/sampling_swap_check.txt
    """

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

    diffusion = ColdDiffusionEmbeddings(
        max_timesteps=args.max_timesteps,
        alpha_max=args.alpha_max,
        device=device,
    )

    init_timestep = math.ceil(args.alpha_init / diffusion.alteration_per_t)
    init_timestep = max(1, min(init_timestep, diffusion.max_timesteps))

    total_swap_events = torch.zeros(1, device=device, dtype=torch.long)
    samples_with_any_swap = torch.zeros(1, device=device, dtype=torch.long)
    total_samples = torch.zeros(1, device=device, dtype=torch.long)
    total_transitions_checked = torch.zeros(1, device=device, dtype=torch.long)
    total_exact_ties = torch.zeros(1, device=device, dtype=torch.long)

    for batch in val_dataloader:
        clean_embeddings_1 = batch["clean_embedding_1"]
        clean_embeddings_2 = batch["clean_embedding_2"]

        batch_size = clean_embeddings_1.shape[0]

        mixed_embeddings = (
            clean_embeddings_1 * (1.0 - args.alpha_init)
            + clean_embeddings_2 * args.alpha_init
        )

        x_t = mixed_embeddings

        previous_assignment = None
        previous_tie = None

        sample_has_swap = torch.zeros(
            batch_size,
            device=device,
            dtype=torch.bool,
        )

        for i in reversed(range(1, init_timestep + 1)):
            t = torch.full(
                (batch_size,),
                i,
                device=device,
                dtype=torch.long,
            )

            predicted_embedding = model(x_t, t)

            extracted_embedding = diffusion.extract_other(
                mixed_embeddings,
                predicted_embedding,
                args.alpha_init,
            )

            distance_to_clean_1 = torch.abs(
                predicted_embedding - clean_embeddings_1
            ).reshape(batch_size, -1).sum(dim=1)

            distance_to_clean_2 = torch.abs(
                predicted_embedding - clean_embeddings_2
            ).reshape(batch_size, -1).sum(dim=1)

            current_tie = distance_to_clean_1 == distance_to_clean_2
            current_assignment = torch.where(
                distance_to_clean_1 <= distance_to_clean_2,
                torch.zeros_like(distance_to_clean_1, dtype=torch.long),
                torch.ones_like(distance_to_clean_2, dtype=torch.long),
            )

            total_exact_ties += current_tie.sum()

            if previous_assignment is not None:
                certain_transition = (~previous_tie) & (~current_tie)

                swaps = (
                    current_assignment != previous_assignment
                ) & certain_transition

                total_swap_events += swaps.sum()
                samples_with_any_swap += (swaps & (~sample_has_swap)).sum()
                sample_has_swap |= swaps

                total_transitions_checked += certain_transition.sum()

            previous_assignment = current_assignment
            previous_tie = current_tie

            x_t = (
                x_t
                - diffusion.mix_embeddings(
                    predicted_embedding,
                    extracted_embedding,
                    t,
                )
                + diffusion.mix_embeddings(
                    predicted_embedding,
                    extracted_embedding,
                    t - 1,
                )
            )

        total_samples += batch_size

    accelerator.wait_for_everyone()

    gathered_swap_events = accelerator.gather(total_swap_events).sum().item()
    gathered_samples_with_any_swap = (
        accelerator.gather(samples_with_any_swap).sum().item()
    )
    gathered_total_samples = accelerator.gather(total_samples).sum().item()
    gathered_transitions_checked = (
        accelerator.gather(total_transitions_checked).sum().item()
    )
    gathered_exact_ties = accelerator.gather(total_exact_ties).sum().item()

    if accelerator.is_main_process:
        os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

        report = (
            "--- Sampling Swap Check ---\n"
            f"Checkpoint: {model_path}\n"
            f"alpha_init: {args.alpha_init}\n"
            f"init_timestep: {init_timestep}\n"
            f"Total validation samples checked: {gathered_total_samples}\n"
            f"Total timestep transitions checked: {gathered_transitions_checked}\n"
            f"Total swap events detected: {gathered_swap_events}\n"
            f"Samples with at least one swap: {gathered_samples_with_any_swap}\n"
            f"Exact distance ties ignored: {gathered_exact_ties}\n"
        )

        out_path = os.path.join(
            base_dir,
            "results",
            "sampling_swap_check.txt",
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nSaved sampling swap check to: {out_path}")
        print(report)

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
        "--alpha_max",
        default=0.5,
        type=float,
        help="Maximum second-embedding weight at the last timestep",
    )

    parser.add_argument(
        "--alpha_init",
        default=0.5,
        type=float,
        help="Second-embedding weight used for evaluation sampling",
    )

    parser.add_argument(
        "--max_timesteps",
        default=300,
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
        "--num_warmup_steps",
        default=500,
        type=int,
        help="Scheduler warmup steps",
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
    eval_model(args, one_shot=False)
    eval_model(args, one_shot=True)
    check_sampling_swaps(args)


if __name__ == "__main__":
    launch()
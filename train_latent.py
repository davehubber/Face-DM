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


def _l1_mean_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.abs(x - y).reshape(x.shape[0], -1).mean(dim=1)


def permutation_invariant_single_prediction_loss(
    predicted_embedding: torch.Tensor,
    clean_embedding_1: torch.Tensor,
    clean_embedding_2: torch.Tensor,
    cosine_weight: float = 0.0,
    magnitude_weight: float = 0.0,  # <-- NEW PARAMETER
    reduction: str = "mean",
) -> torch.Tensor:
    loss_to_1 = _l1_mean_per_sample(predicted_embedding, clean_embedding_1)
    loss_to_2 = _l1_mean_per_sample(predicted_embedding, clean_embedding_2)

    if cosine_weight > 0:
        cos_to_1 = 1.0 - F.cosine_similarity(predicted_embedding, clean_embedding_1, dim=-1)
        cos_to_2 = 1.0 - F.cosine_similarity(predicted_embedding, clean_embedding_2, dim=-1)

        loss_to_1 = loss_to_1 + cosine_weight * cos_to_1
        loss_to_2 = loss_to_2 + cosine_weight * cos_to_2

    if magnitude_weight > 0:
        pred_norm = torch.linalg.vector_norm(predicted_embedding, ord=2, dim=-1)
        clean1_norm = torch.linalg.vector_norm(clean_embedding_1, ord=2, dim=-1)
        clean2_norm = torch.linalg.vector_norm(clean_embedding_2, ord=2, dim=-1)
        
        dim = predicted_embedding.shape[-1]
        mag_loss_1 = torch.abs(pred_norm - clean1_norm) / dim
        mag_loss_2 = torch.abs(pred_norm - clean2_norm) / dim
        
        loss_to_1 = loss_to_1 + magnitude_weight * mag_loss_1
        loss_to_2 = loss_to_2 + magnitude_weight * mag_loss_2

    min_loss = torch.minimum(loss_to_1, loss_to_2)

    if reduction == "sum":
        return min_loss.sum()

    if reduction == "mean":
        return min_loss.mean()

    raise ValueError(f"Unsupported reduction: {reduction}")


def permutation_invariant_pair_l1_cosine_sums(
    predicted_embedding: torch.Tensor,
    extracted_embedding: torch.Tensor,
    clean_embedding_1: torch.Tensor,
    clean_embedding_2: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    predicted_to_1_l1 = _l1_mean_per_sample(predicted_embedding, clean_embedding_1)
    extracted_to_2_l1 = _l1_mean_per_sample(extracted_embedding, clean_embedding_2)

    predicted_to_2_l1 = _l1_mean_per_sample(predicted_embedding, clean_embedding_2)
    extracted_to_1_l1 = _l1_mean_per_sample(extracted_embedding, clean_embedding_1)

    config_1_total_l1 = predicted_to_1_l1 + extracted_to_2_l1
    config_2_total_l1 = predicted_to_2_l1 + extracted_to_1_l1

    use_config_1 = config_1_total_l1 <= config_2_total_l1

    predicted_l1 = torch.where(use_config_1, predicted_to_1_l1, predicted_to_2_l1)
    extracted_l1 = torch.where(use_config_1, extracted_to_2_l1, extracted_to_1_l1)
    total_l1 = predicted_l1 + extracted_l1

    predicted_to_1_cosine = F.cosine_similarity(
        predicted_embedding,
        clean_embedding_1,
        dim=-1,
    )
    extracted_to_2_cosine = F.cosine_similarity(
        extracted_embedding,
        clean_embedding_2,
        dim=-1,
    )

    predicted_to_2_cosine = F.cosine_similarity(
        predicted_embedding,
        clean_embedding_2,
        dim=-1,
    )
    extracted_to_1_cosine = F.cosine_similarity(
        extracted_embedding,
        clean_embedding_1,
        dim=-1,
    )

    predicted_cosine = torch.where(
        use_config_1,
        predicted_to_1_cosine,
        predicted_to_2_cosine,
    )
    extracted_cosine = torch.where(
        use_config_1,
        extracted_to_2_cosine,
        extracted_to_1_cosine,
    )
    total_cosine = predicted_cosine + extracted_cosine

    return (
        predicted_l1.sum(),
        extracted_l1.sum(),
        total_l1.sum(),
        predicted_cosine.sum(),
        extracted_cosine.sum(),
        total_cosine.sum(),
    )


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    dataloader,
    diffusion: ColdDiffusionEmbeddings,
    accelerator: Accelerator,
    cosine_weight: float = 0.0,
    magnitude_weight: float = 0.0,
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

        loss_sum += permutation_invariant_single_prediction_loss(
            predicted_embedding=predicted_embedding,
            clean_embedding_1=clean_embeddings_1,
            clean_embedding_2=clean_embeddings_2,
            cosine_weight=cosine_weight,
            magnitude_weight=magnitude_weight,
            reduction="sum",
        ).detach()

        loss_count += clean_embeddings_1.shape[0]

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
    diffusion: ColdDiffusionEmbeddings,
    accelerator: Accelerator,
    alpha_init: float,
    one_shot: bool = False,
):
    model.eval()

    predicted_l1_sum = torch.zeros(1, device=accelerator.device)
    extracted_l1_sum = torch.zeros(1, device=accelerator.device)
    total_l1_sum = torch.zeros(1, device=accelerator.device)

    predicted_cosine_sum = torch.zeros(1, device=accelerator.device)
    extracted_cosine_sum = torch.zeros(1, device=accelerator.device)
    total_cosine_sum = torch.zeros(1, device=accelerator.device)

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

        (
            predicted_l1_batch_sum,
            extracted_l1_batch_sum,
            total_l1_batch_sum,
            predicted_cosine_batch_sum,
            extracted_cosine_batch_sum,
            total_cosine_batch_sum,
        ) = permutation_invariant_pair_l1_cosine_sums(
            predicted_embedding=predicted_embedding,
            extracted_embedding=extracted_embedding,
            clean_embedding_1=clean_embeddings_1,
            clean_embedding_2=clean_embeddings_2,
        )

        predicted_l1_sum += predicted_l1_batch_sum
        extracted_l1_sum += extracted_l1_batch_sum
        total_l1_sum += total_l1_batch_sum

        predicted_cosine_sum += predicted_cosine_batch_sum
        extracted_cosine_sum += extracted_cosine_batch_sum
        total_cosine_sum += total_cosine_batch_sum

        predicted_count += clean_embeddings_1.shape[0]
        extracted_count += clean_embeddings_2.shape[0]

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

    predicted_count = accelerator.gather(predicted_count).sum()
    extracted_count = accelerator.gather(extracted_count).sum()
    total_count = predicted_count + extracted_count

    predicted_l1 = (
        accelerator.gather(predicted_l1_sum).sum() / predicted_count
    ).item()

    extracted_l1 = (
        accelerator.gather(extracted_l1_sum).sum() / extracted_count
    ).item()

    total_l1 = (
        accelerator.gather(total_l1_sum).sum() / total_count
    ).item()

    predicted_cosine = (
        accelerator.gather(predicted_cosine_sum).sum() / predicted_count
    ).item()

    extracted_cosine = (
        accelerator.gather(extracted_cosine_sum).sum() / extracted_count
    ).item()

    total_cosine = (
        accelerator.gather(total_cosine_sum).sum() / total_count
    ).item()

    model.train()

    return {
        "predicted_l1": predicted_l1,
        "extracted_l1": extracted_l1,
        "total_l1": total_l1,
        "predicted_cosine": predicted_cosine,
        "extracted_cosine": extracted_cosine,
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

                loss = permutation_invariant_single_prediction_loss(
                    predicted_embedding=predicted_embedding,
                    clean_embedding_1=clean_embeddings_1,
                    clean_embedding_2=clean_embeddings_2,
                    cosine_weight=args.cosine_loss_weight,
                    magnitude_weight=args.magnitude_loss_weight,
                    reduction="mean",
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

                    final_loss = permutation_invariant_single_prediction_loss(
                        predicted_embedding=predicted_final,
                        clean_embedding_1=clean_embeddings_1,
                        clean_embedding_2=clean_embeddings_2,
                        cosine_weight=args.cosine_loss_weight,
                        magnitude_weight=args.magnitude_loss_weight,
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
            cosine_weight=args.cosine_loss_weight,
            magnitude_weight=args.magnitude_loss_weight,
            fixed_timestep=None,
        )

        val_final_loss = evaluate_validation_loss(
            model=model,
            dataloader=val_dataloader,
            diffusion=diffusion,
            accelerator=accelerator,
            cosine_weight=args.cosine_loss_weight,
            magnitude_weight=args.magnitude_loss_weight,
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

    metrics = evaluate_embedding_metrics(
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
            f"Predicted Clean Cosine Similarity: {metrics['predicted_cosine']:.8f}\n"
            f"Extracted Other L1: {metrics['extracted_l1']:.8f}\n"
            f"Extracted Other Cosine Similarity: {metrics['extracted_cosine']:.8f}\n"
            f"Permutation-Invariant Pair L1: {metrics['total_l1']:.8f}\n"
            f"Permutation-Invariant Pair Cosine Similarity: {metrics['total_cosine']:.8f}\n"
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


@torch.no_grad()
def eval_iterative_with_perfect_swap_correction(args):
    """
    Diagnostic evaluation.

    This runs iterative sampling like the normal full evaluation, but uses the
    true clean embeddings to detect, with certainty, whether the model's current
    prediction has switched identity relative to its first prediction.

    If a swap is detected, the reverse update is corrected by swapping the order
    of the predicted/extracted embeddings passed to diffusion.mix_embeddings().
    This keeps the reverse path consistent with the identity assignment of the
    model's first prediction.

    It writes a txt report to:
        experiments/<run_name>/results/swap_corrected_iterative_metrics.txt

    Important:
        This diagnostic is exact for alpha_init == 0.5, because the mixed
        embedding is a true average and the predicted/extracted branches are
        algebraically symmetric.
    """

    if abs(float(args.alpha_init) - 0.5) > 1e-8:
        raise ValueError(
            "This exact swap-correction diagnostic assumes args.alpha_init == 0.5. "
            "With alpha_init != 0.5, the two branches no longer have symmetric "
            "weights in the mixture, so swapping branch order is not algebraically "
            "equivalent."
        )

    def _l1_sum_per_sample_local(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.abs(x - y).reshape(x.shape[0], -1).sum(dim=1)

    def _permutation_invariant_pair_l1_sums(
        predicted_a: torch.Tensor,
        predicted_b: torch.Tensor,
        clean_1: torch.Tensor,
        clean_2: torch.Tensor,
    ):
        a_to_1 = _l1_sum_per_sample_local(predicted_a, clean_1)
        b_to_2 = _l1_sum_per_sample_local(predicted_b, clean_2)

        a_to_2 = _l1_sum_per_sample_local(predicted_a, clean_2)
        b_to_1 = _l1_sum_per_sample_local(predicted_b, clean_1)

        config_1_total = a_to_1 + b_to_2
        config_2_total = a_to_2 + b_to_1

        use_config_1 = config_1_total <= config_2_total

        a_loss = torch.where(use_config_1, a_to_1, a_to_2)
        b_loss = torch.where(use_config_1, b_to_2, b_to_1)
        total_loss = torch.minimum(config_1_total, config_2_total)

        return a_loss.sum(), b_loss.sum(), total_loss.sum()

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

    normal_predicted_sum = torch.zeros(1, device=device)
    normal_extracted_sum = torch.zeros(1, device=device)
    normal_total_sum = torch.zeros(1, device=device)

    corrected_initial_branch_sum = torch.zeros(1, device=device)
    corrected_other_branch_sum = torch.zeros(1, device=device)
    corrected_total_sum = torch.zeros(1, device=device)

    embedding_count = torch.zeros(1, device=device)

    total_samples = torch.zeros(1, device=device, dtype=torch.long)
    total_steps_after_first = torch.zeros(1, device=device, dtype=torch.long)

    swapped_predictions_after_first = torch.zeros(1, device=device, dtype=torch.long)
    samples_with_any_swap = torch.zeros(1, device=device, dtype=torch.long)
    identity_transition_events = torch.zeros(1, device=device, dtype=torch.long)
    exact_ties = torch.zeros(1, device=device, dtype=torch.long)

    alpha_init = float(args.alpha_init)

    for batch in val_dataloader:
        clean_1 = batch["clean_embedding_1"]
        clean_2 = batch["clean_embedding_2"]

        batch_size = clean_1.shape[0]

        mixed_embeddings = (
            clean_1 * (1.0 - alpha_init)
            + clean_2 * alpha_init
        )

        # ------------------------------------------------------------
        # 1) Normal iterative sampling baseline, exactly like eval.
        # ------------------------------------------------------------
        x_normal = mixed_embeddings

        for i in reversed(range(1, init_timestep + 1)):
            t = torch.full(
                (batch_size,),
                i,
                device=device,
                dtype=torch.long,
            )

            predicted_normal = model(x_normal, t)

            extracted_normal = diffusion.extract_other(
                mixed_embeddings,
                predicted_normal,
                alpha_init,
            )

            x_normal = (
                x_normal
                - diffusion.mix_embeddings(
                    predicted_normal,
                    extracted_normal,
                    t,
                )
                + diffusion.mix_embeddings(
                    predicted_normal,
                    extracted_normal,
                    t - 1,
                )
            )

        final_normal_predicted = x_normal

        final_normal_extracted = diffusion.extract_other(
            mixed_embeddings,
            final_normal_predicted,
            alpha_init,
        )

        normal_predicted_loss_sum, normal_extracted_loss_sum, normal_total_loss_sum = (
            _permutation_invariant_pair_l1_sums(
                predicted_a=final_normal_predicted,
                predicted_b=final_normal_extracted,
                clean_1=clean_1,
                clean_2=clean_2,
            )
        )

        normal_predicted_sum += normal_predicted_loss_sum
        normal_extracted_sum += normal_extracted_loss_sum
        normal_total_sum += normal_total_loss_sum

        # ------------------------------------------------------------
        # 2) Swap-corrected iterative sampling.
        # ------------------------------------------------------------
        x_corrected = mixed_embeddings

        initial_assignment = None
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

            predicted_current = model(x_corrected, t)

            extracted_current = diffusion.extract_other(
                mixed_embeddings,
                predicted_current,
                alpha_init,
            )

            distance_to_clean_1 = _l1_sum_per_sample_local(
                predicted_current,
                clean_1,
            )

            distance_to_clean_2 = _l1_sum_per_sample_local(
                predicted_current,
                clean_2,
            )

            current_tie = distance_to_clean_1 == distance_to_clean_2
            exact_ties += current_tie.sum()

            current_assignment = torch.where(
                distance_to_clean_1 <= distance_to_clean_2,
                torch.zeros_like(distance_to_clean_1, dtype=torch.long),
                torch.ones_like(distance_to_clean_2, dtype=torch.long),
            )

            if previous_assignment is not None:
                current_assignment_for_update = torch.where(
                    current_tie,
                    previous_assignment,
                    current_assignment,
                )
            else:
                current_assignment_for_update = current_assignment

            if initial_assignment is None:
                initial_assignment = current_assignment_for_update.clone()

            else:
                certain_current = ~current_tie

                swapped_now = (
                    current_assignment_for_update != initial_assignment
                ) & certain_current

                swapped_predictions_after_first += swapped_now.sum()
                samples_with_any_swap += (swapped_now & (~sample_has_swap)).sum()
                sample_has_swap |= swapped_now

                if previous_assignment is not None and previous_tie is not None:
                    certain_transition = (~previous_tie) & (~current_tie)

                    transition_now = (
                        current_assignment != previous_assignment
                    ) & certain_transition

                    identity_transition_events += transition_now.sum()

                total_steps_after_first += batch_size

            same_as_initial = current_assignment_for_update == initial_assignment

            first_branch_for_update = torch.where(
                same_as_initial[:, None],
                predicted_current,
                extracted_current,
            )

            second_branch_for_update = torch.where(
                same_as_initial[:, None],
                extracted_current,
                predicted_current,
            )

            x_corrected = (
                x_corrected
                - diffusion.mix_embeddings(
                    first_branch_for_update,
                    second_branch_for_update,
                    t,
                )
                + diffusion.mix_embeddings(
                    first_branch_for_update,
                    second_branch_for_update,
                    t - 1,
                )
            )

            previous_assignment = current_assignment_for_update
            previous_tie = current_tie

        final_initial_branch = x_corrected

        final_other_branch = diffusion.extract_other(
            mixed_embeddings,
            final_initial_branch,
            alpha_init,
        )

        true_initial_branch = torch.where(
            initial_assignment[:, None] == 0,
            clean_1,
            clean_2,
        )

        true_other_branch = torch.where(
            initial_assignment[:, None] == 0,
            clean_2,
            clean_1,
        )

        corrected_initial_branch_sum += _l1_sum_per_sample_local(
            final_initial_branch,
            true_initial_branch,
        ).sum()

        corrected_other_branch_sum += _l1_sum_per_sample_local(
            final_other_branch,
            true_other_branch,
        ).sum()

        corrected_pi_a_sum, corrected_pi_b_sum, corrected_pi_total_sum = (
            _permutation_invariant_pair_l1_sums(
                predicted_a=final_initial_branch,
                predicted_b=final_other_branch,
                clean_1=clean_1,
                clean_2=clean_2,
            )
        )

        corrected_total_sum += corrected_pi_total_sum

        embedding_count += clean_1.numel()
        total_samples += batch_size

    accelerator.wait_for_everyone()

    normal_predicted_sum = accelerator.gather(normal_predicted_sum).sum()
    normal_extracted_sum = accelerator.gather(normal_extracted_sum).sum()
    normal_total_sum = accelerator.gather(normal_total_sum).sum()

    corrected_initial_branch_sum = accelerator.gather(
        corrected_initial_branch_sum
    ).sum()
    corrected_other_branch_sum = accelerator.gather(
        corrected_other_branch_sum
    ).sum()
    corrected_total_sum = accelerator.gather(corrected_total_sum).sum()

    embedding_count = accelerator.gather(embedding_count).sum()

    total_samples = accelerator.gather(total_samples).sum().item()
    total_steps_after_first = accelerator.gather(total_steps_after_first).sum().item()
    swapped_predictions_after_first = accelerator.gather(
        swapped_predictions_after_first
    ).sum().item()
    samples_with_any_swap = accelerator.gather(samples_with_any_swap).sum().item()
    identity_transition_events = accelerator.gather(identity_transition_events).sum().item()
    exact_ties = accelerator.gather(exact_ties).sum().item()

    normal_predicted_l1 = (normal_predicted_sum / embedding_count).item()
    normal_extracted_l1 = (normal_extracted_sum / embedding_count).item()
    normal_total_l1 = (normal_total_sum / (2.0 * embedding_count)).item()

    corrected_initial_branch_l1 = (
        corrected_initial_branch_sum / embedding_count
    ).item()
    corrected_other_branch_l1 = (
        corrected_other_branch_sum / embedding_count
    ).item()
    corrected_total_l1 = (
        corrected_total_sum / (2.0 * embedding_count)
    ).item()

    delta_total_l1 = normal_total_l1 - corrected_total_l1

    if accelerator.is_main_process:
        os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

        report = (
            "--- Swap-Corrected Iterative Evaluation ---\n"
            f"Checkpoint: {model_path}\n"
            f"alpha_init: {args.alpha_init}\n"
            f"init_timestep: {init_timestep}\n"
            f"Total validation samples checked: {total_samples}\n"
            f"Total post-first timestep predictions checked: {total_steps_after_first}\n"
            f"Predictions assigned to the opposite clean identity after first step: "
            f"{swapped_predictions_after_first}\n"
            f"Samples with at least one post-first swap: {samples_with_any_swap}\n"
            f"Consecutive identity transition events: {identity_transition_events}\n"
            f"Exact distance ties encountered: {exact_ties}\n"
            "\n"
            "--- Normal Iterative Evaluation ---\n"
            f"Predicted Clean L1: {normal_predicted_l1:.8f}\n"
            f"Extracted Other L1: {normal_extracted_l1:.8f}\n"
            f"Permutation-Invariant Pair L1: {normal_total_l1:.8f}\n"
            "\n"
            "--- Swap-Corrected Iterative Evaluation ---\n"
            f"Initial-Prediction Branch L1: {corrected_initial_branch_l1:.8f}\n"
            f"Other Branch L1: {corrected_other_branch_l1:.8f}\n"
            f"Permutation-Invariant Pair L1: {corrected_total_l1:.8f}\n"
            "\n"
            "--- Estimated Impact of Swapping ---\n"
            f"Normal Pair L1 - Swap-Corrected Pair L1: {delta_total_l1:.8f}\n"
        )

        out_path = os.path.join(
            base_dir,
            "results",
            "swap_corrected_iterative_metrics.txt",
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nSaved swap-corrected iterative metrics to: {out_path}")
        print(report)


@torch.no_grad()
def eval_one_shot_embedding_magnitudes(args):
    """
    Diagnostic one-shot evaluation.

    Runs the same one-shot setup as eval_model(args, one_shot=True), but instead
    of computing L1 / cosine metrics against the clean embeddings, it measures
    the L2 magnitude of:

        1) the model-predicted embedding
        2) the mathematically extracted embedding

    It writes a txt report to:
        experiments/<run_name>/results/one_shot_embedding_magnitudes.txt
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

    alpha_init = float(args.alpha_init)

    init_timestep = math.ceil(alpha_init / diffusion.alteration_per_t)
    init_timestep = max(1, min(init_timestep, diffusion.max_timesteps))

    predicted_magnitude_sum = torch.zeros(1, device=device)
    extracted_magnitude_sum = torch.zeros(1, device=device)
    total_count = torch.zeros(1, device=device)

    for batch in val_dataloader:
        clean_embeddings_1 = batch["clean_embedding_1"]
        clean_embeddings_2 = batch["clean_embedding_2"]

        batch_size = clean_embeddings_1.shape[0]

        mixed_embeddings = (
            clean_embeddings_1 * (1.0 - alpha_init)
            + clean_embeddings_2 * alpha_init
        )

        t = torch.full(
            (batch_size,),
            init_timestep,
            device=device,
            dtype=torch.long,
        )

        predicted_embedding = model(mixed_embeddings, t)

        extracted_embedding = diffusion.extract_other(
            mixed_embeddings,
            predicted_embedding,
            alpha_init,
        )

        predicted_magnitude = torch.linalg.vector_norm(
            predicted_embedding,
            ord=2,
            dim=-1,
        )

        extracted_magnitude = torch.linalg.vector_norm(
            extracted_embedding,
            ord=2,
            dim=-1,
        )

        predicted_magnitude_sum += predicted_magnitude.sum()
        extracted_magnitude_sum += extracted_magnitude.sum()
        total_count += batch_size

    accelerator.wait_for_everyone()

    predicted_magnitude_sum = accelerator.gather(predicted_magnitude_sum).sum()
    extracted_magnitude_sum = accelerator.gather(extracted_magnitude_sum).sum()
    total_count = accelerator.gather(total_count).sum()

    predicted_magnitude_avg = (predicted_magnitude_sum / total_count).item()
    extracted_magnitude_avg = (extracted_magnitude_sum / total_count).item()

    if accelerator.is_main_process:
        os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

        report = (
            "--- One-Shot Embedding Magnitude Evaluation (Validation Set) ---\n"
            f"Checkpoint: {model_path}\n"
            f"alpha_init: {alpha_init}\n"
            f"init_timestep: {init_timestep}\n"
            f"Total validation samples checked: {int(total_count.item())}\n"
            f"Predicted Embedding Average L2 Magnitude: {predicted_magnitude_avg:.8f}\n"
            f"Extracted Embedding Average L2 Magnitude: {extracted_magnitude_avg:.8f}\n"
        )

        out_path = os.path.join(
            base_dir,
            "results",
            "one_shot_embedding_magnitudes.txt",
        )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nSaved one-shot embedding magnitudes to: {out_path}")
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

    parser.add_argument(
        "--cosine_loss_weight",
        default=0.0,
        type=float,
        help="Weight for the cosine similarity penalty added to the L1 loss",
    )

    parser.add_argument(
        "--magnitude_loss_weight",
        default=0.0,
        type=float,
        help="Weight for penalizing predictions that do not have the expected L2 norm of a clean embedding",
    )

    args = parser.parse_args()

    train(args)
    eval_model(args, one_shot=False)
    eval_model(args, one_shot=True)
    # check_sampling_swaps(args)
    # eval_iterative_with_perfect_swap_correction(args)


if __name__ == "__main__":
    launch()
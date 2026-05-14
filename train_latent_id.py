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

from utils_latent_id import (
    get_celeba_classifier_zscore_stats,
    get_data,
    get_embedding_dim,
    get_num_classes,
    get_training_normalization_stats,
    setup_logging,
)


# -------------------------
# Cold-diffusion model
# -------------------------

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

    def alpha_from_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        return self.alteration_per_t * t.float()

    def mix_embeddings(
        self,
        clean_embedding_1: torch.Tensor,
        clean_embedding_2: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.alpha_from_timesteps(t).unsqueeze(1)
        return clean_embedding_1 * (1.0 - weight) + clean_embedding_2 * weight

    def extract_other(
        self,
        mixed_embedding: torch.Tensor,
        predicted_embedding: torch.Tensor,
        alpha,
    ) -> torch.Tensor:
        if not torch.is_tensor(alpha):
            alpha = torch.full(
                (mixed_embedding.shape[0],),
                float(alpha),
                device=mixed_embedding.device,
                dtype=mixed_embedding.dtype,
            )
        else:
            alpha = alpha.to(device=mixed_embedding.device, dtype=mixed_embedding.dtype)

        if torch.any(alpha <= 0):
            raise ValueError("alpha must be > 0 to extract the second embedding")

        alpha = alpha.view(-1, 1)

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

        extracted_embedding = self.extract_other(mixed_embedding, x_t, alpha_init)
        return x_t, extracted_embedding


# -------------------------
# Frozen CelebA identity classifier
# -------------------------

class IdentityMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.20,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),

            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict

    return {
        k.replace("module.", "", 1): v
        for k, v in state_dict.items()
    }


def load_identity_classifier(
    checkpoint_path: str,
    input_dim: int,
    num_classes: int,
    fallback_hidden_dim: int,
    fallback_dropout: float,
    device: torch.device,
) -> IdentityMLP:
    if checkpoint_path is None:
        raise ValueError(
            "identity_classifier_path is required when identity_loss_weight > 0."
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Identity classifier checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {}

    state_dict = _strip_module_prefix(state_dict)

    ckpt_input_dim = int(config.get("input_dim", input_dim))
    ckpt_num_classes = int(config.get("num_classes", num_classes))
    hidden_dim = int(config.get("hidden_dim", fallback_hidden_dim))
    dropout = float(config.get("dropout", fallback_dropout))

    if ckpt_input_dim != input_dim:
        raise ValueError(
            f"Classifier input_dim mismatch: checkpoint has {ckpt_input_dim}, "
            f"current embeddings have {input_dim}."
        )

    if ckpt_num_classes != num_classes:
        raise ValueError(
            f"Classifier num_classes mismatch: checkpoint has {ckpt_num_classes}, "
            f"current CelebA dataset has {num_classes}. Make sure you are using the "
            "same CelebA identity label mapping used to train the classifier."
        )

    classifier = IdentityMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    classifier.load_state_dict(state_dict)
    classifier.to(device)
    classifier.eval()

    for p in classifier.parameters():
        p.requires_grad_(False)

    return classifier


def move_stats_to_device(stats, device: torch.device):
    mean, std = stats
    mean = mean.to(device=device, dtype=torch.float32).view(1, -1)
    std = std.to(device=device, dtype=torch.float32).view(1, -1)
    return mean, std


def shared_to_classifier_space(
    x_shared: torch.Tensor,
    shared_mean: torch.Tensor,
    shared_std: torch.Tensor,
    classifier_mean: torch.Tensor,
    classifier_std: torch.Tensor,
) -> torch.Tensor:
    """
    The deaveraging model trains in one shared z-scored space for FFHQ + CelebA.
    The identity classifier was trained in CelebA's own z-scored space.

    Therefore:
        shared z-space -> raw DiffAE z_sem -> CelebA classifier z-space
    """
    x_float = x_shared.float()
    x_raw = x_float * shared_std + shared_mean
    x_classifier = (x_raw - classifier_mean) / classifier_std
    return x_classifier


# -------------------------
# Losses and metrics
# -------------------------

def _l1_mean_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.abs(x - y).reshape(x.shape[0], -1).mean(dim=1)


def permutation_invariant_single_prediction_l1_loss(
    predicted_embedding: torch.Tensor,
    clean_embedding_1: torch.Tensor,
    clean_embedding_2: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    loss_to_1 = _l1_mean_per_sample(predicted_embedding, clean_embedding_1)
    loss_to_2 = _l1_mean_per_sample(predicted_embedding, clean_embedding_2)
    loss = torch.minimum(loss_to_1, loss_to_2)

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"Unsupported reduction: {reduction}")


def permutation_invariant_pair_identity_loss_and_count(
    identity_classifier: nn.Module,
    predicted_embedding: torch.Tensor,
    extracted_embedding: torch.Tensor,
    clean_label_1: torch.Tensor,
    clean_label_2: torch.Tensor,
    active_mask: torch.Tensor,
    shared_mean: torch.Tensor,
    shared_std: torch.Tensor,
    classifier_mean: torch.Tensor,
    classifier_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    active_mask = active_mask.to(device=predicted_embedding.device, dtype=torch.bool)
    active_count = active_mask.float().sum()

    if active_count.item() == 0:
        zero = predicted_embedding.sum() * 0.0
        return zero, active_count

    predicted_active = predicted_embedding[active_mask]
    extracted_active = extracted_embedding[active_mask]
    label_1_active = clean_label_1[active_mask]
    label_2_active = clean_label_2[active_mask]

    predicted_for_classifier = shared_to_classifier_space(
        predicted_active,
        shared_mean=shared_mean,
        shared_std=shared_std,
        classifier_mean=classifier_mean,
        classifier_std=classifier_std,
    )
    extracted_for_classifier = shared_to_classifier_space(
        extracted_active,
        shared_mean=shared_mean,
        shared_std=shared_std,
        classifier_mean=classifier_mean,
        classifier_std=classifier_std,
    )

    predicted_logits = identity_classifier(predicted_for_classifier)
    extracted_logits = identity_classifier(extracted_for_classifier)

    predicted_to_1 = F.cross_entropy(
        predicted_logits,
        label_1_active,
        reduction="none",
    )
    extracted_to_2 = F.cross_entropy(
        extracted_logits,
        label_2_active,
        reduction="none",
    )

    predicted_to_2 = F.cross_entropy(
        predicted_logits,
        label_2_active,
        reduction="none",
    )
    extracted_to_1 = F.cross_entropy(
        extracted_logits,
        label_1_active,
        reduction="none",
    )

    config_1 = predicted_to_1 + extracted_to_2
    config_2 = predicted_to_2 + extracted_to_1

    loss = torch.minimum(config_1, config_2).mean()

    return loss, active_count


@torch.no_grad()
def permutation_invariant_pair_identity_sums(
    identity_classifier: nn.Module,
    predicted_embedding: torch.Tensor,
    extracted_embedding: torch.Tensor,
    clean_label_1: torch.Tensor,
    clean_label_2: torch.Tensor,
    active_mask: torch.Tensor,
    shared_mean: torch.Tensor,
    shared_std: torch.Tensor,
    classifier_mean: torch.Tensor,
    classifier_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    active_mask = active_mask.to(device=predicted_embedding.device, dtype=torch.bool)
    active_count = active_mask.float().sum()

    if active_count.item() == 0:
        zero = torch.zeros(1, device=predicted_embedding.device)
        return zero, zero, zero, zero, zero

    predicted_active = predicted_embedding[active_mask]
    extracted_active = extracted_embedding[active_mask]
    label_1_active = clean_label_1[active_mask]
    label_2_active = clean_label_2[active_mask]

    predicted_for_classifier = shared_to_classifier_space(
        predicted_active,
        shared_mean=shared_mean,
        shared_std=shared_std,
        classifier_mean=classifier_mean,
        classifier_std=classifier_std,
    )
    extracted_for_classifier = shared_to_classifier_space(
        extracted_active,
        shared_mean=shared_mean,
        shared_std=shared_std,
        classifier_mean=classifier_mean,
        classifier_std=classifier_std,
    )

    predicted_logits = identity_classifier(predicted_for_classifier)
    extracted_logits = identity_classifier(extracted_for_classifier)

    predicted_to_1 = F.cross_entropy(predicted_logits, label_1_active, reduction="none")
    extracted_to_2 = F.cross_entropy(extracted_logits, label_2_active, reduction="none")

    predicted_to_2 = F.cross_entropy(predicted_logits, label_2_active, reduction="none")
    extracted_to_1 = F.cross_entropy(extracted_logits, label_1_active, reduction="none")

    config_1 = predicted_to_1 + extracted_to_2
    config_2 = predicted_to_2 + extracted_to_1

    use_config_1 = config_1 <= config_2

    identity_pair_loss = torch.where(use_config_1, config_1, config_2)

    predicted_target = torch.where(use_config_1, label_1_active, label_2_active)
    extracted_target = torch.where(use_config_1, label_2_active, label_1_active)

    predicted_class = predicted_logits.argmax(dim=1)
    extracted_class = extracted_logits.argmax(dim=1)

    predicted_correct = predicted_class.eq(predicted_target)
    extracted_correct = extracted_class.eq(extracted_target)
    both_correct = predicted_correct & extracted_correct

    return (
        identity_pair_loss.sum(),
        predicted_correct.float().sum(),
        extracted_correct.float().sum(),
        both_correct.float().sum(),
        active_count,
    )


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


def _gather_weighted_mean(
    accelerator: Accelerator,
    local_sum: torch.Tensor,
    local_count: torch.Tensor,
) -> float:
    total_sum = accelerator.gather(local_sum.reshape(1)).sum()
    total_count = accelerator.gather(local_count.reshape(1)).sum().clamp_min(1.0)
    return (total_sum / total_count).item()


# -------------------------
# Validation and evaluation
# -------------------------

@torch.no_grad()
def evaluate_validation_losses(
    model: nn.Module,
    dataloader,
    diffusion: ColdDiffusionEmbeddings,
    accelerator: Accelerator,
    identity_classifier: Optional[nn.Module],
    identity_loss_weight: float,
    identity_min_alpha: float,
    shared_mean: torch.Tensor,
    shared_std: torch.Tensor,
    classifier_mean: torch.Tensor,
    classifier_std: torch.Tensor,
    fixed_timestep: Optional[int] = None,
):
    model.eval()

    l1_sum = torch.zeros(1, device=accelerator.device)
    l1_count = torch.zeros(1, device=accelerator.device)

    identity_sum = torch.zeros(1, device=accelerator.device)
    identity_count = torch.zeros(1, device=accelerator.device)

    for batch_idx, batch in enumerate(dataloader):
        clean_embeddings_1 = batch["clean_embedding_1"]
        clean_embeddings_2 = batch["clean_embedding_2"]
        clean_labels_1 = batch["clean_label_1"]
        clean_labels_2 = batch["clean_label_2"]
        is_celeba_pair = batch["is_celeba_pair"]

        batch_size = clean_embeddings_1.shape[0]

        if fixed_timestep is None:
            t = (
                torch.arange(batch_size, device=accelerator.device)
                + batch_idx * batch_size
            ) % diffusion.max_timesteps + 1
        else:
            t = torch.full(
                (batch_size,),
                fixed_timestep,
                device=accelerator.device,
                dtype=torch.long,
            )

        alpha_t = diffusion.alpha_from_timesteps(t)

        x_t = diffusion.mix_embeddings(clean_embeddings_1, clean_embeddings_2, t)
        predicted_embedding = model(x_t, t)

        l1_per_sample = permutation_invariant_single_prediction_l1_loss(
            predicted_embedding=predicted_embedding,
            clean_embedding_1=clean_embeddings_1,
            clean_embedding_2=clean_embeddings_2,
            reduction="none",
        )

        l1_sum += l1_per_sample.sum().detach()
        l1_count += batch_size

        if identity_classifier is not None:
            extracted_embedding = diffusion.extract_other(
                mixed_embedding=x_t,
                predicted_embedding=predicted_embedding,
                alpha=alpha_t,
            )

            active_mask = is_celeba_pair.bool() & (alpha_t >= identity_min_alpha)

            identity_loss, active_count = permutation_invariant_pair_identity_loss_and_count(
                identity_classifier=identity_classifier,
                predicted_embedding=predicted_embedding,
                extracted_embedding=extracted_embedding,
                clean_label_1=clean_labels_1,
                clean_label_2=clean_labels_2,
                active_mask=active_mask,
                shared_mean=shared_mean,
                shared_std=shared_std,
                classifier_mean=classifier_mean,
                classifier_std=classifier_std,
            )

            identity_sum += (identity_loss.detach() * active_count.detach())
            identity_count += active_count.detach()

    l1_loss = _gather_weighted_mean(accelerator, l1_sum, l1_count)

    if identity_classifier is not None:
        identity_loss = _gather_weighted_mean(accelerator, identity_sum, identity_count)
    else:
        identity_loss = 0.0

    total_loss = l1_loss + identity_loss_weight * identity_loss

    model.train()

    return {
        "l1_loss": l1_loss,
        "identity_loss": identity_loss,
        "total_loss": total_loss,
    }


@torch.no_grad()
def evaluate_embedding_metrics(
    model: nn.Module,
    dataloader,
    diffusion: ColdDiffusionEmbeddings,
    accelerator: Accelerator,
    alpha_init: float,
    identity_classifier: Optional[nn.Module],
    shared_mean: torch.Tensor,
    shared_std: torch.Tensor,
    classifier_mean: torch.Tensor,
    classifier_std: torch.Tensor,
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

    identity_pair_loss_sum = torch.zeros(1, device=accelerator.device)
    identity_predicted_correct_sum = torch.zeros(1, device=accelerator.device)
    identity_extracted_correct_sum = torch.zeros(1, device=accelerator.device)
    identity_both_correct_sum = torch.zeros(1, device=accelerator.device)
    identity_pair_count = torch.zeros(1, device=accelerator.device)

    init_timestep = math.ceil(alpha_init / diffusion.alteration_per_t)
    init_timestep = max(1, min(init_timestep, diffusion.max_timesteps))

    first_item = None

    for batch in dataloader:
        clean_embeddings_1 = batch["clean_embedding_1"]
        clean_embeddings_2 = batch["clean_embedding_2"]
        clean_labels_1 = batch["clean_label_1"]
        clean_labels_2 = batch["clean_label_2"]
        is_celeba_pair = batch["is_celeba_pair"]

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

        batch_size = clean_embeddings_1.shape[0]

        predicted_count += batch_size
        extracted_count += batch_size

        if identity_classifier is not None:
            active_mask = is_celeba_pair.bool()

            (
                identity_pair_loss_batch_sum,
                identity_predicted_correct_batch_sum,
                identity_extracted_correct_batch_sum,
                identity_both_correct_batch_sum,
                active_count,
            ) = permutation_invariant_pair_identity_sums(
                identity_classifier=identity_classifier,
                predicted_embedding=predicted_embedding,
                extracted_embedding=extracted_embedding,
                clean_label_1=clean_labels_1,
                clean_label_2=clean_labels_2,
                active_mask=active_mask,
                shared_mean=shared_mean,
                shared_std=shared_std,
                classifier_mean=classifier_mean,
                classifier_std=classifier_std,
            )

            identity_pair_loss_sum += identity_pair_loss_batch_sum
            identity_predicted_correct_sum += identity_predicted_correct_batch_sum
            identity_extracted_correct_sum += identity_extracted_correct_batch_sum
            identity_both_correct_sum += identity_both_correct_batch_sum
            identity_pair_count += active_count

        if first_item is None:
            first_item = {
                "pair_source": batch["pair_source"][0],
                "mixed_embedding": mixed_embeddings[0].detach().cpu(),
                "predicted_embedding": predicted_embedding[0].detach().cpu(),
                "extracted_embedding": extracted_embedding[0].detach().cpu(),
                "clean_embedding_1": clean_embeddings_1[0].detach().cpu(),
                "clean_embedding_2": clean_embeddings_2[0].detach().cpu(),
                "clean_label_1": int(clean_labels_1[0].detach().cpu().item()),
                "clean_label_2": int(clean_labels_2[0].detach().cpu().item()),
                "clean_identity_id_1": batch["clean_identity_id_1"][0],
                "clean_identity_id_2": batch["clean_identity_id_2"][0],
                "clean_source_path_1": batch["clean_source_path_1"][0],
                "clean_source_path_2": batch["clean_source_path_2"][0],
                "clean_sample_id_1": batch["clean_sample_id_1"][0],
                "clean_sample_id_2": batch["clean_sample_id_2"][0],
            }

    predicted_count = accelerator.gather(predicted_count).sum().clamp_min(1.0)
    extracted_count = accelerator.gather(extracted_count).sum().clamp_min(1.0)
    total_count = predicted_count + extracted_count

    predicted_l1 = (accelerator.gather(predicted_l1_sum).sum() / predicted_count).item()
    extracted_l1 = (accelerator.gather(extracted_l1_sum).sum() / extracted_count).item()
    total_l1 = (accelerator.gather(total_l1_sum).sum() / total_count).item()

    predicted_cosine = (
        accelerator.gather(predicted_cosine_sum).sum() / predicted_count
    ).item()
    extracted_cosine = (
        accelerator.gather(extracted_cosine_sum).sum() / extracted_count
    ).item()
    total_cosine = (accelerator.gather(total_cosine_sum).sum() / total_count).item()

    result = {
        "predicted_l1": predicted_l1,
        "extracted_l1": extracted_l1,
        "total_l1": total_l1,
        "predicted_cosine": predicted_cosine,
        "extracted_cosine": extracted_cosine,
        "total_cosine": total_cosine,
        "first_item": first_item,
    }

    if identity_classifier is not None:
        identity_pair_count = accelerator.gather(identity_pair_count).sum().clamp_min(1.0)

        result.update({
            "identity_pair_loss": (
                accelerator.gather(identity_pair_loss_sum).sum()
                / identity_pair_count
            ).item(),
            "identity_predicted_top1": (
                accelerator.gather(identity_predicted_correct_sum).sum()
                / identity_pair_count
            ).item(),
            "identity_extracted_top1": (
                accelerator.gather(identity_extracted_correct_sum).sum()
                / identity_pair_count
            ).item(),
            "identity_both_top1": (
                accelerator.gather(identity_both_correct_sum).sum()
                / identity_pair_count
            ).item(),
        })

    model.train()

    return result


# -------------------------
# Build/train/eval
# -------------------------

def build_model(args, embedding_dim: int) -> MLPSkipNet:
    return MLPSkipNet(
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_time_emb_channels=args.num_time_emb_channels,
    )


def train(args):
    base_dir = setup_logging(args.run_name)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device

    embedding_dim = get_embedding_dim(args)
    num_classes = get_num_classes(args)

    shared_mean, shared_std = move_stats_to_device(
        get_training_normalization_stats(args),
        device,
    )
    classifier_mean, classifier_std = move_stats_to_device(
        get_celeba_classifier_zscore_stats(args),
        device,
    )

    train_dataloader = get_data(args, "train")
    val_dataloader = get_data(args, "val")

    model = build_model(args, embedding_dim=embedding_dim)

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

    identity_classifier = None

    if args.identity_loss_weight > 0.0:
        identity_classifier = load_identity_classifier(
            checkpoint_path=args.identity_classifier_path,
            input_dim=embedding_dim,
            num_classes=num_classes,
            fallback_hidden_dim=args.identity_classifier_hidden_dim,
            fallback_dropout=args.identity_classifier_dropout,
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

    best_metric_value = float("inf")

    for epoch in range(args.epochs):
        model.train()

        train_l1_sum = torch.zeros(1, device=device)
        train_l1_count = torch.zeros(1, device=device)
        train_identity_sum = torch.zeros(1, device=device)
        train_identity_count = torch.zeros(1, device=device)

        train_final_l1_sum = torch.zeros(1, device=device)
        train_final_l1_count = torch.zeros(1, device=device)
        train_final_identity_sum = torch.zeros(1, device=device)
        train_final_identity_count = torch.zeros(1, device=device)

        for batch in train_dataloader:
            clean_embeddings_1 = batch["clean_embedding_1"]
            clean_embeddings_2 = batch["clean_embedding_2"]
            clean_labels_1 = batch["clean_label_1"]
            clean_labels_2 = batch["clean_label_2"]
            is_celeba_pair = batch["is_celeba_pair"]

            batch_size = clean_embeddings_1.shape[0]
            batch_count = torch.tensor(float(batch_size), device=device)

            t = diffusion.sample_timesteps(batch_size)
            alpha_t = diffusion.alpha_from_timesteps(t)

            with accelerator.accumulate(model):
                x_t = diffusion.mix_embeddings(clean_embeddings_1, clean_embeddings_2, t)
                predicted_embedding = model(x_t, t)

                l1_loss = permutation_invariant_single_prediction_l1_loss(
                    predicted_embedding=predicted_embedding,
                    clean_embedding_1=clean_embeddings_1,
                    clean_embedding_2=clean_embeddings_2,
                    reduction="mean",
                )

                if identity_classifier is not None:
                    extracted_embedding = diffusion.extract_other(
                        mixed_embedding=x_t,
                        predicted_embedding=predicted_embedding,
                        alpha=alpha_t,
                    )

                    active_mask = is_celeba_pair.bool() & (alpha_t >= args.identity_min_alpha)

                    identity_loss, identity_active_count = (
                        permutation_invariant_pair_identity_loss_and_count(
                            identity_classifier=identity_classifier,
                            predicted_embedding=predicted_embedding,
                            extracted_embedding=extracted_embedding,
                            clean_label_1=clean_labels_1,
                            clean_label_2=clean_labels_2,
                            active_mask=active_mask,
                            shared_mean=shared_mean,
                            shared_std=shared_std,
                            classifier_mean=classifier_mean,
                            classifier_std=classifier_std,
                        )
                    )
                else:
                    identity_loss = predicted_embedding.sum() * 0.0
                    identity_active_count = torch.zeros(1, device=device)

                total_loss = l1_loss + args.identity_loss_weight * identity_loss

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                ema_model.step(model.parameters())

                train_l1_sum += l1_loss.detach() * batch_count
                train_l1_count += batch_count

                train_identity_sum += identity_loss.detach() * identity_active_count.detach()
                train_identity_count += identity_active_count.detach()

                with torch.no_grad():
                    t_final = torch.full(
                        (batch_size,),
                        diffusion.max_timesteps,
                        device=device,
                        dtype=torch.long,
                    )
                    alpha_final = diffusion.alpha_from_timesteps(t_final)

                    x_final = diffusion.mix_embeddings(
                        clean_embeddings_1,
                        clean_embeddings_2,
                        t_final,
                    )
                    predicted_final = model(x_final, t_final)

                    final_l1_loss = permutation_invariant_single_prediction_l1_loss(
                        predicted_embedding=predicted_final,
                        clean_embedding_1=clean_embeddings_1,
                        clean_embedding_2=clean_embeddings_2,
                        reduction="mean",
                    )

                    if identity_classifier is not None:
                        extracted_final = diffusion.extract_other(
                            mixed_embedding=x_final,
                            predicted_embedding=predicted_final,
                            alpha=alpha_final,
                        )

                        final_active_mask = (
                            is_celeba_pair.bool()
                            & (alpha_final >= args.identity_min_alpha)
                        )

                        final_identity_loss, final_identity_active_count = (
                            permutation_invariant_pair_identity_loss_and_count(
                                identity_classifier=identity_classifier,
                                predicted_embedding=predicted_final,
                                extracted_embedding=extracted_final,
                                clean_label_1=clean_labels_1,
                                clean_label_2=clean_labels_2,
                                active_mask=final_active_mask,
                                shared_mean=shared_mean,
                                shared_std=shared_std,
                                classifier_mean=classifier_mean,
                                classifier_std=classifier_std,
                            )
                        )
                    else:
                        final_identity_loss = predicted_final.sum() * 0.0
                        final_identity_active_count = torch.zeros(1, device=device)

                train_final_l1_sum += final_l1_loss.detach() * batch_count
                train_final_l1_count += batch_count

                train_final_identity_sum += (
                    final_identity_loss.detach()
                    * final_identity_active_count.detach()
                )
                train_final_identity_count += final_identity_active_count.detach()

        accelerator.wait_for_everyone()

        if (epoch + 1) % args.val_every != 0:
            continue

        train_l1_loss = _gather_weighted_mean(accelerator, train_l1_sum, train_l1_count)
        train_identity_loss = _gather_weighted_mean(
            accelerator,
            train_identity_sum,
            train_identity_count,
        )
        train_total_loss = train_l1_loss + args.identity_loss_weight * train_identity_loss

        train_l1_loss_final = _gather_weighted_mean(
            accelerator,
            train_final_l1_sum,
            train_final_l1_count,
        )
        train_identity_loss_final = _gather_weighted_mean(
            accelerator,
            train_final_identity_sum,
            train_final_identity_count,
        )
        train_total_loss_final = (
            train_l1_loss_final
            + args.identity_loss_weight * train_identity_loss_final
        )

        unwrapped_model = accelerator.unwrap_model(model)
        ema_model.store(unwrapped_model.parameters())
        ema_model.copy_to(unwrapped_model.parameters())

        val_metrics = evaluate_validation_losses(
            model=model,
            dataloader=val_dataloader,
            diffusion=diffusion,
            accelerator=accelerator,
            identity_classifier=identity_classifier,
            identity_loss_weight=args.identity_loss_weight,
            identity_min_alpha=args.identity_min_alpha,
            shared_mean=shared_mean,
            shared_std=shared_std,
            classifier_mean=classifier_mean,
            classifier_std=classifier_std,
            fixed_timestep=None,
        )

        val_final_metrics = evaluate_validation_losses(
            model=model,
            dataloader=val_dataloader,
            diffusion=diffusion,
            accelerator=accelerator,
            identity_classifier=identity_classifier,
            identity_loss_weight=args.identity_loss_weight,
            identity_min_alpha=args.identity_min_alpha,
            shared_mean=shared_mean,
            shared_std=shared_std,
            classifier_mean=classifier_mean,
            classifier_std=classifier_std,
            fixed_timestep=diffusion.max_timesteps,
        )

        if args.best_metric == "l1":
            current_metric = val_metrics["l1_loss"]
        elif args.best_metric == "total":
            current_metric = val_metrics["total_loss"]
        else:
            raise ValueError(f"Unsupported best_metric: {args.best_metric}")

        is_best = current_metric < best_metric_value

        if is_best:
            best_metric_value = current_metric

        if accelerator.is_main_process:
            wandb.log(
                {
                    "train_l1_loss": train_l1_loss,
                    "train_identity_loss": train_identity_loss,
                    "train_total_loss": train_total_loss,

                    "val_l1_loss": val_metrics["l1_loss"],
                    "val_identity_loss": val_metrics["identity_loss"],
                    "val_total_loss": val_metrics["total_loss"],

                    "train_l1_loss_final_timestep": train_l1_loss_final,
                    "train_identity_loss_final_timestep": train_identity_loss_final,
                    "train_total_loss_final_timestep": train_total_loss_final,

                    "val_l1_loss_final_timestep": val_final_metrics["l1_loss"],
                    "val_identity_loss_final_timestep": val_final_metrics["identity_loss"],
                    "val_total_loss_final_timestep": val_final_metrics["total_loss"],
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

    embedding_dim = get_embedding_dim(args)
    num_classes = get_num_classes(args)

    shared_mean, shared_std = move_stats_to_device(
        get_training_normalization_stats(args),
        device,
    )
    classifier_mean, classifier_std = move_stats_to_device(
        get_celeba_classifier_zscore_stats(args),
        device,
    )

    val_dataloader = get_data(args, "val")

    model = build_model(args, embedding_dim=embedding_dim)
    model, val_dataloader = accelerator.prepare(model, val_dataloader)

    model_path = os.path.join(base_dir, "checkpoints", "mlp_ema.pt")
    accelerator.unwrap_model(model).load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    identity_classifier = None

    if args.identity_classifier_path is not None:
        identity_classifier = load_identity_classifier(
            checkpoint_path=args.identity_classifier_path,
            input_dim=embedding_dim,
            num_classes=num_classes,
            fallback_hidden_dim=args.identity_classifier_hidden_dim,
            fallback_dropout=args.identity_classifier_dropout,
            device=device,
        )

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
        identity_classifier=identity_classifier,
        shared_mean=shared_mean,
        shared_std=shared_std,
        classifier_mean=classifier_mean,
        classifier_std=classifier_std,
        one_shot=one_shot,
    )

    if accelerator.is_main_process:
        label = "One-Shot" if one_shot else "Iterative"

        report = (
            f"--- {label} Evaluation (Mixed Validation Set) ---\n"
            f"Predicted Clean L1: {metrics['predicted_l1']:.8f}\n"
            f"Predicted Clean Cosine Similarity: {metrics['predicted_cosine']:.8f}\n"
            f"Extracted Other L1: {metrics['extracted_l1']:.8f}\n"
            f"Extracted Other Cosine Similarity: {metrics['extracted_cosine']:.8f}\n"
            f"Permutation-Invariant Pair L1: {metrics['total_l1']:.8f}\n"
            f"Permutation-Invariant Pair Cosine Similarity: {metrics['total_cosine']:.8f}\n"
        )

        if identity_classifier is not None:
            report += (
                f"Identity Pair CE Loss, CelebA pairs only: {metrics['identity_pair_loss']:.8f}\n"
                f"Predicted Identity Top-1, CelebA pairs only: {metrics['identity_predicted_top1']:.8f}\n"
                f"Extracted Identity Top-1, CelebA pairs only: {metrics['identity_extracted_top1']:.8f}\n"
                f"Both Identities Top-1, CelebA pairs only: {metrics['identity_both_top1']:.8f}\n"
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


def launch():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ffhq_dataset_root",
        required=True,
        help="Old FFHQ semantic dataset root containing semantic/train_zsem.pt and semantic/val_zsem.pt.",
    )
    parser.add_argument(
        "--celeba_dataset_root",
        required=True,
        help="CelebA DiffAE dataset root containing raw embeddings, z-scored embeddings, metadata, and train stats.",
    )

    parser.add_argument(
        "--celeba_raw_embeddings_file",
        default="celeba_diffae_zsem.npy",
        help="Raw CelebA DiffAE semantic embeddings. Do not use the z-scored file here.",
    )
    parser.add_argument(
        "--celeba_metadata_file",
        default="celeba_diffae_zsem_metadata.csv",
        help="CelebA metadata CSV with embedding_index, label, split, filename, image_path.",
    )
    parser.add_argument(
        "--celeba_classifier_stats_file",
        default="celeba_diffae_zsem_train_stats.npz",
        help="Stats used to create the CelebA z-scored embeddings used by the identity classifier.",
    )

    parser.add_argument(
        "--training_normalization",
        default="combined",
        choices=["combined", "ffhq", "celeba", "none"],
        help=(
            "Shared training coordinate system for both datasets. "
            "'combined' computes train-set mean/std using FFHQ train + CelebA train."
        ),
    )

    parser.add_argument("--run_name", required=True, help="Name of the experiment folder")

    parser.add_argument(
        "--identity_classifier_path",
        default=None,
        help="Path to the trained CelebA identity classifier checkpoint, usually best_model.pt.",
    )
    parser.add_argument(
        "--identity_loss_weight",
        default=0.01,
        type=float,
        help="Weight of the frozen-classifier auxiliary identity loss on CelebA pairs only.",
    )
    parser.add_argument(
        "--identity_min_alpha",
        default=0.05,
        type=float,
        help=(
            "Only apply pair identity loss when alpha_t >= this value. This avoids unstable "
            "extraction at extremely low mixing levels."
        ),
    )
    parser.add_argument(
        "--identity_classifier_hidden_dim",
        default=512,
        type=int,
        help="Fallback hidden size for the identity classifier if checkpoint config is absent.",
    )
    parser.add_argument(
        "--identity_classifier_dropout",
        default=0.20,
        type=float,
        help="Fallback dropout for the identity classifier if checkpoint config is absent.",
    )

    parser.add_argument(
        "--celeba_pair_probability",
        default=0.25,
        type=float,
        help=(
            "Probability that each sampled pair comes from CelebA. "
            "FFHQ pairs use only L1; CelebA pairs use L1 + identity loss."
        ),
    )
    parser.add_argument(
        "--allow_same_identity_celeba_pairs",
        action="store_true",
        help="Allow same-identity CelebA pairs. By default, CelebA pairs use different identities.",
    )
    parser.add_argument(
        "--no_balanced_celeba_identity_sampling",
        action="store_true",
        help="Disable class-balanced identity sampling for CelebA pairs.",
    )

    parser.add_argument(
        "--best_metric",
        default="l1",
        choices=["l1", "total"],
        help=(
            "Checkpoint selection metric. 'l1' keeps the best geometric deaveraging checkpoint; "
            "'total' follows the combined objective."
        ),
    )

    parser.add_argument(
        "--train_samples_per_epoch",
        default=1_000_000,
        type=int,
        help="Number of random train pairs per epoch.",
    )
    parser.add_argument(
        "--val_samples",
        default=100_000,
        type=int,
        help="Number of deterministic validation pairs.",
    )
    parser.add_argument("--num_workers", default=4, type=int, help="DataLoader worker count")

    parser.add_argument(
        "--alpha_max",
        default=0.5,
        type=float,
        help="Maximum second-embedding weight at the last timestep.",
    )
    parser.add_argument(
        "--alpha_init",
        default=0.5,
        type=float,
        help="Second-embedding weight used for evaluation sampling.",
    )
    parser.add_argument("--max_timesteps", default=300, type=int, help="Number of diffusion timesteps")

    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--epochs", default=150, type=int, help="Number of training epochs")
    parser.add_argument("--lr", default=3e-4, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="AdamW weight decay")

    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--val_every", default=1, type=int, help="Run validation every N epochs")

    parser.add_argument(
        "--mixed_precision",
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Accelerate mixed precision mode.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Gradient clipping norm")
    parser.add_argument("--wandb_project", default="Face-DM", help="Weights & Biases project name")

    parser.add_argument("--hidden_dim", default=2048, type=int, help="Hidden width of the latent MLP")
    parser.add_argument("--num_layers", default=10, type=int, help="Number of MLP layers")
    parser.add_argument(
        "--num_time_emb_channels",
        default=64,
        type=int,
        help="Sinusoidal timestep embedding width.",
    )

    args = parser.parse_args()

    if not (0.0 <= args.celeba_pair_probability <= 1.0):
        raise ValueError("--celeba_pair_probability must be between 0 and 1.")

    if args.identity_loss_weight > 0.0 and args.identity_classifier_path is None:
        raise ValueError(
            "--identity_classifier_path must be provided when --identity_loss_weight > 0."
        )

    train(args)
    eval_model(args, one_shot=False)
    eval_model(args, one_shot=True)


if __name__ == "__main__":
    launch()
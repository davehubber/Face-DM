#!/usr/bin/env python3
"""
Train/evaluate a latent cold diffusion model on the new MAD22/SMDD DiffAE
semantic-embedding datasets.

Expected dataset layout
-----------------------
Each resolved scenario folder must contain:

    train_morph_zsem.npy
    train_acc_zsem.npy
    train_cri_zsem.npy
    train_metadata.csv

    test_morph_zsem.npy
    test_acc_zsem.npy
    test_cri_zsem.npy
    test_metadata.csv

    train_zsem_mean.npy
    train_zsem_std.npy

The model uses:
    morph z_sem      -> morph input
    criminal z_sem   -> condition used to build the coarse estimate
    accomplice z_sem -> target to predict/sample

The coarse estimate is:
    coarse_acc = 2 * morph - criminal

All arrays are z-score normalized using the scenario's train_zsem_mean/std.
Those statistics should have been computed from train source embeddings only
(train_acc_zsem + train_cri_zsem), as in the dataset creation script.

Default dataset:
    SMDD_embeddings/SMDD/identity_disjoint_fixed_1000

Example SMDD:
python train_latent_diffae_new_embeddings.py \
    --embeddings-root /nas-ctm01/homes/dacordeiro/Face-DM \
    --dataset smdd \
    --run-name smdd_diffae_latent_acc_from_morph_cri

Example MAD22 method:
python train_latent_diffae_new_embeddings.py \
    --embeddings-root /nas-ctm01/homes/dacordeiro/Face-DM \
    --dataset mad22 \
    --mad22-method MIPGAN_II \
    --mad22-scenario identity_disjoint_80_20 \
    --run-name mad22_mipgan2_id_disjoint_acc_from_morph_cri
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


# -----------------------------------------------------------------------------
# Data resolution and loading
# -----------------------------------------------------------------------------


MAD22_METHODS = ["FaceMorpher", "MIPGAN_I", "MIPGAN_II", "OpenCV", "Webmorph"]
MAD22_SCENARIOS = ["random_80_20", "identity_disjoint_80_20"]
SMDD_SCENARIOS = ["identity_disjoint_fixed_1000"]


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    method: str
    scenario: str
    data_dir: Path


def resolve_dataset_dir(args: argparse.Namespace) -> DatasetPaths:
    """Resolve the requested new-embedding dataset folder."""
    if args.data_dir is not None:
        data_dir = Path(args.data_dir).expanduser().resolve()
        dataset = args.dataset
        method = args.mad22_method if dataset == "mad22" else "SMDD"
        scenario = args.mad22_scenario if dataset == "mad22" else args.smdd_scenario
        return DatasetPaths(dataset=dataset, method=method, scenario=scenario, data_dir=data_dir)

    root = Path(args.embeddings_root).expanduser().resolve()

    if args.dataset == "smdd":
        scenario = args.smdd_scenario
        if scenario not in SMDD_SCENARIOS:
            raise ValueError(f"Unsupported SMDD scenario: {scenario}. Valid: {SMDD_SCENARIOS}")
        data_dir = root / "SMDD_embeddings" / "SMDD" / scenario
        return DatasetPaths(dataset="smdd", method="SMDD", scenario=scenario, data_dir=data_dir)

    if args.dataset == "mad22":
        method = args.mad22_method
        scenario = args.mad22_scenario
        if method not in MAD22_METHODS:
            raise ValueError(f"Unsupported MAD22 method: {method}. Valid: {MAD22_METHODS}")
        if scenario not in MAD22_SCENARIOS:
            raise ValueError(f"Unsupported MAD22 scenario: {scenario}. Valid: {MAD22_SCENARIOS}")
        data_dir = root / "MAD22_embeddings" / method / scenario
        return DatasetPaths(dataset="mad22", method=method, scenario=scenario, data_dir=data_dir)

    raise ValueError(f"Unknown dataset: {args.dataset}")


def require_files(data_dir: Path, filenames: Sequence[str]) -> None:
    missing = [name for name in filenames if not (data_dir / name).exists()]
    if missing:
        message = "\n".join(f"  - {data_dir / name}" for name in missing)
        raise FileNotFoundError(f"Missing required dataset files:\n{message}")


def load_metadata_count(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def load_mean_std(data_dir: Path, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.load(data_dir / "train_zsem_mean.npy").astype(np.float32).reshape(-1)
    std = np.load(data_dir / "train_zsem_std.npy").astype(np.float32).reshape(-1)
    std = np.maximum(std, eps).astype(np.float32)
    return mean, std


class NewDiffAESplitDataset(Dataset):
    """
    Dataset for the new exported embedding format.

    Returns:
        morph_norm, criminal_norm, accomplice_norm

    The criminal embedding is used as the condition for the coarse estimate.
    The accomplice embedding is the target.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        mean: np.ndarray,
        std: np.ndarray,
        mmap: bool = True,
    ):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.data_dir = Path(data_dir)
        self.split = split
        self.mean = mean.astype(np.float32).reshape(-1)
        self.std = std.astype(np.float32).reshape(-1)

        mmap_mode = "r" if mmap else None
        self.morph = np.load(self.data_dir / f"{split}_morph_zsem.npy", mmap_mode=mmap_mode)
        self.acc = np.load(self.data_dir / f"{split}_acc_zsem.npy", mmap_mode=mmap_mode)
        self.cri = np.load(self.data_dir / f"{split}_cri_zsem.npy", mmap_mode=mmap_mode)
        self.metadata_path = self.data_dir / f"{split}_metadata.csv"

        if not (len(self.morph) == len(self.acc) == len(self.cri)):
            raise ValueError(
                f"Mismatched {split} array lengths in {data_dir}: "
                f"morph={len(self.morph)}, acc={len(self.acc)}, cri={len(self.cri)}"
            )

        meta_count = load_metadata_count(self.metadata_path)
        if meta_count >= 0 and meta_count != len(self.morph):
            raise ValueError(
                f"Metadata row count does not match {split} arrays in {data_dir}: "
                f"metadata={meta_count}, arrays={len(self.morph)}"
            )

        sample_dim = int(np.asarray(self.morph[0]).reshape(-1).shape[0]) if len(self.morph) else 0
        if self.mean.shape[0] != sample_dim or self.std.shape[0] != sample_dim:
            raise ValueError(
                f"Normalization shape mismatch for {split} in {data_dir}: "
                f"mean={self.mean.shape}, std={self.std.shape}, embedding_dim={sample_dim}"
            )

    def __len__(self) -> int:
        return len(self.morph)

    def _normalize_one(self, arr, index: int) -> torch.Tensor:
        x = np.asarray(arr[index], dtype=np.float32).reshape(-1)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32, copy=False))

    def __getitem__(self, index: int):
        morph = self._normalize_one(self.morph, index)
        criminal = self._normalize_one(self.cri, index)
        accomplice = self._normalize_one(self.acc, index)
        return morph, criminal, accomplice


@dataclass
class LoadedData:
    train_dataset: NewDiffAESplitDataset
    test_dataset: NewDiffAESplitDataset
    mean: np.ndarray
    std: np.ndarray
    embedding_dim: int


def load_new_diffae_data(data_dir: Path, mmap: bool = True) -> LoadedData:
    required = [
        "train_morph_zsem.npy",
        "train_acc_zsem.npy",
        "train_cri_zsem.npy",
        "train_metadata.csv",
        "test_morph_zsem.npy",
        "test_acc_zsem.npy",
        "test_cri_zsem.npy",
        "test_metadata.csv",
        "train_zsem_mean.npy",
        "train_zsem_std.npy",
    ]
    require_files(data_dir, required)

    mean, std = load_mean_std(data_dir)
    train_dataset = NewDiffAESplitDataset(data_dir, "train", mean, std, mmap=mmap)
    test_dataset = NewDiffAESplitDataset(data_dir, "test", mean, std, mmap=mmap)
    embedding_dim = train_dataset[0][0].numel()

    return LoadedData(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        mean=mean,
        std=std,
        embedding_dim=embedding_dim,
    )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=time.device) * -scale)
        embeddings = time[:, None].float() * frequencies[None, :]
        emb = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class AdaLNBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        return self.activation(self.norm(h) * (1 + scale) + shift)


class ColdDemorphNet(nn.Module):
    def __init__(
        self,
        x_dim: int = 512,
        hidden_dim: int = 2048,
        num_layers: int = 10,
        time_emb_dim: int = 512,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.blocks = nn.ModuleList(
            [AdaLNBlock(x_dim, hidden_dim, time_emb_dim)]
            + [
                AdaLNBlock(hidden_dim + x_dim, hidden_dim, time_emb_dim)
                for _ in range(num_layers - 1)
            ]
        )
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = x_t
        for index, block in enumerate(self.blocks):
            h = block(h if index == 0 else torch.cat((h, x_t), dim=-1), t_emb)
        return self.final_linear(h)


class LatentColdDiffusion(nn.Module):
    """
    Cold diffusion from target accomplice z_sem to a coarse accomplice estimate.

    Inputs:
        morph      = z_sem of the morph image
        criminal   = z_sem of the second source identity/image
        target_acc = z_sem of the first source identity/image

    Coarse estimate:
        coarse_acc = 2 * morph - criminal
    """

    def __init__(self, model: nn.Module, num_timesteps: int = 10):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def coarse_prediction(self, morph: torch.Tensor, criminal: torch.Tensor) -> torch.Tensor:
        return 2.0 * morph - criminal

    def degrade(self, target_acc: torch.Tensor, coarse_acc: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        return torch.sqrt(gamma) * coarse_acc + torch.sqrt(1.0 - gamma) * target_acc

    def compute_loss(
        self,
        morph: torch.Tensor,
        criminal: torch.Tensor,
        target_acc: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = morph.shape[0]
        if t is None:
            t = torch.randint(
                1,
                self.num_timesteps + 1,
                (batch_size,),
                device=morph.device,
            ).long()

        coarse_acc = self.coarse_prediction(morph, criminal)
        x_t = self.degrade(target_acc, coarse_acc, t)
        prediction = self.model(x_t, t)
        return F.l1_loss(prediction, target_acc)

    @torch.no_grad()
    def sample_loop(self, morph: torch.Tensor, criminal: torch.Tensor) -> torch.Tensor:
        batch_size = morph.shape[0]
        coarse_acc = self.coarse_prediction(morph, criminal)
        x_t = coarse_acc.clone()

        for t in range(self.num_timesteps, 0, -1):
            t_batch = torch.full((batch_size,), t, device=morph.device, dtype=torch.long)
            prediction = self.model(x_t, t_batch)
            deg_t = self.degrade(prediction, coarse_acc, t_batch)
            deg_prev = self.degrade(prediction, coarse_acc, t_batch - 1)
            x_t = x_t - deg_t + deg_prev

        return x_t


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def move_batch_to_device(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=True).float() for tensor in batch)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    diffusion: LatentColdDiffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float,
) -> float:
    diffusion.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        morph, criminal, target_acc = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.compute_loss(morph, criminal, target_acc)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate_diffusion_loss(
    diffusion: LatentColdDiffusion,
    loader: DataLoader,
    device: torch.device,
) -> float:
    diffusion.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        morph, criminal, target_acc = move_batch_to_device(batch, device)
        total_loss += diffusion.compute_loss(morph, criminal, target_acc).item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def predict_batch(
    diffusion: LatentColdDiffusion,
    morph: torch.Tensor,
    criminal: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "iterative":
        return diffusion.sample_loop(morph, criminal)

    if mode == "one_shot":
        t_max = torch.full(
            (morph.shape[0],),
            diffusion.num_timesteps,
            device=morph.device,
            dtype=torch.long,
        )
        coarse_acc = diffusion.coarse_prediction(morph, criminal)
        return diffusion.model(coarse_acc, t_max)

    if mode == "coarse":
        return diffusion.coarse_prediction(morph, criminal)

    raise ValueError("mode must be one of: iterative, one_shot, coarse")


@torch.no_grad()
def evaluate_latent(
    diffusion: LatentColdDiffusion,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    desc: str,
) -> Dict[str, float]:
    diffusion.eval()

    totals = {
        "l1_sum": 0.0,
        "mse_sum": 0.0,
        "cos_sum": 0.0,
        "success_vs_morph_target": 0.0,
        "success_vs_coarse_target": 0.0,
        "n": 0,
        "dim": 0,
    }

    for batch in tqdm(loader, desc=desc):
        morph, criminal, target_acc = move_batch_to_device(batch, device)
        prediction = predict_batch(diffusion, morph, criminal, mode=mode)
        coarse = diffusion.coarse_prediction(morph, criminal)

        batch_size = target_acc.shape[0]
        dim = target_acc.shape[1]

        totals["l1_sum"] += F.l1_loss(prediction, target_acc, reduction="sum").item()
        totals["mse_sum"] += F.mse_loss(prediction, target_acc, reduction="sum").item()
        pred_target_cos = F.cosine_similarity(prediction, target_acc, dim=-1)
        morph_target_cos = F.cosine_similarity(morph, target_acc, dim=-1)
        coarse_target_cos = F.cosine_similarity(coarse, target_acc, dim=-1)

        totals["cos_sum"] += pred_target_cos.sum().item()
        totals["success_vs_morph_target"] += (pred_target_cos > morph_target_cos).float().sum().item()
        totals["success_vs_coarse_target"] += (pred_target_cos > coarse_target_cos).float().sum().item()
        totals["n"] += batch_size
        totals["dim"] = dim

    n = int(totals["n"])
    dim = int(totals["dim"])
    denom = max(1, n * dim)

    return {
        "mode": mode,
        "n": n,
        "embedding_dim": dim,
        "l1": totals["l1_sum"] / denom,
        "mse": totals["mse_sum"] / denom,
        "cosine": totals["cos_sum"] / max(1, n),
        "success_vs_morph_target_pct": 100.0 * totals["success_vs_morph_target"] / max(1, n),
        "success_vs_coarse_target_pct": 100.0 * totals["success_vs_coarse_target"] / max(1, n),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"Mode: {metrics['mode']}\n"
        f"Number of evaluated morphs: {int(metrics['n'])}\n"
        f"Embedding dim: {int(metrics['embedding_dim'])}\n"
        f"L1: {metrics['l1']:.6f}\n"
        f"MSE: {metrics['mse']:.6f}\n"
        f"Cosine similarity: {metrics['cosine']:.6f}\n"
        f"%S vs morph-target cosine: {metrics['success_vs_morph_target_pct']:.2f}%\n"
        f"%S vs coarse-target cosine: {metrics['success_vs_coarse_target_pct']:.2f}%\n"
    )


def save_checkpoint(
    path: Path,
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_model_checkpoint(path: Path, net: nn.Module, device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"])
        return ckpt
    # Backwards-compatible fallback for old scripts that saved only state_dict.
    net.load_state_dict(ckpt)
    return {"model_state_dict": ckpt}


def build_run_name(paths: DatasetPaths, args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name

    return (
        f"{paths.dataset}_{paths.method}_{paths.scenario}_"
        f"acc_from_morph_plus_cri_T{args.num_timesteps}"
    )


def train_and_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    paths = resolve_dataset_dir(args)
    data_dir = paths.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Resolved data directory does not exist: {data_dir}")

    run_name = build_run_name(paths, args)
    exp_dir = Path(args.experiments_root) / run_name
    ckpt_dir = exp_dir / "checkpoints"
    metrics_dir = exp_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    print("\n" + "=" * 100)
    print("LATENT COLDDIFF TRAINING SETUP")
    print("=" * 100)
    print(f"Dataset:         {paths.dataset}")
    print(f"Method:          {paths.method}")
    print(f"Scenario:        {paths.scenario}")
    print(f"Data directory:  {data_dir}")
    print(f"Run name:        {run_name}")
    print(f"Experiment dir:  {exp_dir}")
    print(f"Device:          {device}")
    print("Direction:       morph + criminal -> accomplice")
    print("Coarse estimate: 2 * morph_zsem - criminal_zsem")
    print("Evaluation:      latent losses only, no image decoding")
    print("=" * 100)

    data = load_new_diffae_data(data_dir, mmap=not args.no_mmap)

    train_loader = make_loader(
        data.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = make_loader(
        data.test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    print(f"Train samples:   {len(data.train_dataset)}")
    print(f"Test samples:    {len(data.test_dataset)}")
    print(f"Embedding dim:   {data.embedding_dim}")

    net = ColdDemorphNet(
        x_dim=data.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        time_emb_dim=args.time_emb_dim,
    ).to(device)
    diffusion = LatentColdDiffusion(net, num_timesteps=args.num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = {
        **vars(args),
        "resolved_data_dir": str(data_dir),
        "dataset": paths.dataset,
        "method": paths.method,
        "scenario": paths.scenario,
        "run_name": run_name,
        "embedding_dim": data.embedding_dim,
        "direction_policy": "criminal_condition_accomplice_target",
        "coarse_prediction": "2*morph_zsem - criminal_zsem",
        "normalization": "train_zsem_mean/std from train source embeddings",
        "decode_images_for_eval": False,
    }
    with (exp_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    use_wandb = args.use_wandb and wandb is not None
    if args.use_wandb and wandb is None:
        print("[warning] --use-wandb was set, but wandb could not be imported. Continuing without wandb.")

    if use_wandb:
        wandb.init(project=args.wandb_project, name=run_name, config=config)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                diffusion=diffusion,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                grad_clip=args.grad_clip,
            )
            val_loss = validate_diffusion_loss(diffusion, test_loader, device)

            log_values: Dict[str, float] = {
                "train_loss": train_loss,
                "val_diffusion_loss": val_loss,
                "epoch": epoch,
            }

            if epoch % args.sample_eval_interval == 0:
                sample_metrics = evaluate_latent(
                    diffusion,
                    test_loader,
                    device,
                    mode="iterative",
                    desc=f"Sample eval epoch {epoch}",
                )
                log_values.update({f"val_iterative_{k}": v for k, v in sample_metrics.items() if isinstance(v, (int, float))})

            if use_wandb:
                wandb.log(log_values, step=epoch)

            metrics_text = " | ".join(
                f"{name}: {value:.6f}"
                for name, value in log_values.items()
                if isinstance(value, float)
            )
            print(f"Epoch {epoch:03d} | {metrics_text}")

            save_checkpoint(
                ckpt_dir / "latest.pt",
                net=net,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                config=config,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    net=net,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    config=config,
                )
            else:
                patience_counter += 1

            if args.patience > 0 and patience_counter >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.6f}"
                )
                break

    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found for evaluation in {ckpt_dir}")

    print(f"\nLoading checkpoint for final evaluation: {ckpt_path}")
    load_model_checkpoint(ckpt_path, net, device)
    net.eval()

    all_metrics = []
    for mode in args.eval_modes:
        metrics = evaluate_latent(
            diffusion=diffusion,
            loader=test_loader,
            device=device,
            mode=mode,
            desc=f"Final evaluation ({mode})",
        )
        all_metrics.append(metrics)
        text = format_metrics(metrics)
        print("\n" + text)
        with (metrics_dir / f"eval_{mode}.txt").open("w", encoding="utf-8") as f:
            f.write(text)

    with (metrics_dir / "eval_all_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    if use_wandb:
        for metrics in all_metrics:
            mode = metrics["mode"]
            wandb.log({f"final_{mode}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        wandb.finish()

    print(f"\nDone. Outputs saved to: {exp_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate latent cold diffusion on the new MAD22/SMDD DiffAE "
            "semantic embedding datasets. Default points to SMDD."
        )
    )

    # Dataset selection
    parser.add_argument(
        "--embeddings-root",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM",
        help="Root containing MAD22_embeddings/ and SMDD_embeddings/.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional direct path to a scenario folder. Overrides automatic resolution.",
    )
    parser.add_argument(
        "--dataset",
        choices=["smdd", "mad22"],
        default="smdd",
        help="Dataset to use. Default: smdd.",
    )
    parser.add_argument(
        "--mad22-method",
        choices=MAD22_METHODS,
        default="MIPGAN_II",
        help="MAD22 method folder to use when --dataset mad22.",
    )
    parser.add_argument(
        "--mad22-scenario",
        choices=MAD22_SCENARIOS,
        default="identity_disjoint_80_20",
        help="MAD22 split scenario to use when --dataset mad22.",
    )
    parser.add_argument(
        "--smdd-scenario",
        choices=SMDD_SCENARIOS,
        default="identity_disjoint_fixed_1000",
        help="SMDD split scenario. Default: identity_disjoint_fixed_1000.",
    )

    # Experiment
    parser.add_argument("--experiments-root", type=str, default="experiments")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-mmap", action="store_true", help="Load arrays fully instead of using np.memmap.")

    # Model/training
    parser.add_argument("--num-timesteps", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--time-emb-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument("--sample-eval-interval", type=int, default=10)

    # Evaluation
    parser.add_argument(
        "--eval-modes",
        nargs="+",
        default=["coarse", "one_shot", "iterative"],
        choices=["coarse", "one_shot", "iterative"],
        help="Latent-space evaluation modes to run after training.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate an existing checkpoint.")

    # Logging
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="Face-DM-Conditional")

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()

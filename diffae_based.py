#!/usr/bin/env python3
"""
Train/evaluate the dual-branch identity separation network from:

    Long et al., "Face De-Morphing Based on Diffusion Autoencoders", TIFS 2024.

This script applies only the paper's semantic-latent separation network to the
new MAD22/SMDD DiffAE semantic-embedding datasets already exported as .npy files.
It does not run image encoding/decoding and it does not use the cold-diffusion
refinement model from the previous scripts.

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
    morph z_sem      -> morphed semantic latent code Z_morph
    criminal z_sem   -> reference/criminal semantic latent code Z_cri
    accomplice z_sem -> target semantic latent code Z_acc

Paper architecture implemented here
-----------------------------------
1) Cross-attention inverse linear interpolation branch:

    Q = Z_cri   W_Q
    K = Z_morph W_K
    V = Z_morph W_V
    Z_cri_attn = Z_cri + Attn(Q, K, V)
    Z_acc_1    = 2 * Z_morph - Z_cri_attn

2) MLP branch:

    concat(Z_morph, Z_cri) ->
    Linear(1024,2048)+SiLU+GroupNorm ->
    Linear(2048,4096)+SiLU+GroupNorm ->
    Linear(4096,4096)+SiLU+GroupNorm ->
    Linear(4096,2048)+SiLU+GroupNorm ->
    Linear(2048,1024)+SiLU+GroupNorm ->
    Linear(1024,512)+SiLU+GroupNorm -> Z_acc_2

3) Final prediction:

    Z_acc_pred = Z_acc_1 + Z_acc_2

Paper training parameters used as defaults
------------------------------------------
    epochs      = 100
    batch size  = 16
    lr          = 1e-3
    optimizer   = Adam(beta1=0.9, beta2=0.999)
    loss        = MSE(Z_acc_pred, Z_acc)

Important note on attention
---------------------------
The paper states that Z_sem is a non-spatial 512-D vector and gives W_Q/W_K/W_V
in R^{512x512}. With a literal single token, softmax is over a 1x1 matrix. This
script therefore defaults to --num-attn-tokens 1 for paper-faithful behavior,
but lets you set values such as 4, 8, or 16 to test the tokenized practical
adaptation on your embeddings.

Example SMDD:
python train_dual_branch_identity_separation.py \
    --embeddings-root /nas-ctm01/homes/dacordeiro/Face-DM \
    --dataset smdd \
    --run-name smdd_dual_branch_identity_separation

Example MAD22 method:
python train_dual_branch_identity_separation.py \
    --embeddings-root /nas-ctm01/homes/dacordeiro/Face-DM \
    --dataset mad22 \
    --mad22-method MIPGAN_II \
    --mad22-scenario identity_disjoint_80_20 \
    --run-name mad22_mipgan2_dual_branch_identity_separation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

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
    Dataset for the exported embedding format.

    Returns:
        morph_norm, criminal_norm, accomplice_norm
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
# Model: Long et al. dual-branch identity separation network
# -----------------------------------------------------------------------------


class CrossAttentionInverseLinearInterpolationBranch(nn.Module):
    """
    Cross-attention inverse linear interpolation branch.

    With num_tokens=1, this follows the paper's stated 512-D vector projections:
        W_Q, W_K, W_V in R^{512 x 512}.

    With num_tokens>1, the same branch is applied to a tokenized reshaping of
    the latent vector. This is provided as an ablation because a literal 512-D
    non-spatial vector gives a 1x1 attention matrix.
    """

    def __init__(self, dim: int = 512, num_tokens: int = 1):
        super().__init__()
        if dim % num_tokens != 0:
            raise ValueError(f"dim={dim} must be divisible by num_tokens={num_tokens}")
        self.dim = dim
        self.num_tokens = num_tokens
        self.token_dim = dim // num_tokens

        self.w_q = nn.Linear(self.token_dim, self.token_dim)
        self.w_k = nn.Linear(self.token_dim, self.token_dim)
        self.w_v = nn.Linear(self.token_dim, self.token_dim)
        self.scale = 1.0 / math.sqrt(self.token_dim)

    def forward(self, z_morph: torch.Tensor, z_criminal: torch.Tensor) -> torch.Tensor:
        batch_size = z_morph.shape[0]

        z_criminal_seq = z_criminal.view(batch_size, self.num_tokens, self.token_dim)
        z_morph_seq = z_morph.view(batch_size, self.num_tokens, self.token_dim)

        q = self.w_q(z_criminal_seq)
        k = self.w_k(z_morph_seq)
        v = self.w_v(z_morph_seq)

        attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_probs = F.softmax(attn_logits, dim=-1)
        context_seq = torch.bmm(attn_probs, v)
        context = context_seq.reshape(batch_size, self.dim)

        z_criminal_attn = z_criminal + context
        z_acc_1 = 2.0 * z_morph - z_criminal_attn
        return z_acc_1


class LinearSiLUGroupNorm(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_groups: int = 32):
        super().__init__()
        if out_dim % num_groups != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by num_groups={num_groups}")
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLPComplementBranch(nn.Module):
    """The paper's six-layer MLP branch from Table I."""

    def __init__(self, dim: int = 512, group_norm_groups: int = 32):
        super().__init__()
        if dim != 512:
            raise ValueError("This paper-faithful MLP is fixed to 512-D semantic latent codes.")
        self.net = nn.Sequential(
            LinearSiLUGroupNorm(1024, 2048, group_norm_groups),
            LinearSiLUGroupNorm(2048, 4096, group_norm_groups),
            LinearSiLUGroupNorm(4096, 4096, group_norm_groups),
            LinearSiLUGroupNorm(4096, 2048, group_norm_groups),
            LinearSiLUGroupNorm(2048, 1024, group_norm_groups),
            LinearSiLUGroupNorm(1024, 512, group_norm_groups),
        )

    def forward(self, z_morph: torch.Tensor, z_criminal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((z_morph, z_criminal), dim=-1))


class DualBranchIdentitySeparationNetwork(nn.Module):
    """
    Full identity separation network:
        Z_acc = Z_acc_1 + Z_acc_2
    """

    def __init__(self, dim: int = 512, num_attn_tokens: int = 1, group_norm_groups: int = 32):
        super().__init__()
        if dim != 512:
            raise ValueError("The paper's network is defined for 512-D DiffAE semantic latent codes.")
        self.dim = dim
        self.attn_branch = CrossAttentionInverseLinearInterpolationBranch(
            dim=dim,
            num_tokens=num_attn_tokens,
        )
        self.mlp_branch = MLPComplementBranch(dim=dim, group_norm_groups=group_norm_groups)

    def forward_parts(self, z_morph: torch.Tensor, z_criminal: torch.Tensor) -> Dict[str, torch.Tensor]:
        fixed_inverse = 2.0 * z_morph - z_criminal
        z_acc_1 = self.attn_branch(z_morph, z_criminal)
        z_acc_2 = self.mlp_branch(z_morph, z_criminal)
        proposed = z_acc_1 + z_acc_2
        return {
            "fixed_inverse": fixed_inverse,
            "attn_branch": z_acc_1,
            "mlp_branch": z_acc_2,
            "proposed": proposed,
        }

    def forward(self, z_morph: torch.Tensor, z_criminal: torch.Tensor) -> torch.Tensor:
        return self.forward_parts(z_morph, z_criminal)["proposed"]


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------


def move_batch_to_device(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=True).float() for tensor in batch)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: DualBranchIdentitySeparationNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        morph, criminal, target_acc = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(morph, criminal)
        loss = F.mse_loss(prediction, target_acc)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate_mse_loss(
    model: DualBranchIdentitySeparationNetwork,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        morph, criminal, target_acc = move_batch_to_device(batch, device)
        prediction = model(morph, criminal)
        total_loss += F.mse_loss(prediction, target_acc).item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_latent(
    model: DualBranchIdentitySeparationNetwork,
    loader: DataLoader,
    device: torch.device,
    mode: str,
    desc: str,
) -> Dict[str, float]:
    model.eval()

    totals = {
        "l1_sum": 0.0,
        "mse_sum": 0.0,
        "cos_sum": 0.0,
        "success_vs_morph_target": 0.0,
        "success_vs_fixed_inverse_target": 0.0,
        "n": 0,
        "dim": 0,
    }

    for batch in tqdm(loader, desc=desc):
        morph, criminal, target_acc = move_batch_to_device(batch, device)
        parts = model.forward_parts(morph, criminal)
        if mode not in parts:
            raise ValueError(f"Unknown evaluation mode: {mode}. Valid: {sorted(parts.keys())}")
        prediction = parts[mode]
        fixed_inverse = parts["fixed_inverse"]

        batch_size = target_acc.shape[0]
        dim = target_acc.shape[1]

        totals["l1_sum"] += F.l1_loss(prediction, target_acc, reduction="sum").item()
        totals["mse_sum"] += F.mse_loss(prediction, target_acc, reduction="sum").item()
        pred_target_cos = F.cosine_similarity(prediction, target_acc, dim=-1)
        morph_target_cos = F.cosine_similarity(morph, target_acc, dim=-1)
        fixed_target_cos = F.cosine_similarity(fixed_inverse, target_acc, dim=-1)

        totals["cos_sum"] += pred_target_cos.sum().item()
        totals["success_vs_morph_target"] += (pred_target_cos > morph_target_cos).float().sum().item()
        totals["success_vs_fixed_inverse_target"] += (pred_target_cos > fixed_target_cos).float().sum().item()
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
        "success_vs_fixed_inverse_target_pct": 100.0 * totals["success_vs_fixed_inverse_target"] / max(1, n),
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
        f"%S vs fixed-inverse-target cosine: {metrics['success_vs_fixed_inverse_target_pct']:.2f}%\n"
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_model_checkpoint(path: Path, model: nn.Module, device: torch.device) -> Dict:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt
    model.load_state_dict(ckpt)
    return {"model_state_dict": ckpt}


def build_run_name(paths: DatasetPaths, args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return (
        f"{paths.dataset}_{paths.method}_{paths.scenario}_"
        f"dual_branch_identity_sep_tokens{args.num_attn_tokens}"
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
    print("DUAL-BRANCH IDENTITY SEPARATION TRAINING SETUP")
    print("=" * 100)
    print(f"Dataset:             {paths.dataset}")
    print(f"Method:              {paths.method}")
    print(f"Scenario:            {paths.scenario}")
    print(f"Data directory:      {data_dir}")
    print(f"Run name:            {run_name}")
    print(f"Experiment dir:      {exp_dir}")
    print(f"Device:              {device}")
    print("Direction:           morph + criminal -> accomplice")
    print("Network:             cross-attention inverse interpolation branch + MLP branch")
    print("Loss:                MSE(predicted accomplice z_sem, target accomplice z_sem)")
    print("Evaluation:          latent losses only, no image decoding")
    print("=" * 100)

    data = load_new_diffae_data(data_dir, mmap=not args.no_mmap)
    if data.embedding_dim != 512:
        raise ValueError(
            f"Expected 512-D DiffAE semantic embeddings for the paper network, "
            f"but got embedding_dim={data.embedding_dim}."
        )

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

    print(f"Train samples:       {len(data.train_dataset)}")
    print(f"Test samples:        {len(data.test_dataset)}")
    print(f"Embedding dim:       {data.embedding_dim}")
    print(f"Attention tokens:    {args.num_attn_tokens}")
    if args.num_attn_tokens == 1:
        print("Attention note:      paper-faithful single-token 512-D attention")
    else:
        print("Attention note:      tokenized practical ablation, not literal paper equation")

    model = DualBranchIdentitySeparationNetwork(
        dim=data.embedding_dim,
        num_attn_tokens=args.num_attn_tokens,
        group_norm_groups=args.group_norm_groups,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    print(f"Trainable params:    {count_parameters(model) / 1e6:.2f} M")

    config = {
        **vars(args),
        "resolved_data_dir": str(data_dir),
        "dataset": paths.dataset,
        "method": paths.method,
        "scenario": paths.scenario,
        "run_name": run_name,
        "embedding_dim": data.embedding_dim,
        "direction_policy": "criminal_reference_accomplice_target",
        "architecture": "Long2024_dual_branch_identity_separation_network",
        "cross_attention_branch": "Z_acc_1 = 2*Z_morph - (Z_cri + Attn(Z_cri, Z_morph, Z_morph))",
        "mlp_branch": "1024->2048->4096->4096->2048->1024->512, Linear+SiLU+GroupNorm each layer",
        "final_prediction": "Z_acc = Z_acc_1 + Z_acc_2",
        "loss": "MSE",
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
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                grad_clip=args.grad_clip,
            )
            val_loss = validate_mse_loss(model, test_loader, device)

            log_values: Dict[str, float] = {
                "train_mse_loss": train_loss,
                "val_mse_loss": val_loss,
                "epoch": epoch,
            }

            if epoch % args.eval_interval == 0:
                eval_metrics = evaluate_latent(
                    model=model,
                    loader=test_loader,
                    device=device,
                    mode="proposed",
                    desc=f"Eval epoch {epoch}",
                )
                log_values.update({f"val_proposed_{k}": v for k, v in eval_metrics.items() if isinstance(v, (int, float))})

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
                model=model,
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
                    model=model,
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

    ckpt_path = Path(args.checkpoint) if args.checkpoint else ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found for evaluation in {ckpt_dir}")

    print(f"\nLoading checkpoint for final evaluation: {ckpt_path}")
    load_model_checkpoint(ckpt_path, model, device)
    model.eval()

    all_metrics = []
    for mode in args.eval_modes:
        metrics = evaluate_latent(
            model=model,
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
            "Train/evaluate the Long et al. dual-branch identity separation "
            "network on the new MAD22/SMDD DiffAE semantic embedding datasets."
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

    # Paper model/training defaults
    parser.add_argument("--num-attn-tokens", type=int, default=1, help="1 is paper-faithful; 4/8/16 are tokenized ablations.")
    parser.add_argument("--group-norm-groups", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping; paper uses fixed 100 epochs.")
    parser.add_argument("--eval-interval", type=int, default=10)

    # Evaluation
    parser.add_argument(
        "--eval-modes",
        nargs="+",
        default=["fixed_inverse", "attn_branch", "mlp_branch", "proposed"],
        choices=["fixed_inverse", "attn_branch", "mlp_branch", "proposed"],
        help="Latent-space evaluation modes to run after training.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate an existing checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path for --eval-only or final evaluation.")

    # Logging
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="demorph-ltnt-cold-diff-refine")

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()

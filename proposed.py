#!/usr/bin/env python3
"""Train and evaluate the proposed latent cold-diffusion de-morphing model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
except Exception:  # Optional dependency.
    wandb = None


MAD22_METHODS = ["FaceMorpher", "MIPGAN_I", "MIPGAN_II", "OpenCV", "Webmorph"]
MAD22_SCENARIOS = ["random_80_20", "identity_disjoint_80_20"]
SMDD_SCENARIOS = ["identity_disjoint_fixed_1000"]
REQUIRED_FILES = [
    *[f"{split}_{kind}_zsem.npy" for split in ("train", "test") for kind in ("morph", "acc", "cri")],
    "train_metadata.csv",
    "test_metadata.csv",
    "train_zsem_mean.npy",
    "train_zsem_std.npy",
]


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    method: str
    scenario: str
    data_dir: Path


def resolve_dataset_dir(args: argparse.Namespace) -> DatasetPaths:
    method = args.mad22_method if args.dataset == "mad22" else "SMDD"
    scenario = args.mad22_scenario if args.dataset == "mad22" else args.smdd_scenario
    if args.data_dir:
        return DatasetPaths(args.dataset, method, scenario, Path(args.data_dir).expanduser().resolve())

    root = Path(args.embeddings_root).expanduser().resolve()
    if args.dataset == "smdd":
        if scenario not in SMDD_SCENARIOS:
            raise ValueError(f"Unsupported SMDD scenario: {scenario}. Valid: {SMDD_SCENARIOS}")
        return DatasetPaths("smdd", method, scenario, root / "SMDD_embeddings" / method / scenario)
    if args.dataset == "mad22":
        if method not in MAD22_METHODS:
            raise ValueError(f"Unsupported MAD22 method: {method}. Valid: {MAD22_METHODS}")
        if scenario not in MAD22_SCENARIOS:
            raise ValueError(f"Unsupported MAD22 scenario: {scenario}. Valid: {MAD22_SCENARIOS}")
        return DatasetPaths("mad22", method, scenario, root / "MAD22_embeddings" / method / scenario)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def load_metadata_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as file:
        return sum(1 for _ in csv.DictReader(file))


class NewDiffAESplitDataset(Dataset):
    """Normalized morph, criminal, and accomplice semantic codes."""

    def __init__(self, data_dir: Path, split: str, mean: np.ndarray, std: np.ndarray, mmap: bool = True):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        mmap_mode = "r" if mmap else None
        self.arrays = {
            kind: np.load(data_dir / f"{split}_{kind}_zsem.npy", mmap_mode=mmap_mode)
            for kind in ("morph", "cri", "acc")
        }
        self.mean = mean.astype(np.float32).reshape(-1)
        self.std = std.astype(np.float32).reshape(-1)

        lengths = {kind: len(array) for kind, array in self.arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Mismatched {split} array lengths in {data_dir}: {lengths}")
        if load_metadata_count(data_dir / f"{split}_metadata.csv") != len(self):
            raise ValueError(f"Metadata row count does not match {split} arrays in {data_dir}")

        sample_dim = np.asarray(self.arrays["morph"][0]).size if len(self) else 0
        if self.mean.size != sample_dim or self.std.size != sample_dim:
            raise ValueError(
                f"Normalization shape mismatch for {split}: mean={self.mean.shape}, "
                f"std={self.std.shape}, embedding_dim={sample_dim}"
            )

    def __len__(self) -> int:
        return len(self.arrays["morph"])

    def _normalized(self, kind: str, index: int) -> torch.Tensor:
        value = np.asarray(self.arrays[kind][index], dtype=np.float32).reshape(-1)
        return torch.from_numpy(((value - self.mean) / self.std).astype(np.float32, copy=False))

    def __getitem__(self, index: int):
        return tuple(self._normalized(kind, index) for kind in ("morph", "cri", "acc"))


@dataclass
class LoadedData:
    train_dataset: NewDiffAESplitDataset
    test_dataset: NewDiffAESplitDataset
    mean: np.ndarray
    std: np.ndarray
    embedding_dim: int


def load_new_diffae_data(data_dir: Path, mmap: bool = True) -> LoadedData:
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        paths = "\n".join(f"  - {data_dir / name}" for name in missing)
        raise FileNotFoundError(f"Missing required dataset files:\n{paths}")

    mean = np.load(data_dir / "train_zsem_mean.npy").astype(np.float32).reshape(-1)
    std = np.maximum(np.load(data_dir / "train_zsem_std.npy").astype(np.float32).reshape(-1), 1e-8)
    train = NewDiffAESplitDataset(data_dir, "train", mean, std, mmap)
    test = NewDiffAESplitDataset(data_dir, "test", mean, std, mmap)
    return LoadedData(train, test, mean, std, train[0][0].numel())


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=time.device) * exponent)
        angles = time[:, None].float() * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return F.pad(embedding, (0, self.dim - embedding.shape[-1]))


class AdaLNBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond_proj(condition).chunk(2, dim=-1)
        return F.silu(self.norm(self.linear(x)) * (1 + scale) + shift)


class ColdDemorphNet(nn.Module):
    def __init__(self, x_dim: int = 512, hidden_dim: int = 2048, num_layers: int = 10, time_emb_dim: int = 512):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(x_dim * 2, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.blocks = nn.ModuleList([
            AdaLNBlock(x_dim if i == 0 else hidden_dim + x_dim, hidden_dim, time_emb_dim)
            for i in range(num_layers)
        ])
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(
        self, x_t: torch.Tensor, time: torch.Tensor,
        morph: torch.Tensor, criminal: torch.Tensor,
    ) -> torch.Tensor:
        time_condition = self.time_mlp(time)
        pair_condition = self.pair_mlp(torch.cat((morph, criminal), dim=-1))
        condition = self.condition_mlp(torch.cat((time_condition, pair_condition), dim=-1))
        hidden = x_t
        for index, block in enumerate(self.blocks):
            hidden = block(hidden if index == 0 else torch.cat((hidden, x_t), dim=-1), condition)
        return self.final_linear(hidden)


class LatentColdDiffusion(nn.Module):
    """Linear cold diffusion from the target embedding to the observed morph."""

    def __init__(self, model: nn.Module, num_timesteps: int = 10, terminal_loss_weight: float = 1.0):
        super().__init__()
        if num_timesteps < 1:
            raise ValueError("num_timesteps must be at least 1")
        if terminal_loss_weight < 0:
            raise ValueError("terminal_loss_weight must be non-negative")
        self.model = model
        self.num_timesteps = num_timesteps
        self.terminal_loss_weight = terminal_loss_weight

    def degrade(self, target: torch.Tensor, morph: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        gamma = (time.float() / self.num_timesteps).view(-1, 1)
        return (1.0 - gamma) * target + gamma * morph

    def compute_losses(
        self, morph: torch.Tensor, criminal: torch.Tensor, target: torch.Tensor,
        time: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if time is None:
            time = torch.randint(1, self.num_timesteps + 1, (len(morph),), device=morph.device)
        degraded = self.degrade(target, morph, time)
        path_loss = F.l1_loss(self.model(degraded, time, morph, criminal), target)

        terminal_time = torch.full(
            (len(morph),), self.num_timesteps, device=morph.device, dtype=torch.long
        )
        terminal_loss = F.l1_loss(self.model(morph, terminal_time, morph, criminal), target)
        total_loss = path_loss + self.terminal_loss_weight * terminal_loss
        return {"total": total_loss, "path": path_loss, "terminal": terminal_loss}

    @torch.no_grad()
    def sample_loop(self, morph: torch.Tensor, criminal: torch.Tensor) -> torch.Tensor:
        current = morph.clone()
        for step in range(self.num_timesteps, 0, -1):
            time = torch.full((len(morph),), step, device=morph.device, dtype=torch.long)
            prediction = self.model(current, time, morph, criminal)
            # TACoS correction: remove the predicted state at t and re-degrade it to t-1.
            current += self.degrade(prediction, morph, time - 1) - self.degrade(prediction, morph, time)
        return current


def move_batch_to_device(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=True).float() for tensor in batch)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mean_loader_losses(diffusion: LatentColdDiffusion, loader: DataLoader, device: torch.device, train=False,
                       optimizer=None, grad_clip: float = 0.0, desc: Optional[str] = None) -> Dict[str, float]:
    diffusion.train(train)
    totals = {"total": 0.0, "path": 0.0, "terminal": 0.0}
    count = 0
    iterator = tqdm(loader, desc=desc) if desc else loader
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in iterator:
            morph, criminal, target = move_batch_to_device(batch, device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            losses = diffusion.compute_losses(morph, criminal, target)
            if train:
                losses["total"].backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip)
                optimizer.step()
            for name, loss in losses.items():
                totals[name] += loss.item() * len(target)
            count += len(target)
    return {name: value / max(1, count) for name, value in totals.items()}


def train_one_epoch(diffusion, loader, optimizer, device, epoch, grad_clip) -> Dict[str, float]:
    return mean_loader_losses(diffusion, loader, device, True, optimizer, grad_clip, f"Epoch {epoch}")


def validate_diffusion_losses(diffusion, loader, device) -> Dict[str, float]:
    diffusion.eval()
    totals = {"total": 0.0, "path": 0.0, "terminal": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            morph, criminal, target = move_batch_to_device(batch, device)
            time = torch.arange(count, count + len(target), device=device) % diffusion.num_timesteps + 1
            losses = diffusion.compute_losses(morph, criminal, target, time.long())
            for name, loss in losses.items():
                totals[name] += loss.item() * len(target)
            count += len(target)
    return {name: value / max(1, count) for name, value in totals.items()}


@torch.no_grad()
def predict_batch(diffusion: LatentColdDiffusion, morph: torch.Tensor, criminal: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "one_shot":
        time = torch.full((len(morph),), diffusion.num_timesteps, device=morph.device, dtype=torch.long)
        return diffusion.model(morph, time, morph, criminal)
    if mode == "iterative":
        return diffusion.sample_loop(morph, criminal)
    raise ValueError("mode must be one of: iterative, one_shot")


@torch.no_grad()
def evaluate_latent(diffusion, loader, device, mode: str, desc: str) -> Dict[str, float]:
    diffusion.eval()
    totals = dict(l1=0.0, mse=0.0, cosine=0.0, vs_morph=0.0, n=0, dim=0)
    for batch in tqdm(loader, desc=desc):
        morph, criminal, target = move_batch_to_device(batch, device)
        prediction = predict_batch(diffusion, morph, criminal, mode)
        pred_cos = F.cosine_similarity(prediction, target, dim=-1)
        totals["l1"] += F.l1_loss(prediction, target, reduction="sum").item()
        totals["mse"] += F.mse_loss(prediction, target, reduction="sum").item()
        totals["cosine"] += pred_cos.sum().item()
        totals["vs_morph"] += (pred_cos > F.cosine_similarity(morph, target, dim=-1)).sum().item()
        totals["n"], totals["dim"] = totals["n"] + len(target), target.shape[1]

    n, dim = int(totals["n"]), int(totals["dim"])
    return {
        "mode": mode, "n": n, "embedding_dim": dim,
        "l1": totals["l1"] / max(1, n * dim), "mse": totals["mse"] / max(1, n * dim),
        "cosine": totals["cosine"] / max(1, n),
        "success_vs_morph_target_pct": 100 * totals["vs_morph"] / max(1, n),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"Mode: {metrics['mode']}\nNumber of evaluated morphs: {int(metrics['n'])}\n"
        f"Embedding dim: {int(metrics['embedding_dim'])}\nL1: {metrics['l1']:.6f}\n"
        f"MSE: {metrics['mse']:.6f}\nCosine similarity: {metrics['cosine']:.6f}\n"
        f"%S vs morph-target cosine: {metrics['success_vs_morph_target_pct']:.2f}%\n"
    )


def save_checkpoint(path: Path, net, optimizer, epoch: int, best_val_loss: float, config: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "best_val_loss": best_val_loss, "config": config,
    }, path)


def load_model_checkpoint(path: Path, net, device: torch.device) -> Dict:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    net.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else {"model_state_dict": state_dict}


def build_run_name(paths: DatasetPaths, args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return f"{paths.dataset}_{paths.method}_{paths.scenario}_morph_conditioned_cold_T{args.num_timesteps}"


def print_setup(paths: DatasetPaths, run_name: str, exp_dir: Path, device: torch.device,
                terminal_loss_weight: float) -> None:
    values = {
        "Dataset": paths.dataset, "Method": paths.method, "Scenario": paths.scenario,
        "Data directory": paths.data_dir, "Run name": run_name, "Experiment dir": exp_dir,
        "Device": device, "Direction": "morph + criminal -> accomplice",
        "Degradation": "linear interpolation from target to morph",
        "Conditioning": "ordered [morph, criminal] pair through AdaLN",
        "Timestep sampling": "uniform",
        "Terminal loss weight": terminal_loss_weight,
        "Evaluation": "latent losses only, no image decoding",
    }
    print("\nLATENT COLD-DIFFUSION SETUP")
    for label, value in values.items():
        print(f"{label:16} {value}")


def train_and_evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    paths = resolve_dataset_dir(args)
    if not paths.data_dir.exists():
        raise FileNotFoundError(f"Resolved data directory does not exist: {paths.data_dir}")

    run_name = build_run_name(paths, args)
    exp_dir = Path(args.experiments_root) / run_name
    ckpt_dir, metrics_dir = exp_dir / "checkpoints", exp_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print_setup(paths, run_name, exp_dir, device, args.terminal_loss_weight)

    data = load_new_diffae_data(paths.data_dir, mmap=not args.no_mmap)
    train_loader = make_loader(data.train_dataset, args.batch_size, True, args.num_workers)
    test_loader = make_loader(data.test_dataset, args.eval_batch_size, False, args.num_workers)
    print(f"Train samples:    {len(data.train_dataset)}\nTest samples:     {len(data.test_dataset)}")
    print(f"Embedding dim:    {data.embedding_dim}")

    net = ColdDemorphNet(data.embedding_dim, args.hidden_dim, args.num_layers, args.time_emb_dim).to(device)
    diffusion = LatentColdDiffusion(net, args.num_timesteps, args.terminal_loss_weight).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {
        **vars(args), "resolved_data_dir": str(paths.data_dir), "dataset": paths.dataset,
        "method": paths.method, "scenario": paths.scenario, "run_name": run_name,
        "embedding_dim": data.embedding_dim, "direction_policy": "criminal_condition_accomplice_target",
        "architecture": "morph_conditioned_AdaLN_cold_diffusion",
        "degradation": "(1-gamma)*target + gamma*morph; gamma=t/T",
        "conditioning": "ordered concatenation [morph, criminal] fused with timestep embedding",
        "training_loss": "path_L1 + terminal_loss_weight*terminal_L1",
        "normalization": "train_zsem_mean/std from train source embeddings", "decode_images_for_eval": False,
    }
    with (exp_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    use_wandb = args.use_wandb and wandb is not None
    if args.use_wandb and wandb is None:
        print("[warning] wandb is unavailable; continuing without it.")
    if use_wandb:
        wandb.init(project=args.wandb_project, name=run_name, config=config)

    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            train_losses = train_one_epoch(diffusion, train_loader, optimizer, device, epoch, args.grad_clip)
            val_losses = validate_diffusion_losses(diffusion, test_loader, device)
            log_values = {
                **{f"train_{name}_loss": value for name, value in train_losses.items()},
                **{f"val_{name}_loss": value for name, value in val_losses.items()},
                "epoch": epoch,
            }
            if epoch % args.sample_eval_interval == 0:
                sample = evaluate_latent(diffusion, test_loader, device, "iterative", f"Sample eval epoch {epoch}")
                log_values.update({f"val_iterative_{key}": value for key, value in sample.items() if isinstance(value, (int, float))})

            improved = val_losses["total"] < best_loss
            if improved:
                best_loss, best_epoch, stale_epochs = val_losses["total"], epoch, 0
                save_checkpoint(ckpt_dir / "best.pt", net, optimizer, epoch, best_loss, config)
            else:
                stale_epochs += 1
            save_checkpoint(ckpt_dir / "latest.pt", net, optimizer, epoch, best_loss, config)

            if use_wandb:
                wandb.log(log_values, step=epoch)
            values = " | ".join(f"{key}: {value:.6f}" for key, value in log_values.items() if isinstance(value, float))
            print(f"Epoch {epoch:03d} | {values}")
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} ({best_loss:.6f}).")
                break

    checkpoint = ckpt_dir / "best.pt"
    if not checkpoint.exists():
        checkpoint = ckpt_dir / "latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found for evaluation in {ckpt_dir}")
    print(f"\nLoading checkpoint for final evaluation: {checkpoint}")
    load_model_checkpoint(checkpoint, net, device)

    all_metrics = []
    for mode in args.eval_modes:
        metrics = evaluate_latent(diffusion, test_loader, device, mode, f"Final evaluation ({mode})")
        all_metrics.append(metrics)
        text = format_metrics(metrics)
        print(f"\n{text}")
        with (metrics_dir / f"eval_{mode}.txt").open("w", encoding="utf-8") as file:
            file.write(text)
    with (metrics_dir / "eval_all_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    if use_wandb:
        for metrics in all_metrics:
            mode = metrics["mode"]
            wandb.log({f"final_{mode}_{key}": value for key, value in metrics.items() if isinstance(value, (int, float))})
        wandb.finish()
    print(f"\nDone. Outputs saved to: {exp_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train latent cold diffusion on MAD22 or SMDD DiffAE embeddings.")
    parser.add_argument("--embeddings-root", default="/nas-ctm01/homes/dacordeiro/article")
    parser.add_argument("--data-dir", default=None, help="Direct scenario path; overrides automatic resolution.")
    parser.add_argument("--dataset", choices=["smdd", "mad22"], default="smdd")
    parser.add_argument("--mad22-method", choices=MAD22_METHODS, default="MIPGAN_II")
    parser.add_argument("--mad22-scenario", choices=MAD22_SCENARIOS, default="identity_disjoint_80_20")
    parser.add_argument("--smdd-scenario", choices=SMDD_SCENARIOS, default="identity_disjoint_fixed_1000")

    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-mmap", action="store_true", help="Load arrays fully instead of using memory maps.")

    parser.add_argument("--num-timesteps", type=int, default=10)
    parser.add_argument("--terminal-loss-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=10)
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

    parser.add_argument("--eval-modes", nargs="+", default=["one_shot", "iterative"],
                        choices=["one_shot", "iterative"])
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate the run checkpoint.")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="article")
    return parser


def main() -> None:
    train_and_evaluate(build_argparser().parse_args())


if __name__ == "__main__":
    main()

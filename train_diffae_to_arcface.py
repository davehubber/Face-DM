import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)

        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -----------------------------
# Data alignment
# -----------------------------

def load_and_align_embeddings(
    diffae_embeddings_path: Path,
    diffae_metadata_path: Path,
    arcface_embeddings_path: Path,
    arcface_metadata_path: Path,
    join_key: str = "filename",
):
    diffae = np.load(diffae_embeddings_path).astype(np.float32)
    arcface = np.load(arcface_embeddings_path).astype(np.float32)

    diffae_meta = pd.read_csv(diffae_metadata_path).reset_index(drop=True)
    arcface_meta = pd.read_csv(arcface_metadata_path).reset_index(drop=True)

    if len(diffae_meta) != len(diffae):
        raise ValueError(
            f"DiffAE metadata rows ({len(diffae_meta)}) do not match embeddings ({len(diffae)})."
        )

    if len(arcface_meta) != len(arcface):
        raise ValueError(
            f"ArcFace metadata rows ({len(arcface_meta)}) do not match embeddings ({len(arcface)})."
        )

    if join_key not in diffae_meta.columns:
        raise ValueError(f"join_key='{join_key}' not found in DiffAE metadata.")

    if join_key not in arcface_meta.columns:
        raise ValueError(f"join_key='{join_key}' not found in ArcFace metadata.")

    if diffae_meta[join_key].duplicated().any():
        duplicates = diffae_meta.loc[diffae_meta[join_key].duplicated(), join_key].head().tolist()
        raise ValueError(f"Duplicate join_key values in DiffAE metadata. Examples: {duplicates}")

    if arcface_meta[join_key].duplicated().any():
        duplicates = arcface_meta.loc[arcface_meta[join_key].duplicated(), join_key].head().tolist()
        raise ValueError(f"Duplicate join_key values in ArcFace metadata. Examples: {duplicates}")

    diffae_meta = diffae_meta.copy()
    arcface_meta = arcface_meta.copy()

    diffae_meta["diffae_row"] = np.arange(len(diffae_meta))
    arcface_meta["arcface_row"] = np.arange(len(arcface_meta))

    merged = diffae_meta[[join_key, "diffae_row"]].merge(
        arcface_meta[[join_key, "arcface_row"]],
        on=join_key,
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError(
            f"No rows matched using join_key='{join_key}'. "
            "Try --join-key image_path if paths are identical in both metadata files."
        )

    if len(merged) < min(len(diffae), len(arcface)):
        print(
            f"Warning: only {len(merged)} matched rows. "
            f"DiffAE rows={len(diffae)}, ArcFace rows={len(arcface)}."
        )

    diffae_idx = merged["diffae_row"].to_numpy()
    arcface_idx = merged["arcface_row"].to_numpy()

    x = diffae[diffae_idx]
    y = arcface[arcface_idx]

    # ArcFace targets should already be L2-normalized from the previous script,
    # but normalize again for safety.
    y = l2_normalize_np(y).astype(np.float32)

    return x.astype(np.float32), y.astype(np.float32), merged


# -----------------------------
# Dataset
# -----------------------------

class EmbeddingDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# -----------------------------
# Model
# -----------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class DiffAEToArcFaceMLP(nn.Module):
    """
    Residual MLP for mapping DiffAE semantic embeddings to ArcFace identity embeddings.

    Inputs:
        standardized raw DiffAE z_sem vectors

    Outputs:
        raw 512-D vectors.
        During loss/evaluation, outputs are L2-normalized before comparison.
    """
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 1024,
        num_blocks: int = 6,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        h = self.input(x)
        h = self.blocks(h)
        return self.output(h)


# -----------------------------
# Loss and metrics
# -----------------------------

def prediction_loss(pred_raw, target_unit, mse_weight: float = 1.0):
    pred_unit = F.normalize(pred_raw, p=2, dim=1)
    target_unit = F.normalize(target_unit, p=2, dim=1)

    cosine_loss = 1.0 - F.cosine_similarity(pred_unit, target_unit, dim=1).mean()
    mse_loss = F.mse_loss(pred_unit, target_unit)

    loss = cosine_loss + mse_weight * mse_loss

    return loss, {
        "cosine_loss": float(cosine_loss.detach().cpu()),
        "mse_loss": float(mse_loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, device, mse_weight: float):
    model.eval()

    total_loss = 0.0
    total_cos_loss = 0.0
    total_mse_loss = 0.0
    total_n = 0

    all_pred = []
    all_target = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred_raw = model(x)
        loss, parts = prediction_loss(pred_raw, y, mse_weight=mse_weight)

        bs = x.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        total_cos_loss += parts["cosine_loss"] * bs
        total_mse_loss += parts["mse_loss"] * bs
        total_n += bs

        pred_unit = F.normalize(pred_raw, p=2, dim=1)
        target_unit = F.normalize(y, p=2, dim=1)

        all_pred.append(pred_unit.cpu().numpy())
        all_target.append(target_unit.cpu().numpy())

    pred = np.concatenate(all_pred, axis=0)
    target = np.concatenate(all_target, axis=0)

    cos = np.sum(pred * target, axis=1)
    cos = np.clip(cos, -1.0, 1.0)

    l2 = np.linalg.norm(pred - target, axis=1)
    angle_deg = np.degrees(np.arccos(cos))

    metrics = {
        "loss": total_loss / total_n,
        "cosine_loss": total_cos_loss / total_n,
        "mse_loss": total_mse_loss / total_n,

        "cosine_mean": float(np.mean(cos)),
        "cosine_std": float(np.std(cos)),
        "cosine_median": float(np.median(cos)),
        "cosine_p05": float(np.percentile(cos, 5)),
        "cosine_p95": float(np.percentile(cos, 95)),
        "cosine_min": float(np.min(cos)),
        "cosine_max": float(np.max(cos)),

        "l2_mean": float(np.mean(l2)),
        "l2_median": float(np.median(l2)),

        "angle_deg_mean": float(np.mean(angle_deg)),
        "angle_deg_median": float(np.median(angle_deg)),
    }

    return metrics, pred, target


@torch.no_grad()
def retrieval_accuracy(pred_unit: np.ndarray, target_unit: np.ndarray, batch_size: int = 512):
    """
    Retrieval diagnostic:
    for each predicted ArcFace embedding, is the corresponding true ArcFace embedding
    the nearest one among all test targets?

    This is not the same as identity classification, but it is a useful sanity check.
    """
    pred_unit = l2_normalize_np(pred_unit.astype(np.float32))
    target_unit = l2_normalize_np(target_unit.astype(np.float32))

    n = pred_unit.shape[0]
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = pred_unit[start:end] @ target_unit.T

        top10 = np.argpartition(-sims, kth=min(10, n) - 1, axis=1)[:, :min(10, n)]

        for local_i, candidates in enumerate(top10):
            true_idx = start + local_i

            candidate_scores = sims[local_i, candidates]
            ordered = candidates[np.argsort(-candidate_scores)]

            if ordered[0] == true_idx:
                correct_top1 += 1
            if true_idx in ordered[:min(5, n)]:
                correct_top5 += 1
            if true_idx in ordered[:min(10, n)]:
                correct_top10 += 1

    return {
        "retrieval_top1": correct_top1 / n,
        "retrieval_top5": correct_top5 / n,
        "retrieval_top10": correct_top10 / n,
    }


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--diffae-embeddings", type=str, required=True)
    parser.add_argument("--diffae-metadata", type=str, required=True)
    parser.add_argument("--arcface-embeddings", type=str, required=True)
    parser.add_argument("--arcface-metadata", type=str, required=True)

    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--experiments-root", type=str, default="experiments")

    parser.add_argument("--join-key", type=str, default="filename")

    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)

    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--mse-weight", type=float, default=1.0)

    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-retrieval", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    run_dir = Path(args.experiments_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    save_json(config, run_dir / "config.json")

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    x_raw, y, merged = load_and_align_embeddings(
        diffae_embeddings_path=Path(args.diffae_embeddings),
        diffae_metadata_path=Path(args.diffae_metadata),
        arcface_embeddings_path=Path(args.arcface_embeddings),
        arcface_metadata_path=Path(args.arcface_metadata),
        join_key=args.join_key,
    )

    n, input_dim = x_raw.shape
    _, output_dim = y.shape

    print(f"Aligned samples: {n}")
    print(f"DiffAE input dim: {input_dim}")
    print(f"ArcFace target dim: {output_dim}")

    if input_dim != 512:
        print(f"Warning: expected DiffAE z_sem dimension 512, got {input_dim}")

    if output_dim != 512:
        print(f"Warning: expected ArcFace dimension 512, got {output_dim}")

    # Save alignment table
    merged.to_csv(run_dir / "aligned_metadata.csv", index=False)

    # Random split
    indices = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}. "
            "Adjust --train-frac and --val-frac."
        )

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    np.savez(
        run_dir / "splits.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    # DiffAE preprocessing:
    # The DiffAE embeddings from the previous script are raw z_sem vectors.
    # Standardize them using train-only statistics.
    x_mean = x_raw[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    x_std = x_raw[train_idx].std(axis=0, keepdims=True).astype(np.float32)
    x_std = np.maximum(x_std, 1e-6)

    np.save(run_dir / "diffae_train_mean.npy", x_mean)
    np.save(run_dir / "diffae_train_std.npy", x_std)

    x = ((x_raw - x_mean) / x_std).astype(np.float32)

    # ArcFace targets are already L2-normalized, but we normalized again when loading.
    target_norms = np.linalg.norm(y, axis=1)
    print(
        "ArcFace target norm check: "
        f"mean={target_norms.mean():.6f}, "
        f"std={target_norms.std():.6f}, "
        f"min={target_norms.min():.6f}, "
        f"max={target_norms.max():.6f}"
    )

    train_ds = EmbeddingDataset(x[train_idx], y[train_idx])
    val_ds = EmbeddingDataset(x[val_idx], y[val_idx])
    test_ds = EmbeddingDataset(x[test_idx], y[test_idx])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = DiffAEToArcFaceMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    best_val_cosine = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_cos_loss_sum = 0.0
        train_mse_loss_sum = 0.0
        train_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", leave=False)

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred_raw = model(xb)
            loss, parts = prediction_loss(pred_raw, yb, mse_weight=args.mse_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            bs = xb.shape[0]
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_cos_loss_sum += parts["cosine_loss"] * bs
            train_mse_loss_sum += parts["mse_loss"] * bs
            train_n += bs

            pbar.set_postfix({
                "loss": train_loss_sum / train_n,
                "lr": optimizer.param_groups[0]["lr"],
            })

        train_metrics = {
            "train_loss": train_loss_sum / train_n,
            "train_cosine_loss": train_cos_loss_sum / train_n,
            "train_mse_loss": train_mse_loss_sum / train_n,
        }

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            device=device,
            mse_weight=args.mse_weight,
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.6f} | "
            f"val_loss={row['val_loss']:.6f} | "
            f"val_cosine_mean={row['val_cosine_mean']:.6f} | "
            f"val_angle_deg_mean={row['val_angle_deg_mean']:.3f}"
        )

        if val_metrics["cosine_mean"] > best_val_cosine:
            best_val_cosine = val_metrics["cosine_mean"]
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "x_mean": x_mean,
                "x_std": x_std,
                "best_epoch": best_epoch,
                "best_val_cosine": best_val_cosine,
            }

            torch.save(checkpoint, run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch={best_epoch}, best val cosine={best_val_cosine:.6f}"
            )
            break

    # Load best model
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, val_pred, val_target = evaluate(
        model,
        val_loader,
        device=device,
        mse_weight=args.mse_weight,
    )

    test_metrics, test_pred, test_target = evaluate(
        model,
        test_loader,
        device=device,
        mse_weight=args.mse_weight,
    )

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_cosine_during_training": float(best_val_cosine),
        "val": val_metrics,
        "test": test_metrics,
    }

    if args.compute_retrieval:
        print("Computing retrieval diagnostics on test set...")
        retrieval = retrieval_accuracy(test_pred, test_target, batch_size=512)
        metrics["test"].update(retrieval)

    save_json(metrics, run_dir / "metrics.json")

    np.save(run_dir / "test_pred_arcface_unit.npy", test_pred.astype(np.float32))
    np.save(run_dir / "test_target_arcface_unit.npy", test_target.astype(np.float32))

    with open(run_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("DiffAE semantic embedding → ArcFace embedding regression\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Run name: {args.run_name}\n")
        f.write(f"Aligned samples: {n}\n")
        f.write(f"Train/val/test: {n_train}/{n_val}/{n_test}\n")
        f.write(f"Input dim: {input_dim}\n")
        f.write(f"Output dim: {output_dim}\n\n")

        f.write("Preprocessing\n")
        f.write("-" * 80 + "\n")
        f.write("DiffAE z_sem inputs: standardized with train-set mean/std.\n")
        f.write("ArcFace targets: L2-normalized unit vectors.\n")
        f.write("Model outputs: L2-normalized before loss/evaluation.\n\n")

        f.write("Architecture\n")
        f.write("-" * 80 + "\n")
        f.write(f"Residual MLP: hidden_dim={args.hidden_dim}, num_blocks={args.num_blocks}, dropout={args.dropout}\n\n")

        f.write("Best model\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation cosine during training: {best_val_cosine:.6f}\n\n")

        f.write("Validation metrics\n")
        f.write("-" * 80 + "\n")
        for k, v in val_metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\nTest metrics\n")
        f.write("-" * 80 + "\n")
        for k, v in metrics["test"].items():
            f.write(f"{k}: {v}\n")

    print("\nDone.")
    print(f"Saved experiment to: {run_dir}")
    print(f"Best epoch: {best_epoch}")
    print(f"Test cosine mean: {test_metrics['cosine_mean']:.6f}")
    print(f"Test angle mean: {test_metrics['angle_deg_mean']:.3f} degrees")


if __name__ == "__main__":
    main()

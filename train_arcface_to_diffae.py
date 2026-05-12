import argparse
import json
import math
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


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


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
    arcface_embeddings_path: Path,
    arcface_metadata_path: Path,
    diffae_embeddings_path: Path,
    diffae_metadata_path: Path,
    join_key: str = "filename",
):
    arcface = np.load(arcface_embeddings_path).astype(np.float32)
    diffae = np.load(diffae_embeddings_path).astype(np.float32)

    arcface_meta = pd.read_csv(arcface_metadata_path).reset_index(drop=True)
    diffae_meta = pd.read_csv(diffae_metadata_path).reset_index(drop=True)

    if len(arcface_meta) != len(arcface):
        raise ValueError(
            f"ArcFace metadata rows ({len(arcface_meta)}) do not match embeddings ({len(arcface)})."
        )

    if len(diffae_meta) != len(diffae):
        raise ValueError(
            f"DiffAE metadata rows ({len(diffae_meta)}) do not match embeddings ({len(diffae)})."
        )

    if join_key not in arcface_meta.columns:
        raise ValueError(f"join_key='{join_key}' not found in ArcFace metadata.")

    if join_key not in diffae_meta.columns:
        raise ValueError(f"join_key='{join_key}' not found in DiffAE metadata.")

    if arcface_meta[join_key].duplicated().any():
        duplicates = arcface_meta.loc[arcface_meta[join_key].duplicated(), join_key].head().tolist()
        raise ValueError(f"Duplicate join_key values in ArcFace metadata. Examples: {duplicates}")

    if diffae_meta[join_key].duplicated().any():
        duplicates = diffae_meta.loc[diffae_meta[join_key].duplicated(), join_key].head().tolist()
        raise ValueError(f"Duplicate join_key values in DiffAE metadata. Examples: {duplicates}")

    arcface_meta = arcface_meta.copy()
    diffae_meta = diffae_meta.copy()

    arcface_meta["arcface_row"] = np.arange(len(arcface_meta))
    diffae_meta["diffae_row"] = np.arange(len(diffae_meta))

    merged = arcface_meta[[join_key, "arcface_row"]].merge(
        diffae_meta[[join_key, "diffae_row"]],
        on=join_key,
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError(
            f"No rows matched using join_key='{join_key}'. "
            "Try --join-key image_path if paths are identical in both metadata files."
        )

    if len(merged) < min(len(arcface), len(diffae)):
        print(
            f"Warning: only {len(merged)} matched rows. "
            f"ArcFace rows={len(arcface)}, DiffAE rows={len(diffae)}."
        )

    arcface_idx = merged["arcface_row"].to_numpy()
    diffae_idx = merged["diffae_row"].to_numpy()

    x = arcface[arcface_idx].astype(np.float32)
    y = diffae[diffae_idx].astype(np.float32)

    # Your ArcFace extraction script saved L2-normalized vectors.
    # Normalize again for safety.
    x = l2_normalize_np(x).astype(np.float32)

    return x, y, merged


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


class ArcFaceToDiffAEMLP(nn.Module):
    """
    Residual MLP for mapping ArcFace identity embeddings to DiffAE semantic embeddings.

    Input:
        standardized ArcFace embedding, originally 512-D unit vector

    Output:
        standardized DiffAE z_sem vector

    The output is not L2-normalized, because DiffAE z_sem is not a unit-sphere embedding.
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

def prediction_loss(pred_z: torch.Tensor, target_z: torch.Tensor, cosine_weight: float = 0.10):
    """
    Main loss is MSE in standardized DiffAE z_sem space.

    A small cosine term is included only as a geometry-preserving auxiliary loss.
    It does not force the output to unit norm.
    """
    mse_loss = F.mse_loss(pred_z, target_z)

    cos = F.cosine_similarity(pred_z, target_z, dim=1)
    cosine_loss = 1.0 - cos.mean()

    loss = mse_loss + cosine_weight * cosine_loss

    return loss, {
        "mse_loss": float(mse_loss.detach().cpu()),
        "cosine_loss": float(cosine_loss.detach().cpu()),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    cosine_weight: float,
):
    model.eval()

    total_loss = 0.0
    total_mse_loss = 0.0
    total_cosine_loss = 0.0
    total_n = 0

    all_pred_z = []
    all_target_z = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred_z = model(x)
        loss, parts = prediction_loss(pred_z, y, cosine_weight=cosine_weight)

        bs = x.shape[0]
        total_loss += float(loss.detach().cpu()) * bs
        total_mse_loss += parts["mse_loss"] * bs
        total_cosine_loss += parts["cosine_loss"] * bs
        total_n += bs

        all_pred_z.append(pred_z.detach().cpu().numpy())
        all_target_z.append(y.detach().cpu().numpy())

    pred_z = np.concatenate(all_pred_z, axis=0).astype(np.float32)
    target_z = np.concatenate(all_target_z, axis=0).astype(np.float32)

    # Metrics in standardized z_sem space.
    diff_z = pred_z - target_z
    mse_z_per_sample = np.mean(diff_z ** 2, axis=1)
    mae_z_per_sample = np.mean(np.abs(diff_z), axis=1)

    pred_z_unit = l2_normalize_np(pred_z)
    target_z_unit = l2_normalize_np(target_z)
    cos_z = np.sum(pred_z_unit * target_z_unit, axis=1)
    cos_z = np.clip(cos_z, -1.0, 1.0)

    # Convert back to raw DiffAE z_sem space.
    pred_raw = pred_z * y_std + y_mean
    target_raw = target_z * y_std + y_mean

    diff_raw = pred_raw - target_raw
    mse_raw_per_sample = np.mean(diff_raw ** 2, axis=1)
    mae_raw_per_sample = np.mean(np.abs(diff_raw), axis=1)
    l2_raw_per_sample = np.linalg.norm(diff_raw, axis=1)

    pred_raw_unit = l2_normalize_np(pred_raw)
    target_raw_unit = l2_normalize_np(target_raw)
    cos_raw = np.sum(pred_raw_unit * target_raw_unit, axis=1)
    cos_raw = np.clip(cos_raw, -1.0, 1.0)

    angle_raw_deg = np.degrees(np.arccos(cos_raw))

    metrics = {
        "loss": total_loss / total_n,
        "mse_loss": total_mse_loss / total_n,
        "cosine_loss": total_cosine_loss / total_n,

        "standardized_mse_mean": float(np.mean(mse_z_per_sample)),
        "standardized_mse_median": float(np.median(mse_z_per_sample)),
        "standardized_mae_mean": float(np.mean(mae_z_per_sample)),
        "standardized_mae_median": float(np.median(mae_z_per_sample)),
        "standardized_cosine_mean": float(np.mean(cos_z)),
        "standardized_cosine_median": float(np.median(cos_z)),
        "standardized_cosine_p05": float(np.percentile(cos_z, 5)),
        "standardized_cosine_p95": float(np.percentile(cos_z, 95)),

        "raw_mse_mean": float(np.mean(mse_raw_per_sample)),
        "raw_mse_median": float(np.median(mse_raw_per_sample)),
        "raw_mae_mean": float(np.mean(mae_raw_per_sample)),
        "raw_mae_median": float(np.median(mae_raw_per_sample)),
        "raw_l2_mean": float(np.mean(l2_raw_per_sample)),
        "raw_l2_median": float(np.median(l2_raw_per_sample)),
        "raw_cosine_mean": float(np.mean(cos_raw)),
        "raw_cosine_std": float(np.std(cos_raw)),
        "raw_cosine_median": float(np.median(cos_raw)),
        "raw_cosine_p05": float(np.percentile(cos_raw, 5)),
        "raw_cosine_p95": float(np.percentile(cos_raw, 95)),
        "raw_angle_deg_mean": float(np.mean(angle_raw_deg)),
        "raw_angle_deg_median": float(np.median(angle_raw_deg)),
    }

    return metrics, pred_raw.astype(np.float32), target_raw.astype(np.float32)


@torch.no_grad()
def retrieval_accuracy(pred_raw: np.ndarray, target_raw: np.ndarray, batch_size: int = 512):
    """
    Retrieval diagnostic:
    For each predicted DiffAE z_sem, is the corresponding true DiffAE z_sem
    the nearest one among all test targets by cosine similarity?
    """
    pred_unit = l2_normalize_np(pred_raw.astype(np.float32))
    target_unit = l2_normalize_np(target_raw.astype(np.float32))

    n = pred_unit.shape[0]
    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = pred_unit[start:end] @ target_unit.T

        k = min(10, n)
        topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]

        for local_i, candidates in enumerate(topk):
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

    parser.add_argument("--arcface-embeddings", type=str, required=True)
    parser.add_argument("--arcface-metadata", type=str, required=True)
    parser.add_argument("--diffae-embeddings", type=str, required=True)
    parser.add_argument("--diffae-metadata", type=str, required=True)

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

    parser.add_argument(
        "--cosine-weight",
        type=float,
        default=0.10,
        help="Auxiliary cosine loss weight in standardized DiffAE target space.",
    )

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

    x_raw, y_raw, merged = load_and_align_embeddings(
        arcface_embeddings_path=Path(args.arcface_embeddings),
        arcface_metadata_path=Path(args.arcface_metadata),
        diffae_embeddings_path=Path(args.diffae_embeddings),
        diffae_metadata_path=Path(args.diffae_metadata),
        join_key=args.join_key,
    )

    n, input_dim = x_raw.shape
    _, output_dim = y_raw.shape

    print(f"Aligned samples: {n}")
    print(f"ArcFace input dim: {input_dim}")
    print(f"DiffAE target dim: {output_dim}")

    if input_dim != 512:
        print(f"Warning: expected ArcFace dimension 512, got {input_dim}")

    if output_dim != 512:
        print(f"Warning: expected DiffAE z_sem dimension 512, got {output_dim}")

    merged.to_csv(run_dir / "aligned_metadata.csv", index=False)

    # Random split.
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

    # -----------------------------
    # Normalization strategy
    # -----------------------------
    # ArcFace input:
    # already L2-normalized from extraction, but feature-wise standardization helps MLP training.
    #
    # DiffAE target:
    # raw z_sem vectors from model.encode(x), so we z-score them using train-only statistics.
    # The model predicts standardized z_sem.
    # -----------------------------

    x_mean = x_raw[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    x_std = x_raw[train_idx].std(axis=0, keepdims=True).astype(np.float32)
    x_std = np.maximum(x_std, 1e-6)

    y_mean = y_raw[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    y_std = y_raw[train_idx].std(axis=0, keepdims=True).astype(np.float32)
    y_std = np.maximum(y_std, 1e-6)

    np.save(run_dir / "arcface_train_mean.npy", x_mean)
    np.save(run_dir / "arcface_train_std.npy", x_std)
    np.save(run_dir / "diffae_train_mean.npy", y_mean)
    np.save(run_dir / "diffae_train_std.npy", y_std)

    x = ((x_raw - x_mean) / x_std).astype(np.float32)
    y = ((y_raw - y_mean) / y_std).astype(np.float32)

    arcface_norms = np.linalg.norm(x_raw, axis=1)
    diffae_norms = np.linalg.norm(y_raw, axis=1)

    print(
        "ArcFace raw input norm check: "
        f"mean={arcface_norms.mean():.6f}, "
        f"std={arcface_norms.std():.6f}, "
        f"min={arcface_norms.min():.6f}, "
        f"max={arcface_norms.max():.6f}"
    )

    print(
        "DiffAE raw target norm check: "
        f"mean={diffae_norms.mean():.6f}, "
        f"std={diffae_norms.std():.6f}, "
        f"min={diffae_norms.min():.6f}, "
        f"max={diffae_norms.max():.6f}"
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

    model = ArcFaceToDiffAEMLP(
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

    best_val_raw_cosine = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_mse_loss_sum = 0.0
        train_cosine_loss_sum = 0.0
        train_n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", leave=False)

        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            loss, parts = prediction_loss(
                pred,
                yb,
                cosine_weight=args.cosine_weight,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            bs = xb.shape[0]
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_mse_loss_sum += parts["mse_loss"] * bs
            train_cosine_loss_sum += parts["cosine_loss"] * bs
            train_n += bs

            pbar.set_postfix({
                "loss": train_loss_sum / train_n,
                "lr": optimizer.param_groups[0]["lr"],
            })

        train_metrics = {
            "train_loss": train_loss_sum / train_n,
            "train_mse_loss": train_mse_loss_sum / train_n,
            "train_cosine_loss": train_cosine_loss_sum / train_n,
        }

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            cosine_weight=args.cosine_weight,
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
            f"val_raw_cosine_mean={row['val_raw_cosine_mean']:.6f} | "
            f"val_raw_mse_mean={row['val_raw_mse_mean']:.6f}"
        )

        # Best model selected by raw-space cosine similarity.
        # You could also use val_loss or raw_mse_mean; cosine is a useful global semantic diagnostic.
        if val_metrics["raw_cosine_mean"] > best_val_raw_cosine:
            best_val_raw_cosine = val_metrics["raw_cosine_mean"]
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "best_epoch": best_epoch,
                "best_val_raw_cosine": best_val_raw_cosine,
            }

            torch.save(checkpoint, run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch={best_epoch}, best val raw cosine={best_val_raw_cosine:.6f}"
            )
            break

    # Load best model.
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics, val_pred_raw, val_target_raw = evaluate(
        model,
        val_loader,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        cosine_weight=args.cosine_weight,
    )

    test_metrics, test_pred_raw, test_target_raw = evaluate(
        model,
        test_loader,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        cosine_weight=args.cosine_weight,
    )

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_raw_cosine_during_training": float(best_val_raw_cosine),
        "val": val_metrics,
        "test": test_metrics,
    }

    if args.compute_retrieval:
        print("Computing retrieval diagnostics on test set...")
        retrieval = retrieval_accuracy(test_pred_raw, test_target_raw, batch_size=512)
        metrics["test"].update(retrieval)

    save_json(metrics, run_dir / "metrics.json")

    np.save(run_dir / "test_pred_diffae_zsem_raw.npy", test_pred_raw.astype(np.float32))
    np.save(run_dir / "test_target_diffae_zsem_raw.npy", test_target_raw.astype(np.float32))

    with open(run_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("ArcFace embedding → DiffAE semantic embedding regression\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Run name: {args.run_name}\n")
        f.write(f"Aligned samples: {n}\n")
        f.write(f"Train/val/test: {n_train}/{n_val}/{n_test}\n")
        f.write(f"Input dim: {input_dim}\n")
        f.write(f"Output dim: {output_dim}\n\n")

        f.write("Preprocessing\n")
        f.write("-" * 80 + "\n")
        f.write("ArcFace inputs: L2-normalized from extraction, then train-set feature standardized.\n")
        f.write("DiffAE targets: raw z_sem vectors, train-set feature standardized.\n")
        f.write("Model outputs: standardized DiffAE z_sem vectors.\n")
        f.write("Evaluation: predictions are converted back to raw DiffAE z_sem space.\n\n")

        f.write("Architecture\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"Residual MLP: hidden_dim={args.hidden_dim}, "
            f"num_blocks={args.num_blocks}, dropout={args.dropout}\n\n"
        )

        f.write("Best model\n")
        f.write("-" * 80 + "\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation raw cosine during training: {best_val_raw_cosine:.6f}\n\n")

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
    print(f"Test raw cosine mean: {test_metrics['raw_cosine_mean']:.6f}")
    print(f"Test raw MSE mean: {test_metrics['raw_mse_mean']:.6f}")
    print(f"Test raw L2 mean: {test_metrics['raw_l2_mean']:.6f}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Data loading
# -------------------------

def read_metadata_csv(metadata_path: Path) -> List[dict]:
    rows = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "embedding_index",
            "filename",
            "image_path",
            "identity_id",
            "label",
            "split",
        }

        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Metadata file is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            row["embedding_index"] = int(row["embedding_index"])
            row["label"] = int(row["label"])
            rows.append(row)

    rows = sorted(rows, key=lambda r: r["embedding_index"])
    return rows


def validate_dataset(embeddings: np.ndarray, rows: List[dict]):
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D, got shape {embeddings.shape}")

    if len(rows) != embeddings.shape[0]:
        raise ValueError(
            f"Metadata rows and embeddings do not match: "
            f"{len(rows)} rows vs {embeddings.shape[0]} embeddings"
        )

    expected_indices = list(range(len(rows)))
    actual_indices = [row["embedding_index"] for row in rows]

    if actual_indices != expected_indices:
        raise ValueError(
            "embedding_index values are not exactly 0..N-1 after sorting. "
            "This script assumes metadata rows directly index the embedding array."
        )

    labels = sorted(set(row["label"] for row in rows))
    expected_labels = list(range(len(labels)))

    if labels != expected_labels:
        raise ValueError(
            "Labels are not contiguous from 0..num_classes-1. "
            f"Found min={min(labels)}, max={max(labels)}, num_unique={len(labels)}"
        )

    split_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    for split in ["train", "val", "test"]:
        if split not in split_counts:
            raise ValueError(f"Missing split: {split}")

    print("Dataset validation passed.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Number of classes: {len(labels)}")
    print(f"Split counts: {split_counts}")


class DiffAESemanticIdentityDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, rows: List[dict], split: str):
        self.embeddings = embeddings
        self.rows = [row for row in rows if row["split"] == split]
        self.split = split

        if len(self.rows) == 0:
            raise ValueError(f"No rows found for split: {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        emb_idx = row["embedding_index"]

        x = torch.from_numpy(self.embeddings[emb_idx]).float()
        y = torch.tensor(row["label"], dtype=torch.long)

        return {
            "x": x,
            "y": y,
            "embedding_index": emb_idx,
            "filename": row["filename"],
            "identity_id": row["identity_id"],
        }


def make_balanced_sampler(dataset: DiffAESemanticIdentityDataset) -> WeightedRandomSampler:
    labels = [row["label"] for row in dataset.rows]

    label_counts: Dict[int, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    weights = [1.0 / label_counts[label] for label in labels]
    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    return sampler


# -------------------------
# Model
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

    def forward(self, x):
        return self.net(x)


# -------------------------
# Metrics
# -------------------------

@torch.no_grad()
def topk_correct(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5, 10)):
    num_classes = logits.shape[1]
    max_k = min(max(topk), num_classes)

    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()

    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = {}

    for k in topk:
        k_eff = min(k, num_classes)
        correct_k = correct[:k_eff].reshape(-1).float().sum().item()
        results[f"top{k}"] = correct_k

    return results


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    train: bool,
    amp: bool,
    topk=(1, 5, 10),
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    total_correct = {f"top{k}": 0.0 for k in topk}

    scaler = run_epoch.scaler if train and amp else None

    pbar = tqdm(loader, desc="train" if train else "eval", leave=False)

    for batch in pbar:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        batch_size = x.shape[0]

        if train:
            optimizer.zero_grad(set_to_none=True)

            if amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        else:
            with torch.no_grad():
                logits = model(x)
                loss = F.cross_entropy(logits, y)

        correct = topk_correct(logits.detach(), y, topk=topk)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        for key, value in correct.items():
            total_correct[key] += value

        pbar.set_postfix({
            "loss": total_loss / total_samples,
            "top1": total_correct["top1"] / total_samples,
        })

    if train and scheduler is not None:
        scheduler.step()

    metrics = {
        "loss": total_loss / total_samples,
    }

    for k in topk:
        metrics[f"top{k}"] = total_correct[f"top{k}"] / total_samples

    return metrics


run_epoch.scaler = torch.cuda.amp.GradScaler()


@torch.no_grad()
def save_predictions(
    model: nn.Module,
    dataset: DiffAESemanticIdentityDataset,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    topk: int = 5,
):
    model.eval()

    rows_by_embedding_index = {
        row["embedding_index"]: row
        for row in dataset.rows
    }

    output_rows = []

    for batch in tqdm(loader, desc=f"saving {dataset.split} predictions"):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        logits = model(x)
        probs = F.softmax(logits, dim=1)

        k_eff = min(topk, probs.shape[1])
        top_probs, top_labels = probs.topk(k_eff, dim=1)

        for i in range(x.shape[0]):
            emb_idx = int(batch["embedding_index"][i])
            row = rows_by_embedding_index[emb_idx]

            true_label = int(y[i].item())
            pred_label = int(top_labels[i, 0].item())
            pred_prob = float(top_probs[i, 0].item())

            output_row = {
                "embedding_index": emb_idx,
                "filename": row["filename"],
                "identity_id": row["identity_id"],
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_prob": pred_prob,
                "correct_top1": int(pred_label == true_label),
            }

            for j in range(k_eff):
                output_row[f"top{j + 1}_label"] = int(top_labels[i, j].item())
                output_row[f"top{j + 1}_prob"] = float(top_probs[i, j].item())

            output_rows.append(output_row)

    fieldnames = list(output_rows[0].keys())

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory produced by the CelebA DiffAE encoding script.",
    )
    parser.add_argument(
        "--embeddings-file",
        type=str,
        default="celeba_diffae_zsem_zscore.npy",
        help="Embedding file inside dataset-dir. Use the z-scored file by default.",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="celeba_diffae_zsem_metadata.csv",
        help="Metadata CSV inside dataset-dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints, logs, and predictions.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument(
        "--no-balanced-sampler",
        action="store_true",
        help="Disable class-balanced sampling for the training loader.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = dataset_dir / args.embeddings_file
    metadata_path = dataset_dir / args.metadata_file

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    device = torch.device(args.device)

    print(f"Dataset dir: {dataset_dir}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")

    embeddings = np.load(embeddings_path).astype(np.float32)
    rows = read_metadata_csv(metadata_path)

    validate_dataset(embeddings, rows)

    input_dim = embeddings.shape[1]
    num_classes = len(set(row["label"] for row in rows))

    train_dataset = DiffAESemanticIdentityDataset(embeddings, rows, split="train")
    val_dataset = DiffAESemanticIdentityDataset(embeddings, rows, split="val")
    test_dataset = DiffAESemanticIdentityDataset(embeddings, rows, split="test")

    if args.no_balanced_sampler:
        train_sampler = None
        train_shuffle = True
        print("Training sampler: normal shuffled sampling")
    else:
        train_sampler = make_balanced_sampler(train_dataset)
        train_shuffle = False
        print("Training sampler: class-balanced WeightedRandomSampler")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = IdentityMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    amp = (device.type == "cuda") and (not args.no_amp)

    print(model)
    print(f"AMP enabled: {amp}")

    config = vars(args).copy()
    config.update({
        "input_dim": input_dim,
        "num_classes": num_classes,
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
    })

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics_history = []

    best_val_top1 = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    best_ckpt_path = out_dir / "best_model.pt"
    last_ckpt_path = out_dir / "last_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            train=True,
            amp=amp,
            topk=(1, 5, 10),
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scheduler=None,
            device=device,
            train=False,
            amp=False,
            topk=(1, 5, 10),
        )

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "train_top10": train_metrics["top10"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "val_top10": val_metrics["top10"],
        }

        metrics_history.append(row)

        print(
            f"train loss={row['train_loss']:.4f} "
            f"top1={row['train_top1']:.4f} "
            f"top5={row['train_top5']:.4f} "
            f"top10={row['train_top10']:.4f}"
        )
        print(
            f"val   loss={row['val_loss']:.4f} "
            f"top1={row['val_top1']:.4f} "
            f"top5={row['val_top5']:.4f} "
            f"top10={row['val_top10']:.4f}"
        )

        # Save last checkpoint every epoch.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            last_ckpt_path,
        )

        # Main selection criterion: validation top-1.
        # Tie-breaker: validation loss.
        improved = (
            val_metrics["top1"] > best_val_top1
            or (
                val_metrics["top1"] == best_val_top1
                and val_metrics["loss"] < best_val_loss
            )
        )

        if improved:
            best_val_top1 = val_metrics["top1"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )

            print(f"Saved new best checkpoint: {best_ckpt_path}")

        else:
            epochs_without_improvement += 1

        # Save metrics CSV after every epoch.
        metrics_csv_path = out_dir / "metrics.csv"

        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_history[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_history)

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after {args.patience} epochs without validation improvement."
            )
            break

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation top-1: {best_val_top1:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print("\nLoading best checkpoint for final test evaluation.")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        scheduler=None,
        device=device,
        train=False,
        amp=False,
        topk=(1, 5, 10),
    )

    print("\nFinal test metrics:")
    print(f"test loss={test_metrics['loss']:.4f}")
    print(f"test top1={test_metrics['top1']:.4f}")
    print(f"test top5={test_metrics['top5']:.4f}")
    print(f"test top10={test_metrics['top10']:.4f}")

    final_results = {
        "best_epoch": best_epoch,
        "best_val_top1": best_val_top1,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "test_top10": test_metrics["top10"],
    }

    with open(out_dir / "final_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    save_predictions(
        model=model,
        dataset=test_dataset,
        loader=test_loader,
        device=device,
        out_path=out_dir / "test_predictions.csv",
        topk=5,
    )

    print(f"\nSaved final results to: {out_dir / 'final_results.json'}")
    print(f"Saved test predictions to: {out_dir / 'test_predictions.csv'}")
    print(f"Saved best checkpoint to: {best_ckpt_path}")
    print(f"Saved last checkpoint to: {last_ckpt_path}")


if __name__ == "__main__":
    main()

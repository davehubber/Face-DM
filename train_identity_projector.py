import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_zscore_stats(dataset_root: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    train_path = dataset_root / "semantic" / "train_zsem.pt"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train semantic file: {train_path}")

    train_pack = torch.load(train_path, map_location="cpu")
    train_z = train_pack["z_sem"].to(torch.float32)

    mean = train_z.mean(dim=0)
    std = train_z.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


class IdentityProjector(nn.Module):
    """
    Maps normalized DiffAE semantic embeddings to normalized ArcFace identity embeddings.

    Input:
        z_sem normalized with train-set z-score stats.

    Output:
        raw identity vector. Normalize it before cosine loss/comparison.
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            in_dim = input_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(in_dim, hidden_dim))
                if use_layernorm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim

            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ZsemIdentityDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        z_mean: torch.Tensor,
        z_std: torch.Tensor,
        max_rows: Optional[int] = None,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        semantic_path = dataset_root / "semantic" / f"{split}_zsem.pt"
        identity_path = dataset_root / "identity" / f"{split}_arcface.pt"

        if not semantic_path.exists():
            raise FileNotFoundError(f"Could not find semantic file: {semantic_path}")
        if not identity_path.exists():
            raise FileNotFoundError(f"Could not find identity file: {identity_path}")

        semantic_pack = torch.load(semantic_path, map_location="cpu")
        identity_pack = torch.load(identity_path, map_location="cpu")

        z_sem = semantic_pack["z_sem"].to(torch.float32)
        id_emb = identity_pack["identity_embeddings"].to(torch.float32)

        n_common = min(z_sem.shape[0], id_emb.shape[0])
        if max_rows is not None:
            n_common = min(n_common, int(max_rows))

        if n_common <= 0:
            raise ValueError(f"No common rows found for split '{split}'.")

        z_sem = z_sem[:n_common]
        id_emb = id_emb[:n_common]

        valid_mask = identity_pack.get("valid_mask", torch.ones(id_emb.shape[0], dtype=torch.bool))
        valid_mask = valid_mask[:n_common].to(torch.bool)

        finite_mask = torch.isfinite(id_emb).all(dim=1)
        nonzero_mask = id_emb.abs().sum(dim=1) > 0
        valid_mask = valid_mask & finite_mask & nonzero_mask

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()

        if valid_indices.numel() == 0:
            raise ValueError(
                f"No valid identity rows found for split '{split}'. "
                f"Check identity/{split}_arcface.pt."
            )

        z_sem = z_sem[valid_indices]
        id_emb = id_emb[valid_indices]

        z_mean = z_mean.to(torch.float32)
        z_std = z_std.to(torch.float32)

        self.z_sem = (z_sem - z_mean) / z_std
        self.id_emb = F.normalize(id_emb, dim=1)
        self.valid_indices = valid_indices.to(torch.long)

        self.sample_ids = list(semantic_pack["sample_ids"][:n_common])
        self.source_paths = list(semantic_pack["source_paths"][:n_common])
        self.relative_paths = list(semantic_pack.get("relative_paths", [""] * n_common)[:n_common])

        print(
            f"Loaded {split}: "
            f"{self.z_sem.shape[0]} valid rows from {n_common} common rows "
            f"(semantic rows={semantic_pack['z_sem'].shape[0]}, identity rows={identity_pack['identity_embeddings'].shape[0]})"
        )

    def __len__(self):
        return self.z_sem.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "z_sem": self.z_sem[idx],
            "identity": self.id_emb[idx],
            "row_index": self.valid_indices[idx],
        }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    compute_retrieval: bool = True,
) -> Dict[str, float]:
    model.eval()

    cos_values = []
    mse_values = []

    all_pred = []
    all_target = []

    for batch in dataloader:
        z = batch["z_sem"].to(device, non_blocking=True)
        target = batch["identity"].to(device, non_blocking=True)

        pred = model(z)
        pred = F.normalize(pred, dim=1)
        target = F.normalize(target, dim=1)

        cos = F.cosine_similarity(pred, target, dim=1)
        mse = F.mse_loss(pred, target, reduction="none").mean(dim=1)

        cos_values.append(cos.detach().cpu())
        mse_values.append(mse.detach().cpu())

        if compute_retrieval:
            all_pred.append(pred.detach().cpu())
            all_target.append(target.detach().cpu())

    cos_values = torch.cat(cos_values, dim=0)
    mse_values = torch.cat(mse_values, dim=0)

    metrics = {
        "cosine_mean": float(cos_values.mean().item()),
        "cosine_median": float(cos_values.median().item()),
        "cosine_p05": float(torch.quantile(cos_values, 0.05).item()),
        "mse_mean": float(mse_values.mean().item()),
    }

    if compute_retrieval and len(all_pred) > 0:
        pred_mat = F.normalize(torch.cat(all_pred, dim=0), dim=1)
        target_mat = F.normalize(torch.cat(all_target, dim=0), dim=1)

        # Retrieval can be memory-heavy for very large validation sets.
        # It is fine for ~10k, but you can disable it with --no-retrieval.
        sim = pred_mat @ target_mat.t()
        rank1 = sim.argmax(dim=1)
        correct = torch.arange(sim.shape[0])
        retrieval_top1 = (rank1 == correct).float().mean()

        true_sim = sim.diag()
        best_negative = sim.masked_fill(torch.eye(sim.shape[0], dtype=torch.bool), -1e9).max(dim=1).values
        margin = true_sim - best_negative

        metrics.update(
            {
                "retrieval_top1": float(retrieval_top1.item()),
                "retrieval_margin_mean": float(margin.mean().item()),
                "retrieval_margin_median": float(margin.median().item()),
            }
        )

    model.train()
    return metrics


def save_checkpoint(
    out_path: Path,
    model: nn.Module,
    args,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    train_dataset: ZsemIdentityDataset,
    val_dataset: ZsemIdentityDataset,
    best_val_cosine: float,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "IdentityProjector",
        "config": {
            "input_dim": args.input_dim,
            "output_dim": args.output_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "use_layernorm": not args.no_layernorm,
        },
        "z_mean": z_mean.cpu(),
        "z_std": z_std.cpu(),
        "input_normalization": "z_sem is expected to be z-score normalized with these train-set statistics",
        "target_normalization": "ArcFace identity embeddings are expected to be L2-normalized",
        "best_val_cosine": float(best_val_cosine),
        "train_valid_indices": train_dataset.valid_indices.cpu(),
        "val_valid_indices": val_dataset.valid_indices.cpu(),
        "notes": [
            "Rows correspond to indices in semantic/{split}_zsem.pt.",
            "Use train_valid_indices/val_valid_indices later to restrict identity-aware pair sampling.",
            "During de-averaging training, pass normalized predicted z_sem directly into this projector.",
            "Normalize projector outputs with F.normalize before cosine identity loss.",
        ],
    }

    torch.save(checkpoint, out_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/encoded_ffhq256_semantic_split",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Default: <dataset-root>/identity/identity_projector.pt",
    )

    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-layernorm", action="store_true")

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument(
        "--loss",
        choices=["cosine", "cosine_mse"],
        default="cosine",
        help="cosine = 1-cos. cosine_mse adds small MSE on normalized identity vectors.",
    )
    parser.add_argument("--mse-weight", type=float, default=0.1)

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional extra cap. Usually leave as None; the identity file length already caps it.",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=None,
        help="Optional extra cap. Usually leave as None; the identity file length already caps it.",
    )
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    out_path = Path(args.out_path) if args.out_path is not None else dataset_root / "identity" / "identity_projector.pt"
    metrics_path = out_path.with_suffix(".metrics.json")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    z_mean, z_std = load_zscore_stats(dataset_root)

    train_dataset = ZsemIdentityDataset(
        dataset_root=dataset_root,
        split="train",
        z_mean=z_mean,
        z_std=z_std,
        max_rows=args.max_train_rows,
    )
    val_dataset = ZsemIdentityDataset(
        dataset_root=dataset_root,
        split="val",
        z_mean=z_mean,
        z_std=z_std,
        max_rows=args.max_val_rows,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    model = IdentityProjector(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_layernorm=not args.no_layernorm,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_cosine = -float("inf")
    best_epoch = -1
    history = []

    print("\nTraining identity projector")
    print(f"Dataset root: {dataset_root}")
    print(f"Output path:  {out_path}")
    print(f"Train rows:   {len(train_dataset)}")
    print(f"Val rows:     {len(val_dataset)}")
    print(f"Device:       {device}")
    print("")

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_cos = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch in pbar:
            z = batch["z_sem"].to(device, non_blocking=True)
            target = batch["identity"].to(device, non_blocking=True)

            pred = model(z)
            pred = F.normalize(pred, dim=1)
            target = F.normalize(target, dim=1)

            cos = F.cosine_similarity(pred, target, dim=1)
            cosine_loss = (1.0 - cos).mean()

            if args.loss == "cosine_mse":
                mse_loss = F.mse_loss(pred, target)
                loss = cosine_loss + args.mse_weight * mse_loss
            else:
                loss = cosine_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            running_loss += float(loss.detach().item())
            running_cos += float(cos.detach().mean().item())
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": running_loss / max(num_batches, 1),
                    "cos": running_cos / max(num_batches, 1),
                }
            )

        train_loss = running_loss / max(num_batches, 1)
        train_cos = running_cos / max(num_batches, 1)

        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cosine_batch_mean": train_cos,
        }

        if do_val:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                compute_retrieval=not args.no_retrieval,
            )
            record.update({f"val_{k}": v for k, v in val_metrics.items()})

            val_cos = val_metrics["cosine_mean"]
            is_best = val_cos > best_val_cosine

            if is_best:
                best_val_cosine = val_cos
                best_epoch = epoch
                save_checkpoint(
                    out_path=out_path,
                    model=model,
                    args=args,
                    z_mean=z_mean,
                    z_std=z_std,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    best_val_cosine=best_val_cosine,
                )

            msg = (
                f"Epoch {epoch:03d} | "
                f"train loss {train_loss:.6f} | "
                f"train cos {train_cos:.6f} | "
                f"val cos {val_metrics['cosine_mean']:.6f} | "
                f"val p05 {val_metrics['cosine_p05']:.6f}"
            )

            if "retrieval_top1" in val_metrics:
                msg += f" | val top1 {val_metrics['retrieval_top1']:.4f}"

            if is_best:
                msg += " | saved best"

            print(msg)
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_loss:.6f} | "
                f"train cos {train_cos:.6f}"
            )

        history.append(record)

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_epoch": best_epoch,
                    "best_val_cosine": best_val_cosine,
                    "history": history,
                    "args": vars(args),
                },
                f,
                indent=2,
            )

    print("\nDone.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val cosine: {best_val_cosine:.6f}")
    print(f"Saved projector to: {out_path}")
    print(f"Saved metrics to:   {metrics_path}")


if __name__ == "__main__":
    main()

import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MorphPairDataset(Dataset):
    def __init__(self, morph_embeddings: np.ndarray, bonafide_embeddings: np.ndarray, metadata_path: Path):
        self.morph_embeddings = morph_embeddings
        self.bonafide_embeddings = bonafide_embeddings
        self.records = self.load_records(metadata_path)

    @staticmethod
    def load_records(metadata_path: Path):
        records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                morph_idx = int(row["embedding_index"])
                col_a, col_b = source_index_columns(row)
                source_idx_a = int(row[col_a])
                source_idx_b = int(row[col_b])
                if source_idx_a >= 0 and source_idx_b >= 0:
                    records.append((morph_idx, source_idx_a, source_idx_b))
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        morph_idx, source_idx_a, source_idx_b = self.records[index]
        morph = self.morph_embeddings[morph_idx]
        source_a = self.bonafide_embeddings[source_idx_a]
        source_b = self.bonafide_embeddings[source_idx_b]
        return torch.from_numpy(morph), torch.from_numpy(source_a), torch.from_numpy(source_b)


def source_index_columns(row):
    if "source_idx_A_in_train_bf" in row:
        return "source_idx_A_in_train_bf", "source_idx_B_in_train_bf"
    return "source_idx_A_in_eval_bf", "source_idx_B_in_eval_bf"


def resolve_existing_path(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    candidates = ", ".join(str(root / name) for name in names)
    raise FileNotFoundError(f"None of these files were found: {candidates}")


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def normalize_embeddings(embeddings: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((embeddings - mean) / std).astype(np.float32)


def load_normalization(root: Path):
    train_bonafide = load_embeddings(root / "train_bonafide_zsem.npy")
    mean = train_bonafide.mean(axis=0, keepdims=True)
    std = train_bonafide.std(axis=0, keepdims=True) + 1e-8
    return mean, std


def load_train_eval_data(root: Path):
    mean, std = load_normalization(root)
    return {
        "train_bonafide": normalize_embeddings(load_embeddings(root / "train_bonafide_zsem.npy"), mean, std),
        "train_morphs": normalize_embeddings(load_embeddings(root / "train_morph_zsem.npy"), mean, std),
        "eval_bonafide": normalize_embeddings(load_embeddings(root / "eval_bonafide_zsem.npy"), mean, std),
        "eval_morphs": normalize_embeddings(load_embeddings(root / "eval_morph_zsem.npy"), mean, std),
    }


def make_loader(morphs, bonafides, metadata_path, batch_size, shuffle, num_workers):
    return DataLoader(
        MorphPairDataset(morphs, bonafides, metadata_path),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def move_batch_to_device(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def random_direction(morph, source_a, source_b):
    mask = torch.rand((morph.shape[0], 1), device=morph.device) > 0.5
    reference = torch.where(mask, source_a, source_b)
    target = torch.where(mask, source_b, source_a)
    return reference, target


class CrossAttentionInverseInterpolationBranch(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(self, morph, reference):
        q = self.query(reference)
        k = self.key(morph)
        v = self.value(morph)
        weights = torch.softmax(q.unsqueeze(2) * k.unsqueeze(1) / math.sqrt(self.dim), dim=-1)
        attended = torch.bmm(weights, v.unsqueeze(-1)).squeeze(-1)
        reference_attended = reference + attended
        return 2.0 * morph - reference_attended


def pair_metrics(prediction, target, morph):
    return {
        "mse": F.mse_loss(prediction, target, reduction="mean"),
        "l1": F.l1_loss(prediction, target, reduction="mean"),
        "cos": F.cosine_similarity(prediction, target, dim=-1).mean(),
        "success": (F.cosine_similarity(prediction, target, dim=-1) > F.cosine_similarity(prediction, morph, dim=-1)).float().mean(),
    }


def average_metric_dicts(first, second):
    return {key: 0.5 * (first[key] + second[key]) for key in first}


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    totals = {"loss": 0.0, "l1": 0.0, "cos": 0.0, "success": 0.0}

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        morph, source_a, source_b = move_batch_to_device(batch, device)
        reference, target = random_direction(morph, source_a, source_b)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(morph, reference)
        metrics = pair_metrics(prediction, target, morph)
        metrics["mse"].backward()
        optimizer.step()
        totals["loss"] += metrics["mse"].item()
        totals["l1"] += metrics["l1"].item()
        totals["cos"] += metrics["cos"].item()
        totals["success"] += metrics["success"].item()

    return {f"train_{key}": value / len(loader) for key, value in totals.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    totals = {"mse": 0.0, "l1": 0.0, "cos": 0.0, "success": 0.0}

    for batch in loader:
        morph, source_a, source_b = move_batch_to_device(batch, device)
        pred_b = model(morph, source_a)
        pred_a = model(morph, source_b)
        metrics_b = pair_metrics(pred_b, source_b, morph)
        metrics_a = pair_metrics(pred_a, source_a, morph)
        metrics = average_metric_dicts(metrics_a, metrics_b)
        for key in totals:
            totals[key] += metrics[key].item()

    return {f"val_{key}": value / len(loader) for key, value in totals.items()}


def train_cross_attention_branch(data_dir: str, run_name: str, batch_size=128, epochs=200, lr=1e-4, weight_decay=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(data_dir)
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_train_eval_data(root)
    train_loader = make_loader(data["train_morphs"], data["train_bonafide"], root / "train_morph_metadata.csv", batch_size, True, 8)
    val_loader = make_loader(data["eval_morphs"], data["eval_bonafide"], root / "eval_morph_metadata.csv", batch_size, False, 4)

    model = CrossAttentionInverseInterpolationBranch().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_mse = float("inf")

    wandb.init(
        project="Face-DM-Conditional",
        name=run_name,
        config={"batch_size": batch_size, "epochs": epochs, "lr": lr, "weight_decay": weight_decay},
    )

    for epoch in range(1, epochs + 1):
        log_values = train_one_epoch(model, train_loader, optimizer, device, epoch)
        log_values.update(validate(model, val_loader, device))
        wandb.log(log_values, step=epoch)
        metrics = " | ".join(f"{name}: {value:.4f}" for name, value in log_values.items())
        print(f"Epoch {epoch:03d} | {metrics}")

        if log_values["val_mse"] < best_val_mse:
            best_val_mse = log_values["val_mse"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    wandb.finish()


def load_eval_records(metadata_path: Path):
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            morph_idx = int(row["embedding_index"])
            col_a, col_b = source_index_columns(row)
            source_idx_a = int(row[col_a])
            source_idx_b = int(row[col_b])
            if source_idx_a >= 0 and source_idx_b >= 0:
                records.append((morph_idx, source_idx_a, source_idx_b))
    return records


def metric_or_default(total, denominator, default):
    return total / denominator if denominator else default


def add_direction_metrics(totals, prefix, prediction, target, morph):
    totals[f"{prefix}_mse"] += F.mse_loss(prediction, target, reduction="sum").item()
    totals[f"{prefix}_l1"] += F.l1_loss(prediction, target, reduction="sum").item()
    totals[f"{prefix}_cos"] += F.cosine_similarity(prediction, target, dim=-1).sum().item()
    totals[f"{prefix}_success"] += (F.cosine_similarity(prediction, target, dim=-1) > F.cosine_similarity(prediction, morph, dim=-1)).float().sum().item()


@torch.no_grad()
def evaluate_cross_attention_branch(data_dir: str, run_name: str, batch_size=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(data_dir)
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / "eval_metrics.txt"

    mean, std = load_normalization(root)
    eval_bonafide = normalize_embeddings(load_embeddings(root / "eval_bonafide_zsem.npy"), mean, std)
    eval_morphs = normalize_embeddings(
        load_embeddings(resolve_existing_path(root, "eval_morph_zsem.npy", "eval_morph_zsem_1000.npy")),
        mean,
        std,
    )
    records = load_eval_records(resolve_existing_path(root, "eval_morph_metadata.csv", "eval_morph_metadata_1000.csv"))

    model = CrossAttentionInverseInterpolationBranch().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    totals = {
        "learned_a_to_b_mse": 0.0,
        "learned_a_to_b_l1": 0.0,
        "learned_a_to_b_cos": 0.0,
        "learned_a_to_b_success": 0.0,
        "learned_b_to_a_mse": 0.0,
        "learned_b_to_a_l1": 0.0,
        "learned_b_to_a_cos": 0.0,
        "learned_b_to_a_success": 0.0,
        "linear_a_to_b_mse": 0.0,
        "linear_a_to_b_l1": 0.0,
        "linear_a_to_b_cos": 0.0,
        "linear_a_to_b_success": 0.0,
        "linear_b_to_a_mse": 0.0,
        "linear_b_to_a_l1": 0.0,
        "linear_b_to_a_cos": 0.0,
        "linear_b_to_a_success": 0.0,
    }

    for start in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
        batch_records = records[start : start + batch_size]
        morph = torch.tensor(np.array([eval_morphs[i] for i, _, _ in batch_records]), device=device)
        source_a = torch.tensor(np.array([eval_bonafide[i] for _, i, _ in batch_records]), device=device)
        source_b = torch.tensor(np.array([eval_bonafide[i] for _, _, i in batch_records]), device=device)

        learned_b = model(morph, source_a)
        learned_a = model(morph, source_b)
        linear_b = 2.0 * morph - source_a
        linear_a = 2.0 * morph - source_b

        add_direction_metrics(totals, "learned_a_to_b", learned_b, source_b, morph)
        add_direction_metrics(totals, "learned_b_to_a", learned_a, source_a, morph)
        add_direction_metrics(totals, "linear_a_to_b", linear_b, source_b, morph)
        add_direction_metrics(totals, "linear_b_to_a", linear_a, source_a, morph)

    n = len(records)
    dim = 512

    def values(prefix):
        return {
            "mse": metric_or_default(totals[f"{prefix}_mse"], n * dim, float("inf")),
            "l1": metric_or_default(totals[f"{prefix}_l1"], n * dim, float("inf")),
            "cos": metric_or_default(totals[f"{prefix}_cos"], n, 0.0),
            "success": metric_or_default(totals[f"{prefix}_success"] * 100, n, 0.0),
        }

    learned_ab = values("learned_a_to_b")
    learned_ba = values("learned_b_to_a")
    linear_ab = values("linear_a_to_b")
    linear_ba = values("linear_b_to_a")

    def mean_pair(first, second, key):
        return 0.5 * (first[key] + second[key])

    results = (
        "--- CROSS-ATTENTION BRANCH ONLY EVALUATION ---\n"
        f"Condition A -> Recover B Learned MSE : {learned_ab['mse']:.6f}\n"
        f"Condition A -> Recover B Learned L1  : {learned_ab['l1']:.6f}\n"
        f"Condition A -> Recover B Learned Cos : {learned_ab['cos']:.6f}\n"
        f"Condition A -> Recover B Learned %S  : {learned_ab['success']:.2f}%\n"
        f"Condition B -> Recover A Learned MSE : {learned_ba['mse']:.6f}\n"
        f"Condition B -> Recover A Learned L1  : {learned_ba['l1']:.6f}\n"
        f"Condition B -> Recover A Learned Cos : {learned_ba['cos']:.6f}\n"
        f"Condition B -> Recover A Learned %S  : {learned_ba['success']:.2f}%\n"
        f"Mean Learned MSE                     : {mean_pair(learned_ab, learned_ba, 'mse'):.6f}\n"
        f"Mean Learned L1                      : {mean_pair(learned_ab, learned_ba, 'l1'):.6f}\n"
        f"Mean Learned Cos                     : {mean_pair(learned_ab, learned_ba, 'cos'):.6f}\n"
        f"Mean Learned %S                      : {mean_pair(learned_ab, learned_ba, 'success'):.2f}%\n"
        "\n"
        "--- ANALYTICAL INVERSE INTERPOLATION BASELINE ---\n"
        f"Condition A -> Recover B Linear MSE  : {linear_ab['mse']:.6f}\n"
        f"Condition A -> Recover B Linear L1   : {linear_ab['l1']:.6f}\n"
        f"Condition A -> Recover B Linear Cos  : {linear_ab['cos']:.6f}\n"
        f"Condition A -> Recover B Linear %S   : {linear_ab['success']:.2f}%\n"
        f"Condition B -> Recover A Linear MSE  : {linear_ba['mse']:.6f}\n"
        f"Condition B -> Recover A Linear L1   : {linear_ba['l1']:.6f}\n"
        f"Condition B -> Recover A Linear Cos  : {linear_ba['cos']:.6f}\n"
        f"Condition B -> Recover A Linear %S   : {linear_ba['success']:.2f}%\n"
        f"Mean Linear MSE                      : {mean_pair(linear_ab, linear_ba, 'mse'):.6f}\n"
        f"Mean Linear L1                       : {mean_pair(linear_ab, linear_ba, 'l1'):.6f}\n"
        f"Mean Linear Cos                      : {mean_pair(linear_ab, linear_ba, 'cos'):.6f}\n"
        f"Mean Linear %S                       : {mean_pair(linear_ab, linear_ba, 'success'):.2f}%\n"
    )

    print(f"\n{results}")
    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(results)


if __name__ == "__main__":
    DATA_DIR = "/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings_v2"
    RUN_NAME = "diffae_cross_attention_branch_only"

    train_cross_attention_branch(DATA_DIR, RUN_NAME)
    evaluate_cross_attention_branch(DATA_DIR, RUN_NAME)

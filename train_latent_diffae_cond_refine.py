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


class BReferenceATargetMorphDataset(Dataset):
    """
    Conditional morph dataset with a single fixed direction per morph pair.

    Source B is always used as the reference/condition and source A is always
    used as the target. The model never sees the reverse A -> B case for the
    same morph embedding.
    """

    def __init__(
        self,
        morph_embeddings: np.ndarray,
        bonafide_embeddings: np.ndarray,
        metadata_path: Path,
    ):
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

                if source_idx_a < 0 or source_idx_b < 0:
                    continue

                # Fixed direction for every pair:
                # reference/condition = B, target = A.
                records.append((morph_idx, source_idx_b, source_idx_a))

        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        morph_idx, condition_idx, target_idx = self.records[index]
        morph = self.morph_embeddings[morph_idx]
        condition = self.bonafide_embeddings[condition_idx]
        target = self.bonafide_embeddings[target_idx]
        return torch.from_numpy(morph), torch.from_numpy(condition), torch.from_numpy(target)


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


def load_normalized_data(root: Path):
    train_bonafide = load_embeddings(root / "train_bonafide_zsem.npy")
    train_morphs = load_embeddings(root / "train_morph_zsem.npy")
    mean = train_bonafide.mean(axis=0, keepdims=True)
    std = train_bonafide.std(axis=0, keepdims=True) + 1e-8

    return {
        "train_bonafide": normalize_embeddings(train_bonafide, mean, std),
        "train_morphs": normalize_embeddings(train_morphs, mean, std),
        "eval_bonafide": normalize_embeddings(load_embeddings(root / "eval_bonafide_zsem.npy"), mean, std),
        "eval_morphs": normalize_embeddings(load_embeddings(root / "eval_morph_zsem.npy"), mean, std),
        "mean": mean,
        "std": std,
    }


def make_loader(morphs, bonafides, metadata_path, batch_size, shuffle, num_workers):
    dataset = BReferenceATargetMorphDataset(
        morphs,
        bonafides,
        metadata_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=time.device) * -scale)
        embeddings = time[:, None] * frequencies[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class AdaLNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, cond_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)
        self.activation = nn.SiLU()

    def forward(self, x, cond):
        h = self.linear(x)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        return self.activation(self.norm(h) * (1 + scale) + shift)


class ColdDemorphNet(nn.Module):
    def __init__(self, x_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        self.blocks = nn.ModuleList(
            [AdaLNBlock(x_dim, hidden_dim, time_emb_dim)]
            + [AdaLNBlock(hidden_dim + x_dim, hidden_dim, time_emb_dim) for _ in range(num_layers - 1)]
        )
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t, t):
        t_emb = self.time_mlp(t)
        h = x_t
        for index, block in enumerate(self.blocks):
            h = block(h if index == 0 else torch.cat((h, x_t), dim=-1), t_emb)
        return self.final_linear(h)


class MathBaselineColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=10):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def coarse_prediction(self, morph, condition):
        return 2.0 * morph - condition

    def degrade(self, target, coarse, t):
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        return torch.sqrt(gamma) * coarse + torch.sqrt(1 - gamma) * target

    def compute_loss(self, morph, condition, target, t=None):
        batch_size = morph.shape[0]
        if t is None:
            t = torch.randint(1, self.num_timesteps + 1, (batch_size,), device=morph.device).long()
        coarse = self.coarse_prediction(morph, condition)
        x_t = self.degrade(target, coarse, t)
        prediction = self.model(x_t, t)
        return F.l1_loss(prediction, target)

    @torch.no_grad()
    def sample_loop(self, morph, condition):
        batch_size = morph.shape[0]
        coarse = self.coarse_prediction(morph, condition)
        x_t = coarse.clone()

        for t in range(self.num_timesteps, 0, -1):
            t_batch = torch.full((batch_size,), t, device=morph.device, dtype=torch.long)
            prediction = self.model(x_t, t_batch)
            deg_t = self.degrade(prediction, coarse, t_batch)
            deg_prev = self.degrade(prediction, coarse, t_batch - 1)
            x_t = x_t - deg_t + deg_prev

        return x_t


def move_batch_to_device(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def train_one_epoch(diffusion, loader, optimizer, device, epoch):
    diffusion.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        morph, condition, target = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.compute_loss(morph, condition, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_diffusion_loss(diffusion, loader, device):
    diffusion.eval()
    total_loss = 0.0

    for batch in loader:
        morph, condition, target = move_batch_to_device(batch, device)
        total_loss += diffusion.compute_loss(morph, condition, target).item()

    return total_loss / len(loader)


@torch.no_grad()
def validate_sample_loss(diffusion, loader, device):
    diffusion.eval()
    morph, condition, target = move_batch_to_device(next(iter(loader)), device)
    prediction = diffusion.sample_loop(morph, condition)
    return F.l1_loss(prediction, target).item()


def train_math_diffusion_b_reference_a_target(
    data_dir: str,
    run_name: str,
    num_timesteps=10,
    batch_size=256,
    epochs=200,
    sample_eval_interval=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(data_dir)
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_normalized_data(root)
    train_loader = make_loader(
        data["train_morphs"],
        data["train_bonafide"],
        root / "train_morph_metadata.csv",
        batch_size,
        True,
        8,
    )
    val_loader = make_loader(
        data["eval_morphs"],
        data["eval_bonafide"],
        root / "eval_morph_metadata.csv",
        batch_size,
        False,
        4,
    )

    net = ColdDemorphNet(num_layers=10).to(device)
    diffusion = MathBaselineColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    best_val_loss = float("inf")

    wandb.init(
        project="Face-DM-Conditional",
        name=run_name,
        config={
            "timesteps": num_timesteps,
            "batch_size": batch_size,
            "epochs": epochs,
            "sample_eval_interval": sample_eval_interval,
            "direction_policy": "B_as_reference_A_as_target",
        },
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(diffusion, train_loader, optimizer, device, epoch)
        val_loss = validate_diffusion_loss(diffusion, val_loader, device)
        log_values = {"train_loss": train_loss, "val_loss": val_loss}

        if epoch % sample_eval_interval == 0:
            log_values["val_sample_l1"] = validate_sample_loss(diffusion, val_loader, device)

        wandb.log(log_values, step=epoch)
        metrics = " | ".join(f"{name}: {value:.4f}" for name, value in log_values.items())
        print(f"Epoch {epoch:03d} | {metrics}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), ckpt_dir / "best.pt")

    wandb.finish()


def load_b_reference_a_target_records(metadata_path: Path):
    return BReferenceATargetMorphDataset.load_records(metadata_path)


def metric_or_default(total, denominator, default):
    return total / denominator if denominator else default


@torch.no_grad()
def evaluate_batch(diffusion, net, morph, condition, num_timesteps, mode):
    if mode == "iterative":
        return diffusion.sample_loop(morph, condition)

    t_max = torch.full((morph.shape[0],), num_timesteps, device=morph.device).long()
    coarse = diffusion.coarse_prediction(morph, condition)
    return net(coarse, t_max)


def evaluate_math_diffusion_b_reference_a_target(
    data_dir: str,
    run_name: str,
    num_timesteps: int = 10,
    mode: str = "iterative",
):
    if mode not in {"iterative", "one_shot"}:
        raise ValueError("mode must be 'iterative' or 'one_shot'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(data_dir)
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / f"eval_metrics_b_reference_a_target_{mode}.txt"

    train_bonafide = load_embeddings(root / "train_bonafide_zsem.npy")
    mean = train_bonafide.mean(axis=0, keepdims=True)
    std = train_bonafide.std(axis=0, keepdims=True) + 1e-8
    eval_bonafide = normalize_embeddings(load_embeddings(root / "eval_bonafide_zsem.npy"), mean, std)
    eval_morphs = normalize_embeddings(
        load_embeddings(resolve_existing_path(root, "eval_morph_zsem.npy", "eval_morph_zsem_1000.npy")),
        mean,
        std,
    )
    records = load_b_reference_a_target_records(
        resolve_existing_path(root, "eval_morph_metadata.csv", "eval_morph_metadata_1000.csv")
    )

    net = ColdDemorphNet(num_layers=10).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    diffusion = MathBaselineColdDemorph(net, num_timesteps=num_timesteps).to(device)

    totals = {
        "l1": 0.0,
        "cos": 0.0,
        "success": 0.0,
    }
    batch_size = 500

    for start in tqdm(range(0, len(records), batch_size), desc=f"Evaluating ({mode.upper()})"):
        batch_records = records[start : start + batch_size]
        morph = torch.tensor(np.array([eval_morphs[i] for i, _, _ in batch_records]), device=device)
        condition = torch.tensor(np.array([eval_bonafide[i] for _, i, _ in batch_records]), device=device)
        target = torch.tensor(np.array([eval_bonafide[i] for _, _, i in batch_records]), device=device)

        prediction = evaluate_batch(diffusion, net, morph, condition, num_timesteps, mode)

        totals["l1"] += F.l1_loss(prediction, target, reduction="sum").item()
        totals["cos"] += F.cosine_similarity(prediction, target, dim=-1).sum().item()
        totals["success"] += (
            F.cosine_similarity(prediction, target, dim=-1)
            > F.cosine_similarity(prediction, morph, dim=-1)
        ).float().sum().item()

    n = len(records)
    embedding_dim = eval_bonafide.shape[1]
    avg_l1 = metric_or_default(totals["l1"], n * embedding_dim, float("inf"))
    avg_cos = metric_or_default(totals["cos"], n, 0.0)
    pct_success = metric_or_default(totals["success"] * 100, n, 0.0)

    results = (
        f"--- MATH BASELINE B-REFERENCE/A-TARGET CONDITIONAL EVALUATION ({mode.upper()}) ---\n"
        f"Direction                       : reference=B, target=A\n"
        f"Number of evaluated morphs      : {n}\n"
        f"Conditional L1 Distance         : {avg_l1:.6f}\n"
        f"Conditional Cosine Sim          : {avg_cos:.6f}\n"
        f"Conditional %S                  : {pct_success:.2f}%\n"
    )
    print(f"\n{results}")
    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(results)


if __name__ == "__main__":
    DATA_DIR = "/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings_v2"
    RUN_NAME = "diffae_conditional_refiner_varPres_Bref_Atarget"
    TIMESTEPS = 100

    train_math_diffusion_b_reference_a_target(
        DATA_DIR,
        RUN_NAME,
        num_timesteps=TIMESTEPS,
    )
    evaluate_math_diffusion_b_reference_a_target(
        DATA_DIR,
        RUN_NAME,
        num_timesteps=TIMESTEPS,
        mode="one_shot",
    )
    evaluate_math_diffusion_b_reference_a_target(
        DATA_DIR,
        RUN_NAME,
        num_timesteps=TIMESTEPS,
        mode="iterative",
    )

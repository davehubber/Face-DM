import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MIPGANIICriminalFromMorphDataset(Dataset):
    """
    Row-aligned MAD22 MIPGAN_II dataset.

    For each row i:
      morph_embeddings[i]      = z_sem of the MIPGAN_II morph image
      criminal_embeddings[i]   = z_cri, the first bona fide source in the filename
      accomplice_embeddings[i] = z_acc, the second bona fide source in the filename

    The model learns:
      input state  = degraded/interpolated state between criminal and morph
      condition    = accomplice embedding
      target       = clean criminal embedding
    """

    def __init__(
        self,
        morph_embeddings: np.ndarray,
        criminal_embeddings: np.ndarray,
        accomplice_embeddings: np.ndarray,
    ):
        if len(morph_embeddings) != len(criminal_embeddings):
            raise ValueError("morph_embeddings and criminal_embeddings must have the same length.")
        if len(morph_embeddings) != len(accomplice_embeddings):
            raise ValueError("morph_embeddings and accomplice_embeddings must have the same length.")

        self.morph_embeddings = morph_embeddings
        self.criminal_embeddings = criminal_embeddings
        self.accomplice_embeddings = accomplice_embeddings

    def __len__(self):
        return len(self.morph_embeddings)

    def __getitem__(self, index):
        morph = self.morph_embeddings[index]
        accomplice = self.accomplice_embeddings[index]
        criminal = self.criminal_embeddings[index]
        return torch.from_numpy(morph), torch.from_numpy(accomplice), torch.from_numpy(criminal)


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


def load_manifest(root: Path):
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_normalized_mipgan_ii_dataset(root: Path):
    train_morphs = load_embeddings(root / "train_morph_zsem.npy")
    train_criminals = load_embeddings(root / "train_z_cri.npy")
    train_accomplices = load_embeddings(root / "train_z_acc.npy")

    test_morphs = load_embeddings(resolve_existing_path(root, "test_morph_zsem.npy", "eval_morph_zsem.npy"))
    test_criminals = load_embeddings(resolve_existing_path(root, "test_z_cri.npy", "eval_z_cri.npy"))
    test_accomplices = load_embeddings(resolve_existing_path(root, "test_z_acc.npy", "eval_z_acc.npy"))

    # Keep the stored dataset raw. Training normalization is computed only from
    # clean training source embeddings, then reused for train morphs and test data.
    train_sources = np.concatenate([train_criminals, train_accomplices], axis=0)
    mean = train_sources.mean(axis=0, keepdims=True)
    std = train_sources.std(axis=0, keepdims=True) + 1e-8

    return {
        "train_morphs": normalize_embeddings(train_morphs, mean, std),
        "train_criminals": normalize_embeddings(train_criminals, mean, std),
        "train_accomplices": normalize_embeddings(train_accomplices, mean, std),
        "test_morphs": normalize_embeddings(test_morphs, mean, std),
        "test_criminals": normalize_embeddings(test_criminals, mean, std),
        "test_accomplices": normalize_embeddings(test_accomplices, mean, std),
        "mean": mean,
        "std": std,
    }


def make_loader(morphs, criminals, accomplices, batch_size, shuffle, num_workers):
    dataset = MIPGANIICriminalFromMorphDataset(
        morph_embeddings=morphs,
        criminal_embeddings=criminals,
        accomplice_embeddings=accomplices,
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


class AccompliceConditionedColdDemorphNet(nn.Module):
    """
    Predicts the clean criminal embedding from a degraded state x_t.

    The network receives x_t as the main input and is conditioned on:
      1. the timestep t;
      2. the accomplice embedding.

    The accomplice condition is injected into every block through AdaLN.
    """

    def __init__(
        self,
        x_dim=512,
        hidden_dim=2048,
        num_layers=10,
        time_emb_dim=512,
        cond_dim=512,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, cond_dim),
        )
        self.accomplice_mlp = nn.Sequential(
            nn.LayerNorm(x_dim),
            nn.Linear(x_dim, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

        self.blocks = nn.ModuleList(
            [AdaLNBlock(x_dim, hidden_dim, cond_dim)]
            + [AdaLNBlock(hidden_dim + x_dim, hidden_dim, cond_dim) for _ in range(num_layers - 1)]
        )
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t, accomplice, t):
        time_cond = self.time_mlp(t)
        accomplice_cond = self.accomplice_mlp(accomplice)
        cond = self.cond_mlp(torch.cat((time_cond, accomplice_cond), dim=-1))

        h = x_t
        for index, block in enumerate(self.blocks):
            h = block(h if index == 0 else torch.cat((h, x_t), dim=-1), cond)
        return self.final_linear(h)


class MorphAnchoredColdDemorph(nn.Module):
    """
    Cold diffusion process anchored at the observed morph embedding.

    Forward degradation:
      x_t = sqrt(1 - gamma_t) * criminal + sqrt(gamma_t) * morph
      gamma_t = t / T

    Therefore:
      x_0 = criminal
      x_T = morph

    Sampling starts from x_T = morph and uses TACOS-style correction:
      x_{t-1} = x_t - D_t(predicted_criminal, morph) + D_{t-1}(predicted_criminal, morph)
    """

    def __init__(self, model, num_timesteps=10):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, criminal, morph, t):
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        return torch.sqrt(1 - gamma) * criminal + torch.sqrt(gamma) * morph

    def compute_loss(self, morph, accomplice, criminal, t=None):
        batch_size = morph.shape[0]
        if t is None:
            t = torch.randint(1, self.num_timesteps + 1, (batch_size,), device=morph.device).long()

        x_t = self.degrade(criminal, morph, t)
        criminal_prediction = self.model(x_t, accomplice, t)
        return F.l1_loss(criminal_prediction, criminal)

    @torch.no_grad()
    def sample_loop(self, morph, accomplice):
        batch_size = morph.shape[0]
        x_t = morph.clone()

        for t in range(self.num_timesteps, 0, -1):
            t_batch = torch.full((batch_size,), t, device=morph.device, dtype=torch.long)
            criminal_prediction = self.model(x_t, accomplice, t_batch)

            deg_t = self.degrade(criminal_prediction, morph, t_batch)
            deg_prev = self.degrade(criminal_prediction, morph, t_batch - 1)
            x_t = x_t - deg_t + deg_prev

        return x_t

    @torch.no_grad()
    def sample_one_shot(self, morph, accomplice):
        batch_size = morph.shape[0]
        t_max = torch.full((batch_size,), self.num_timesteps, device=morph.device, dtype=torch.long)
        return self.model(morph, accomplice, t_max)


def move_batch_to_device(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def train_one_epoch(diffusion, loader, optimizer, device, epoch, grad_clip):
    diffusion.train()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        morph, accomplice, criminal = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.compute_loss(morph, accomplice, criminal)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), grad_clip)
        optimizer.step()

        batch_size = morph.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate_diffusion_loss(diffusion, loader, device):
    diffusion.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        morph, accomplice, criminal = move_batch_to_device(batch, device)
        loss = diffusion.compute_loss(morph, accomplice, criminal)
        batch_size = morph.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def validate_sample_l1(diffusion, loader, device, mode):
    diffusion.eval()
    total_l1 = 0.0
    total_values = 0

    for batch in loader:
        morph, accomplice, criminal = move_batch_to_device(batch, device)
        if mode == "iterative":
            criminal_prediction = diffusion.sample_loop(morph, accomplice)
        elif mode == "one_shot":
            criminal_prediction = diffusion.sample_one_shot(morph, accomplice)
        else:
            raise ValueError("mode must be 'iterative' or 'one_shot'")

        total_l1 += F.l1_loss(criminal_prediction, criminal, reduction="sum").item()
        total_values += criminal.numel()

    return total_l1 / total_values


def train_mipgan_ii_morph_anchor_tacos_model(
    data_dir: str,
    run_name: str,
    num_timesteps=10,
    batch_size=256,
    epochs=200,
    sample_eval_interval=10,
    num_workers_train=8,
    num_workers_test=4,
    lr=1e-4,
    weight_decay=0.01,
    grad_clip=1.0,
    use_wandb=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(data_dir)
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MAD22 MIPGAN_II dataset arrays...")
    manifest = load_manifest(root)
    data = load_normalized_mipgan_ii_dataset(root)

    train_loader = make_loader(
        data["train_morphs"],
        data["train_criminals"],
        data["train_accomplices"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_train,
    )
    test_loader = make_loader(
        data["test_morphs"],
        data["test_criminals"],
        data["test_accomplices"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_test,
    )

    net = AccompliceConditionedColdDemorphNet(num_layers=10).to(device)
    diffusion = MorphAnchoredColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_test_loss = float("inf")

    if use_wandb:
        wandb.init(
            project="Face-DM-Conditional",
            name=run_name,
            config={
                "timesteps": num_timesteps,
                "batch_size": batch_size,
                "epochs": epochs,
                "sample_eval_interval": sample_eval_interval,
                "lr": lr,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "data_setup": "MAD22_MIPGAN_II_80_20_morph_split",
                "method": manifest.get("method", "MIPGAN_II"),
                "train_ratio": manifest.get("train_ratio", 0.80),
                "test_ratio": manifest.get("test_ratio", 0.20),
                "direction_policy": "morph_plus_accomplice_to_criminal",
                "degradation": "sqrt_interpolation_between_criminal_and_morph",
                "sampling": "starts_from_morph_with_tacos_redegradation_correction",
                "conditioning": "accomplice_embedding_adaln",
                "run_name": run_name,
            },
        )

    print("Held-out split is the 20% MIPGAN_II morph split.")
    print(f"Train morphs: {len(train_loader.dataset)}")
    print(f"Test morphs : {len(test_loader.dataset)}")
    print("Forward degradation: criminal -> morph")
    print("Sampling: starts from morph and moves toward criminal with TACOS correction")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(diffusion, train_loader, optimizer, device, epoch, grad_clip)
        test_diffusion_l1 = validate_diffusion_loss(diffusion, test_loader, device)

        log_values = {
            "train_loss": train_loss,
            "test_diffusion_l1": test_diffusion_l1,
        }

        if epoch % sample_eval_interval == 0:
            log_values["test_one_shot_l1"] = validate_sample_l1(diffusion, test_loader, device, mode="one_shot")
            log_values["test_iterative_tacos_l1"] = validate_sample_l1(diffusion, test_loader, device, mode="iterative")

        if use_wandb:
            wandb.log(log_values, step=epoch)

        metrics = " | ".join(f"{name}: {value:.4f}" for name, value in log_values.items())
        print(f"Epoch {epoch:03d} | {metrics}")

        if test_diffusion_l1 < best_test_loss:
            best_test_loss = test_diffusion_l1
            torch.save(net.state_dict(), ckpt_dir / "best.pt")
            print("  Saved new best checkpoint.")

    if use_wandb:
        wandb.finish()


def metric_or_default(total, denominator, default):
    return total / denominator if denominator else default


@torch.no_grad()
def evaluate_loader(diffusion, loader, device, mode):
    diffusion.eval()
    totals = {
        "l1": 0.0,
        "cos": 0.0,
        "success_pred_closer_than_morph": 0.0,
        "success_improves_over_morph": 0.0,
        "cos_gain_over_morph": 0.0,
    }
    n = 0
    embedding_dim = None

    for batch in loader:
        morph, accomplice, criminal = move_batch_to_device(batch, device)

        if mode == "iterative":
            criminal_prediction = diffusion.sample_loop(morph, accomplice)
        elif mode == "one_shot":
            criminal_prediction = diffusion.sample_one_shot(morph, accomplice)
        else:
            raise ValueError("mode must be 'iterative' or 'one_shot'")

        if embedding_dim is None:
            embedding_dim = criminal.shape[1]

        pred_criminal_cos = F.cosine_similarity(criminal_prediction, criminal, dim=-1)
        pred_morph_cos = F.cosine_similarity(criminal_prediction, morph, dim=-1)
        morph_criminal_cos = F.cosine_similarity(morph, criminal, dim=-1)

        totals["l1"] += F.l1_loss(criminal_prediction, criminal, reduction="sum").item()
        totals["cos"] += pred_criminal_cos.sum().item()
        totals["success_pred_closer_than_morph"] += (pred_criminal_cos > pred_morph_cos).float().sum().item()
        totals["success_improves_over_morph"] += (pred_criminal_cos > morph_criminal_cos).float().sum().item()
        totals["cos_gain_over_morph"] += (pred_criminal_cos - morph_criminal_cos).sum().item()
        n += criminal.shape[0]

    if embedding_dim is None:
        raise ValueError("Cannot evaluate an empty loader.")

    return {
        "n": n,
        "l1": metric_or_default(totals["l1"], n * embedding_dim, float("inf")),
        "cos": metric_or_default(totals["cos"], n, 0.0),
        "success_pred_closer_than_morph": metric_or_default(totals["success_pred_closer_than_morph"] * 100, n, 0.0),
        "success_improves_over_morph": metric_or_default(totals["success_improves_over_morph"] * 100, n, 0.0),
        "cos_gain_over_morph": metric_or_default(totals["cos_gain_over_morph"], n, 0.0),
    }


def evaluate_mipgan_ii_morph_anchor_tacos_model(
    data_dir: str,
    run_name: str,
    num_timesteps: int = 10,
    mode: str = "iterative",
    batch_size: int = 500,
    num_workers: int = 4,
):
    if mode not in {"iterative", "one_shot"}:
        raise ValueError("mode must be 'iterative' or 'one_shot'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(data_dir)
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / f"test_metrics_morph_anchor_tacos_{mode}.txt"

    manifest = load_manifest(root)
    method = manifest.get("method", "MIPGAN_II")
    data = load_normalized_mipgan_ii_dataset(root)
    test_loader = make_loader(
        data["test_morphs"],
        data["test_criminals"],
        data["test_accomplices"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    net = AccompliceConditionedColdDemorphNet(num_layers=10).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    diffusion = MorphAnchoredColdDemorph(net, num_timesteps=num_timesteps).to(device)

    metrics = evaluate_loader(diffusion, test_loader, device, mode)
    results = (
        f"--- MIPGAN_II MORPH-ANCHORED TACOS EVALUATION ({mode.upper()}) ---\n"
        f"Dataset                                      : MAD22 {method}\n"
        f"Split                                        : held-out 20% morph split\n"
        f"Task                                         : morph + accomplice -> criminal\n"
        f"Degradation                                  : criminal-to-morph interpolation\n"
        f"Sampling                                     : starts from morph with TACOS correction\n"
        f"Number of evaluated morphs                   : {metrics['n']}\n"
        f"Criminal L1 Distance                         : {metrics['l1']:.6f}\n"
        f"Criminal Cosine Sim                          : {metrics['cos']:.6f}\n"
        f"Criminal %S, pred closer than morph          : {metrics['success_pred_closer_than_morph']:.2f}%\n"
        f"Criminal % improved over original morph      : {metrics['success_improves_over_morph']:.2f}%\n"
        f"Mean cosine gain over original morph         : {metrics['cos_gain_over_morph']:.6f}\n"
    )

    print("\n" + results)
    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MIPGAN_II morph-anchored cold diffusion de-morphing model with accomplice conditioning."
    )
    parser.add_argument("--data-dir", type=str, default="mipgan_ii_dataset")
    parser.add_argument("--run-name", type=str, default="final_model_v2")
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--sample-eval-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers-train", type=int, default=8)
    parser.add_argument("--num-workers-test", type=int, default=4)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.skip_train:
        train_mipgan_ii_morph_anchor_tacos_model(
            data_dir=args.data_dir,
            run_name=args.run_name,
            num_timesteps=args.timesteps,
            batch_size=args.batch_size,
            epochs=args.epochs,
            sample_eval_interval=args.sample_eval_interval,
            num_workers_train=args.num_workers_train,
            num_workers_test=args.num_workers_test,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            use_wandb=not args.no_wandb,
        )

    if not args.skip_eval:
        evaluate_mipgan_ii_morph_anchor_tacos_model(
            data_dir=args.data_dir,
            run_name=args.run_name,
            num_timesteps=args.timesteps,
            mode="one_shot",
            batch_size=500,
            num_workers=args.num_workers_test,
        )
        evaluate_mipgan_ii_morph_anchor_tacos_model(
            data_dir=args.data_dir,
            run_name=args.run_name,
            num_timesteps=args.timesteps,
            mode="iterative",
            batch_size=500,
            num_workers=args.num_workers_test,
        )

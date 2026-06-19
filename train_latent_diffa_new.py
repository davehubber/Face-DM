import math
from pathlib import Path
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
from tqdm import tqdm

# ==========================================
# 1. Dataset & Data Loading
# ==========================================
def resolve_existing_path(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of these files were found: " + ", ".join(str(root / name) for name in names)
    )


class UnconditionalMorphPairDataset(Dataset):
    """
    Loads real morph embeddings and their two bona fide source embeddings.
    No condition is returned. The model only receives the concatenated two-branch state.
    """
    def __init__(self, morph_embs: np.ndarray, bf_embs: np.ndarray, metadata_path: Path):
        self.morph_embs = morph_embs
        self.bf_embs = bf_embs
        self.records = []
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx_m = int(row['embedding_index'])
                
                col_a = 'source_idx_A_in_train_bf' if 'source_idx_A_in_train_bf' in row else 'source_idx_A_in_eval_bf'
                col_b = 'source_idx_B_in_train_bf' if 'source_idx_B_in_train_bf' in row else 'source_idx_B_in_eval_bf'
                
                idx_a = int(row[col_a])
                idx_b = int(row[col_b])
                
                if idx_a >= 0 and idx_b >= 0:
                    self.records.append((idx_m, idx_a, idx_b))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        idx_m, idx_a, idx_b = self.records[idx]
        
        M = self.morph_embs[idx_m]
        A = self.bf_embs[idx_a]
        B = self.bf_embs[idx_b]

        return (
            torch.tensor(M, dtype=torch.float32),
            torch.tensor(A, dtype=torch.float32),
            torch.tensor(B, dtype=torch.float32)
        )


# ==========================================
# 2. Network Architecture
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaLNBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.silu = nn.SiLU()
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x, cond):
        h = self.linear(x)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        return self.silu(self.norm(h) * (1 + scale) + shift)


class UnconditionalColdDemorphNet(nn.Module):
    def __init__(self, x_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
        super().__init__()
        self.x_dim = x_dim
        self.state_dim = x_dim * 2

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(self.state_dim, hidden_dim, time_emb_dim))
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + self.state_dim, hidden_dim, time_emb_dim))

        self.final_linear = nn.Linear(hidden_dim, self.state_dim)

    def forward(self, x_t_pair, t):
        t_emb = self.time_mlp(t)
        h = x_t_pair

        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, t_emb)
            else:
                h = block(torch.cat([h, x_t_pair], dim=-1), t_emb)
                
        return self.final_linear(h)


# ==========================================
# 3. Unconditional Two-Branch Cold Diffusion
# ==========================================
class UnconditionalPairColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=300):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade_one(self, M, clean, t):
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        return gamma * M + (1 - gamma) * clean

    def degrade_pair(self, M, clean_pair, t):
        clean_1, clean_2 = clean_pair.chunk(2, dim=-1)
        x_1 = self.degrade_one(M, clean_1, t)
        x_2 = self.degrade_one(M, clean_2, t)
        return torch.cat([x_1, x_2], dim=-1)

    def make_state(self, M, A, B, t):
        x_a = self.degrade_one(M, A, t)
        x_b = self.degrade_one(M, B, t)
        return torch.cat([x_a, x_b], dim=-1)

    def sample_training_timesteps(self, batch_size, device):
        return torch.randint(1, self.num_timesteps + 1, (batch_size,), device=device).long()

    def permutation_invariant_l1(self, pred_pair, A, B):
        pred_1, pred_2 = pred_pair.chunk(2, dim=-1)

        loss_ab = 0.5 * (
            torch.abs(pred_1 - A).mean(dim=-1) +
            torch.abs(pred_2 - B).mean(dim=-1)
        )
        loss_ba = 0.5 * (
            torch.abs(pred_1 - B).mean(dim=-1) +
            torch.abs(pred_2 - A).mean(dim=-1)
        )

        return torch.minimum(loss_ab, loss_ba).mean()

    def compute_loss(self, M, A, B, t=None):
        b = M.shape[0]
        if t is None:
            t = self.sample_training_timesteps(b, M.device)

        x_t_pair = self.make_state(M, A, B, t)
        pred_pair = self.model(x_t_pair, t)

        return self.permutation_invariant_l1(pred_pair, A, B)

    def cosine_order_distance(self, left_1, left_2, right_1, right_2):
        d_11 = 1.0 - F.cosine_similarity(left_1, right_1, dim=-1)
        d_22 = 1.0 - F.cosine_similarity(left_2, right_2, dim=-1)
        return d_11 + d_22

    def align_clean_pair_to_state(self, M, pred_pair, state_t, t):
        """
        Aligns the order of the current clean predictions with the order of the current
        two-branch state before applying the full cold-diffusion correction.
        """
        pred_1, pred_2 = pred_pair.chunk(2, dim=-1)
        state_1, state_2 = state_t.chunk(2, dim=-1)

        deg_1 = self.degrade_one(M, pred_1, t)
        deg_2 = self.degrade_one(M, pred_2, t)

        same_distance = self.cosine_order_distance(deg_1, deg_2, state_1, state_2)
        swap_distance = self.cosine_order_distance(deg_2, deg_1, state_1, state_2)
        swap_mask = (swap_distance < same_distance).view(-1, 1)

        aligned_1 = torch.where(swap_mask, pred_2, pred_1)
        aligned_2 = torch.where(swap_mask, pred_1, pred_2)

        return torch.cat([aligned_1, aligned_2], dim=-1)

    def align_pair_to_targets(self, pred_pair, A, B):
        pred_1, pred_2 = pred_pair.chunk(2, dim=-1)

        loss_ab = 0.5 * (
            torch.abs(pred_1 - A).mean(dim=-1) +
            torch.abs(pred_2 - B).mean(dim=-1)
        )
        loss_ba = 0.5 * (
            torch.abs(pred_1 - B).mean(dim=-1) +
            torch.abs(pred_2 - A).mean(dim=-1)
        )
        swap_mask = (loss_ba < loss_ab).view(-1, 1)

        aligned_1 = torch.where(swap_mask, pred_2, pred_1)
        aligned_2 = torch.where(swap_mask, pred_1, pred_2)

        return aligned_1, aligned_2

    @torch.no_grad()
    def sample_loop(self, M):
        """
        Starts from [M, M]. At each reverse step, predicts two clean embeddings,
        aligns their order to the current two-branch state, and then applies the full
        cold-diffusion correction:

            x_{t-1} = x_t - D_t(pred_x0) + D_{t-1}(pred_x0)
        """
        device = M.device
        b = M.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        state_t = torch.cat([M, M], dim=-1)

        for t in tqdm(timesteps, desc="Sampling", leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)

            pred_pair = self.model(state_t, t_batch)
            pred_pair = self.align_clean_pair_to_state(M, pred_pair, state_t, t_batch)

            deg_t = self.degrade_pair(M, pred_pair, t_batch)
            deg_t_prev = self.degrade_pair(M, pred_pair, t_prev_batch)

            state_t = state_t - deg_t + deg_t_prev

        return state_t


# ==========================================
# 4. Early Stopping
# ==========================================
class EarlyStopping:
    def __init__(self, patience: int = 50, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# ==========================================
# 5. Shared Data Loading
# ==========================================
def load_normalized_datasets(data_dir: str):
    root = Path(data_dir)

    train_bf = np.load(root / "train_bonafide_zsem.npy").astype(np.float32)
    train_morphs = np.load(root / "train_morph_zsem.npy").astype(np.float32)

    global_mean = train_bf.mean(axis=0, keepdims=True)
    global_std = train_bf.std(axis=0, keepdims=True) + 1e-8

    train_bf = (train_bf - global_mean) / global_std
    train_morphs = (train_morphs - global_mean) / global_std

    eval_bf = np.load(root / "eval_bonafide_zsem.npy").astype(np.float32)
    eval_morphs = np.load(resolve_existing_path(root, "eval_morph_zsem.npy", "eval_morph_zsem_1000.npy")).astype(np.float32)

    eval_bf = (eval_bf - global_mean) / global_std
    eval_morphs = (eval_morphs - global_mean) / global_std

    train_metadata = root / "train_morph_metadata.csv"
    eval_metadata = resolve_existing_path(root, "eval_morph_metadata.csv", "eval_morph_metadata_1000.csv")

    return train_bf, train_morphs, train_metadata, eval_bf, eval_morphs, eval_metadata


# ==========================================
# 6. Training Loop
# ==========================================
def train_unconditional_demorph(
    data_dir: str,
    run_name: str,
    num_timesteps: int = 300,
    epochs: int = 150,
    patience: int = 50,
    batch_size: int = 8192
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets into memory...")
    train_bf, train_morphs, train_metadata, eval_bf, eval_morphs, eval_metadata = load_normalized_datasets(data_dir)

    train_dataset = UnconditionalMorphPairDataset(train_morphs, train_bf, train_metadata)
    eval_dataset = UnconditionalMorphPairDataset(eval_morphs, eval_bf, eval_metadata)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(eval_dataset, batch_size=1000, shuffle=False, num_workers=4)

    net = UnconditionalColdDemorphNet(x_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = UnconditionalPairColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    early_stopper = EarlyStopping(patience=patience)

    wandb.init(
        project="Face-DM-Unconditional",
        name=run_name,
        dir=str(exp_dir),
        config={
            "batch_size": batch_size,
            "num_layers": 10,
            "timesteps": num_timesteps,
            "state_dim": 1024,
            "loss": "permutation_invariant_l1",
            "sampling": "full_step_tacos_order_alignment"
        }
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0

        for batch_M, batch_A, batch_B in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            batch_M = batch_M.to(device)
            batch_A = batch_A.to(device)
            batch_B = batch_B.to(device)

            optimizer.zero_grad()
            loss = diffusion.compute_loss(batch_M, batch_A, batch_B)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        net.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_M, batch_A, batch_B in val_loader:
                batch_M = batch_M.to(device)
                batch_A = batch_A.to(device)
                batch_B = batch_B.to(device)
                batch_size_now = batch_M.shape[0]

                t = diffusion.sample_training_timesteps(batch_size_now, device)
                loss = diffusion.compute_loss(batch_M, batch_A, batch_B, t=t)

                val_loss += loss.item() * batch_size_now
                val_count += batch_size_now

        avg_val_loss = val_loss / val_count

        sample_l1 = 0.0
        sample_cos = 0.0
        if (epoch + 1) % 25 == 0 or (epoch + 1) == epochs:
            with torch.no_grad():
                b_M, b_A, b_B = next(iter(val_loader))
                b_M, b_A, b_B = b_M.to(device), b_A.to(device), b_B.to(device)
                pred_pair = diffusion.sample_loop(b_M)
                pred_A, pred_B = diffusion.align_pair_to_targets(pred_pair, b_A, b_B)
                sample_l1 = 0.5 * (
                    F.l1_loss(pred_A, b_A).item() +
                    F.l1_loss(pred_B, b_B).item()
                )
                sample_cos = 0.5 * (
                    F.cosine_similarity(pred_A, b_A, dim=-1).mean().item() +
                    F.cosine_similarity(pred_B, b_B, dim=-1).mean().item()
                )

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_sample_L1": sample_l1,
            "val_sample_cosine": sample_cos
        })

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} " +
              (f"| Sample L1: {sample_l1:.4f} | Sample Cos: {sample_cos:.4f}" if sample_l1 > 0 else ""))

        ckpt_data = {"epoch": epoch + 1, "model_state_dict": net.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(ckpt_data, ckpt_dir / "last.pt")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt_data, ckpt_dir / "best.pt")

        if early_stopper(avg_val_loss):
            print("\n[EARLY STOPPING TRIGGERED]")
            break

    wandb.finish()


# ==========================================
# 7. Evaluation Script
# ==========================================
def evaluate_unconditional_demorph(data_dir: str, run_name: str, num_timesteps: int = 300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"

    _, _, _, eval_bf, eval_morphs, eval_metadata = load_normalized_datasets(data_dir)

    eval_dataset = UnconditionalMorphPairDataset(eval_morphs, eval_bf, eval_metadata)
    eval_loader = DataLoader(eval_dataset, batch_size=500, shuffle=False, num_workers=4)

    net = UnconditionalColdDemorphNet().to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    net.eval()
    diffusion = UnconditionalPairColdDemorph(net, num_timesteps=num_timesteps).to(device)

    total_l1 = 0.0
    total_cos = 0.0
    total_records = 0

    with torch.no_grad():
        for batch_M, batch_A, batch_B in tqdm(eval_loader, desc="Evaluating"):
            batch_M = batch_M.to(device)
            batch_A = batch_A.to(device)
            batch_B = batch_B.to(device)

            pred_pair = diffusion.sample_loop(batch_M)
            pred_A, pred_B = diffusion.align_pair_to_targets(pred_pair, batch_A, batch_B)

            batch_size_now = batch_M.shape[0]
            total_l1 += 0.5 * (
                F.l1_loss(pred_A, batch_A, reduction="sum").item() +
                F.l1_loss(pred_B, batch_B, reduction="sum").item()
            )
            total_cos += 0.5 * (
                F.cosine_similarity(pred_A, batch_A, dim=-1).sum().item() +
                F.cosine_similarity(pred_B, batch_B, dim=-1).sum().item()
            )
            total_records += batch_size_now

    avg_l1 = total_l1 / (total_records * 512) if total_records > 0 else float("inf")
    avg_cos = total_cos / total_records if total_records > 0 else 0.0

    results = (
        f"--- UNCONDITIONAL TWO-BRANCH EVALUATION ---\n"
        f"Recovered Pair L1 Distance : {avg_l1:.6f}\n"
        f"Recovered Pair Cosine Sim  : {avg_cos:.6f}\n"
        f"Evaluated Morphs          : {total_records}\n"
    )
    print("\n" + results)
    with open(exp_dir / "eval_metrics.txt", "w") as f:
        f.write(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings")
    parser.add_argument("--run-name", type=str, default="diffae_unconditional_pair_tacos")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
    parser.add_argument("--num-timesteps", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.mode == "train":
        train_unconditional_demorph(
            data_dir=args.data_dir,
            run_name=args.run_name,
            num_timesteps=args.num_timesteps,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size
        )
    else:
        evaluate_unconditional_demorph(
            data_dir=args.data_dir,
            run_name=args.run_name,
            num_timesteps=args.num_timesteps
        )

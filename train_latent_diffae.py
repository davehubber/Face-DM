import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
from tqdm import tqdm
import csv

# ==========================================
# 1. Dataset & Data Loading
# ==========================================
class ColdDiffAEDemorphDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, scores: np.ndarray, epoch_size: int = 1000000, deterministic: bool = False):
        self.embeddings, self.scores, self.epoch_size, self.deterministic = embeddings, scores, epoch_size, deterministic
        self.num_samples = len(embeddings)
        if len(scores) != self.num_samples: raise ValueError(f"Embeddings/scores length mismatch: {self.num_samples} vs {len(scores)}")
        if self.deterministic:
            np.random.seed(42)
            idx1 = np.random.randint(0, self.num_samples, size=epoch_size)
            offsets = np.random.randint(1, self.num_samples, size=epoch_size)
            idx2 = (idx1 + offsets) % self.num_samples
            self.pairs = np.stack([idx1, idx2], axis=1)

    def __len__(self): return self.epoch_size

    def __getitem__(self, idx):
        if self.deterministic: idx1, idx2 = self.pairs[idx]
        else:
            idx1 = np.random.randint(0, self.num_samples)
            idx2 = (idx1 + np.random.randint(1, self.num_samples)) % self.num_samples

        # Deterministic sorting: z1 always has the higher attribute logit score.
        if self.scores[idx1] < self.scores[idx2]: idx1, idx2 = idx2, idx1
        return torch.tensor(self.embeddings[idx1], dtype=torch.float32), torch.tensor(self.embeddings[idx2], dtype=torch.float32)


def load_data_and_scores(base_path_str: str, split: str, score_attr: str = "smiling"):
    base_path = Path(base_path_str).resolve()
    parent = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_val", "").replace("_test", "")
    split_path = parent / f"{stem}_{split}.npy"
    mean_path = parent / f"{stem}_train_mean.npy"
    std_path = parent / f"{stem}_train_std.npy"
    score_path = parent / f"{stem}_{split}_{score_attr.lower()}_logit.npy"

    if not split_path.exists(): raise FileNotFoundError(f"Missing split file: {split_path}")
    if not mean_path.exists() or not std_path.exists(): raise FileNotFoundError(f"Missing normalization stats: {mean_path} or {std_path}")
    if not score_path.exists(): raise FileNotFoundError(f"Missing {score_attr} score file: {score_path}. Run precompute_diffae_attribute_scores.py first.")

    data_raw = np.load(split_path).astype(np.float32)
    train_mean = np.load(mean_path).astype(np.float32).reshape(-1)
    train_std = np.load(std_path).astype(np.float32).reshape(-1)
    train_std = np.where(train_std == 0, 1.0, train_std).astype(np.float32)
    data_norm = ((data_raw - train_mean) / train_std).astype(np.float32)

    scores = np.load(score_path).astype(np.float32).reshape(-1)
    if len(scores) != len(data_norm): raise ValueError(f"Score length mismatch for {split}: {len(scores)} scores vs {len(data_norm)} embeddings")
    print(f"[{split}] embeddings: {data_norm.shape} | sorting score: {score_attr}_logit | score file: {score_path}")
    return data_norm, scores


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
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


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


class ColdDemorphNet(nn.Module):
    def __init__(self, x_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim * 2), nn.SiLU(), nn.Linear(time_emb_dim * 2, time_emb_dim))
        self.blocks = nn.ModuleList([AdaLNBlock(x_dim, hidden_dim, time_emb_dim)] + [AdaLNBlock(hidden_dim + x_dim, hidden_dim, time_emb_dim) for _ in range(num_layers - 1)])
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, t):
        t_emb, h = self.time_mlp(t), x
        for i, block in enumerate(self.blocks): h = block(h if i == 0 else torch.cat([h, x], dim=-1), t_emb)
        return self.final_linear(h)


# ==========================================
# 3. Cold Diffusion Process
# ==========================================
class DeterministicColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=50):
        super().__init__()
        self.model, self.num_timesteps = model, num_timesteps

    def degrade(self, z, c, t):
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        return (1.0 - gamma) * z + gamma * c

    def compute_loss(self, z1, z2):
        b = z1.shape[0]
        c = (z1 + z2) / 2.0
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=z1.device).long()
        x_t = self.degrade(z1, c, t)
        pred_z = self.model(x_t, t)
        return F.l1_loss(pred_z, z1)

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        device, b = c.device, c.shape[0]
        x_t = c.clone()
        for t in tqdm(torch.arange(self.num_timesteps, 0, -1, device=device).long(), desc="TACOs Sampling", leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            pred_z1 = self.model(x_t, t_batch)
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            x_t = x_t - self.degrade(pred_z1, c, t_batch) + self.degrade(pred_z1, c, t_prev_batch)
        final_z1 = x_t
        final_z2 = 2.0 * c - final_z1
        return final_z1, final_z2


# ==========================================
# 4. Training & Validation Mechanics
# ==========================================
def train_cold_demorph(diffae_path_str: str, run_name: str, score_attr: str = "smiling", num_timesteps: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"

    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f: csv.writer(f).writerow(["Epoch", "Train_Loss", "Val_Cheap_Loss", "Val_TACOs_Reconstruct_L1"])

    train_embs, train_scores = load_data_and_scores(diffae_path_str, "train", score_attr)
    val_embs, val_scores = load_data_and_scores(diffae_path_str, "val", score_attr)

    train_loader = DataLoader(ColdDiffAEDemorphDataset(train_embs, train_scores, epoch_size=1_000_000), batch_size=25_000, shuffle=True, num_workers=8)
    val_loader = DataLoader(ColdDiffAEDemorphDataset(val_embs, val_scores, epoch_size=10_000, deterministic=True), batch_size=10_000, shuffle=False, num_workers=4)

    net = ColdDemorphNet().to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)

    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={"learning_rate": 1e-4, "batch_size": 25_000, "num_layers": 10, "hidden_dim": 2048, "num_timesteps": num_timesteps, "deterministic_sorting": f"{score_attr}_logit"})

    epochs, best_val_loss = 50, float("inf")

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for batch_z1, batch_z2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            optimizer.zero_grad()
            loss = diffusion.compute_loss(batch_z1, batch_z2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        net.eval()
        val_cheap_loss, val_tacos_loss = 0.0, None

        with torch.no_grad():
            for batch_z1, batch_z2 in val_loader: val_cheap_loss += diffusion.compute_loss(batch_z1.to(device), batch_z2.to(device)).item()
        avg_val_cheap_loss = val_cheap_loss / len(val_loader)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_tacos_loss_total = 0.0
            with torch.no_grad():
                for batch_z1, batch_z2 in val_loader:
                    batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                    batch_c = (batch_z1 + batch_z2) / 2.0
                    pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c)
                    val_tacos_loss_total += (F.l1_loss(pred_z1, batch_z1, reduction="none").mean(dim=1) + F.l1_loss(pred_z2, batch_z2, reduction="none").mean(dim=1)).mean().item()
            val_tacos_loss = val_tacos_loss_total / len(val_loader)

        log_dict = {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_cheap_loss": avg_val_cheap_loss}
        if val_tacos_loss is not None: log_dict["val_tacos_reconstruct_l1"] = val_tacos_loss
        wandb.log(log_dict)

        with open(metrics_path, "a", newline="") as f: csv.writer(f).writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_cheap_loss:.6f}", f"{val_tacos_loss:.6f}" if val_tacos_loss is not None else "N/A"])

        checkpoint_data = {"epoch": epoch + 1, "model_state_dict": net.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "score_attr": score_attr}
        torch.save(checkpoint_data, ckpt_dir / "last.pt")

        if avg_val_cheap_loss < best_val_loss:
            best_val_loss = avg_val_cheap_loss
            torch.save(checkpoint_data, ckpt_dir / "best.pt")

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Cheap Loss: {avg_val_cheap_loss:.4f}" + (f" | Val TACOs L1: {val_tacos_loss:.4f}" if val_tacos_loss is not None else ""))

    wandb.finish()


# ==========================================
# 5. Partition Testing Evaluation Script
# ==========================================
def evaluate_cold_demorph(diffae_path_str: str, run_name: str, score_attr: str = "smiling", num_timesteps: int = 50, mode: str = "iterative"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / f"eval_{mode}.txt"

    test_embs, test_scores = load_data_and_scores(diffae_path_str, "test", score_attr)
    test_loader = DataLoader(ColdDiffAEDemorphDataset(test_embs, test_scores, epoch_size=10_000, deterministic=True), batch_size=10_000, shuffle=False, num_workers=4)

    net = ColdDemorphNet().to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    net.eval()

    total_l1_1, total_l1_2, total_cosine_1, total_cosine_2 = 0.0, 0.0, 0.0, 0.0
    total_cos_z1_z2, total_cos_z1_c, total_cos_z2_c, total_cos_pred1_pred2 = 0.0, 0.0, 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in test_loader:
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = (batch_z1 + batch_z2) / 2.0

            if mode == "iterative": pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c)
            else:
                pred_z1 = net(batch_c, torch.full((batch_z1.shape[0],), num_timesteps, device=device).long())
                pred_z2 = 2.0 * batch_c - pred_z1

            total_l1_1 += F.l1_loss(pred_z1, batch_z1).item()
            total_l1_2 += F.l1_loss(pred_z2, batch_z2).item()
            total_cosine_1 += F.cosine_similarity(pred_z1, batch_z1, dim=-1).mean().item()
            total_cosine_2 += F.cosine_similarity(pred_z2, batch_z2, dim=-1).mean().item()
            total_cos_z1_z2 += F.cosine_similarity(batch_z1, batch_z2, dim=-1).mean().item()
            total_cos_z1_c += F.cosine_similarity(batch_z1, batch_c, dim=-1).mean().item()
            total_cos_z2_c += F.cosine_similarity(batch_z2, batch_c, dim=-1).mean().item()
            total_cos_pred1_pred2 += F.cosine_similarity(pred_z1, pred_z2, dim=-1).mean().item()
            num_batches += 1

    label = f"{score_attr.capitalize()} classifier logit"
    results_text = (
        f"--- Evaluation Results: {mode.upper()} ---\nRun Name: {run_name}\nSorting Criterion: Highest {label}\n----------------------------------------\n"
        f"Base Reference Angular Properties:\n"
        f"  - CosSim(True Baseline z1, True Baseline z2): {total_cos_z1_z2 / num_batches:.6f}\n"
        f"  - CosSim(True Baseline z1, Average mixture c): {total_cos_z1_c / num_batches:.6f}\n"
        f"  - CosSim(True Baseline z2, Average mixture c): {total_cos_z2_c / num_batches:.6f}\n\n"
        f"[Embedding 1 (Highest {label}) Performance Alignment]\n"
        f"  - L1 Distance:                       {total_l1_1 / num_batches:.6f}\n"
        f"  - Cosine Similarity to Target:       {total_cosine_1 / num_batches:.6f}\n\n"
        f"[Embedding 2 (Lowest {label}) Performance Alignment]\n"
        f"  - L1 Distance:                       {total_l1_2 / num_batches:.6f}\n"
        f"  - Cosine Similarity to Target:       {total_cosine_2 / num_batches:.6f}\n\n"
        f"Generated Outputs Inter-Relationship:\n"
        f"  - CosSim(Aligned Pred z1, Aligned Pred z2):  {total_cos_pred1_pred2 / num_batches:.6f}\n"
    )

    print("\n" + results_text)
    with open(out_file_path, "w") as f: f.write(results_text)


if __name__ == "__main__":
    BASE_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    SCORE_ATTR = "young"
    RUN_NAME = "diffae_young"

    train_cold_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME, score_attr=SCORE_ATTR, num_timesteps=50)
    evaluate_cold_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME, score_attr=SCORE_ATTR, num_timesteps=50, mode="one_shot")
    evaluate_cold_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME, score_attr=SCORE_ATTR, num_timesteps=50, mode="iterative")

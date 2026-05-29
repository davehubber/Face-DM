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
    def __init__(self, embeddings: np.ndarray, epoch_size: int = 1000000, deterministic: bool = False):
        self.embeddings = embeddings
        self.epoch_size = epoch_size
        self.deterministic = deterministic
        self.num_samples = len(embeddings)
        
        if self.deterministic:
            np.random.seed(42)
            idx1 = np.random.randint(0, self.num_samples, size=epoch_size)
            offsets = np.random.randint(1, self.num_samples, size=epoch_size)
            idx2 = (idx1 + offsets) % self.num_samples
            self.pairs = np.stack([idx1, idx2], axis=1)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if self.deterministic:
            idx1, idx2 = self.pairs[idx]
        else:
            idx1 = np.random.randint(0, self.num_samples)
            offset = np.random.randint(1, self.num_samples)
            idx2 = (idx1 + offset) % self.num_samples
            
        return torch.tensor(self.embeddings[idx1], dtype=torch.float32), torch.tensor(self.embeddings[idx2], dtype=torch.float32)

def load_split_and_normalize(base_path_str: str, split: str) -> np.ndarray:
    base_path = Path(base_path_str).resolve()
    parent = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_val", "").replace("_test", "")
    
    split_path = parent / f"{stem}_{split}.npy"
    mean_path = parent / f"{stem}_train_mean.npy"
    std_path = parent / f"{stem}_train_std.npy"
    
    data = np.load(split_path).astype(np.float32)
    if mean_path.exists() and std_path.exists():
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        data = (data - mean) / std
    else:
        raise FileNotFoundError(f"Normalization statistics missing at {mean_path} or {std_path}")
    return data

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

class ColdDemorphNet(nn.Module):
    def __init__(self, x_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(x_dim, hidden_dim, time_emb_dim))
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + x_dim, hidden_dim, time_emb_dim))
            
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = x
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, t_emb)
            else:
                h = block(torch.cat([h, x], dim=-1), t_emb)
        return self.final_linear(h)

# ==========================================
# 3. Cold Diffusion Process
# ==========================================
class DeterministicColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=200):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, z, c, t):
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        return (1.0 - gamma) * z + gamma * c

    def compute_loss(self, z1, z2):
        b = z1.shape[0]
        c = (z1 + z2) / 2.0
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=z1.device).long()
        
        x_t = self.degrade(z1, c, t)
        pred_z = self.model(x_t, t)
        
        loss_z1 = F.l1_loss(pred_z, z1, reduction='none').mean(dim=1)
        
        is_final_step = (t == self.num_timesteps)
        if is_final_step.any():
            loss_z2 = F.l1_loss(pred_z, z2, reduction='none').mean(dim=1)
            loss_perm = torch.min(loss_z1, loss_z2)
            loss = torch.where(is_final_step, loss_perm, loss_z1)
        else:
            loss = loss_z1
            
        return loss.mean()

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        device = c.device
        b = c.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = c.clone()
        
        for t in tqdm(timesteps, desc='TACOs Sampling', leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            pred_z1 = self.model(x_t, t_batch)
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            deg_t = self.degrade(pred_z1, c, t_batch)
            deg_t_prev = self.degrade(pred_z1, c, t_prev_batch)
            
            x_t = x_t - deg_t + deg_t_prev
            
        final_z1 = x_t
        final_z2 = 2.0 * c - final_z1
        return final_z1, final_z2

# ==========================================
# 4. Training & Validation Mechanics
# ==========================================
def train_cold_demorph(diffae_path_str: str, run_name: str, num_timesteps: int = 200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Train_L1", "Val_Cheap_L1", "Val_TACOs_Reconstruct_L1"])
            
    train_embs = load_split_and_normalize(diffae_path_str, "train")
    val_embs = load_split_and_normalize(diffae_path_str, "val")
    
    train_loader = DataLoader(ColdDiffAEDemorphDataset(train_embs, epoch_size=1_000_000), batch_size=2_048, shuffle=True, num_workers=8)
    val_loader = DataLoader(ColdDiffAEDemorphDataset(val_embs, epoch_size=10_000, deterministic=True), batch_size=10_000, shuffle=False, num_workers=4)

    net = ColdDemorphNet().to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 3e-4, "batch_size": 2_048, "num_layers": 10, "hidden_dim": 2048, "num_timesteps": num_timesteps
    })

    epochs = 100
    best_val_loss = float("inf")
    
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
        val_cheap_loss = 0.0
        val_tacos_loss = None
        
        with torch.no_grad():
            for batch_z1, batch_z2 in val_loader:
                val_cheap_loss += diffusion.compute_loss(batch_z1.to(device), batch_z2.to(device)).item()
        avg_val_cheap_loss = val_cheap_loss / len(val_loader)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_tacos_loss_total = 0.0
            with torch.no_grad():
                for batch_z1, batch_z2 in val_loader:
                    batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                    batch_c = (batch_z1 + batch_z2) / 2.0
                    pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c)
                    
                    dist_a = F.l1_loss(pred_z1, batch_z1, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z2, reduction='none').mean(dim=1)
                    dist_b = F.l1_loss(pred_z1, batch_z2, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z1, reduction='none').mean(dim=1)
                    val_tacos_loss_total += torch.min(dist_a, dist_b).mean().item()
            val_tacos_loss = val_tacos_loss_total / len(val_loader)
            
        log_dict = {"epoch": epoch + 1, "train_l1": avg_train_loss, "val_cheap_l1": avg_val_cheap_loss}
        if val_tacos_loss is not None: log_dict["val_tacos_reconstruct_l1"] = val_tacos_loss
        wandb.log(log_dict)
        
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_cheap_loss:.6f}", f"{val_tacos_loss:.6f}" if val_tacos_loss is not None else "N/A"])
            
        checkpoint_data = {'epoch': epoch + 1, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint_data, ckpt_dir / "last.pt")
        
        if avg_val_cheap_loss < best_val_loss:
            best_val_loss = avg_val_cheap_loss
            torch.save(checkpoint_data, ckpt_dir / "best.pt")
            
        print(f"Epoch {epoch+1} | Train L1: {avg_train_loss:.4f} | Val Cheap L1: {avg_val_cheap_loss:.4f}" + (f" | Val TACOs L1: {val_tacos_loss:.4f}" if val_tacos_loss is not None else ""))
    wandb.finish()

# ==========================================
# 5. Partition Testing Evaluation Script
# ==========================================
def evaluate_cold_demorph(diffae_path_str: str, run_name: str, num_timesteps: int = 200, mode: str = 'iterative'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / f"eval_{mode}.txt"
    
    test_embs = load_split_and_normalize(diffae_path_str, "test")
    test_loader = DataLoader(ColdDiffAEDemorphDataset(test_embs, epoch_size=10_000, deterministic=True), batch_size=10_000, shuffle=False, num_workers=4)

    net = ColdDemorphNet().to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    net.eval()

    total_l1_1, total_l1_2, total_cosine_1, total_cosine_2 = 0.0, 0.0, 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in test_loader:
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = (batch_z1 + batch_z2) / 2.0
            
            if mode == 'iterative':
                pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c)
            else:
                pred_z1 = net(batch_c, torch.full((batch_z1.shape[0],), num_timesteps, device=device).long())
                pred_z2 = 2.0 * batch_c - pred_z1

            dist_a = F.l1_loss(pred_z1, batch_z1, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z2, reduction='none').mean(dim=1)
            dist_b = F.l1_loss(pred_z1, batch_z2, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z1, reduction='none').mean(dim=1)
            mask_a = (dist_a <= dist_b).unsqueeze(-1)
            
            aligned_pred_z1 = torch.where(mask_a, pred_z1, pred_z2)
            aligned_pred_z2 = torch.where(mask_a, pred_z2, pred_z1)

            total_l1_1 += F.l1_loss(aligned_pred_z1, batch_z1).item()
            total_l1_2 += F.l1_loss(aligned_pred_z2, batch_z2).item()
            total_cosine_1 += F.cosine_similarity(aligned_pred_z1, batch_z1, dim=-1).mean().item()
            total_cosine_2 += F.cosine_similarity(aligned_pred_z2, batch_z2, dim=-1).mean().item()
            num_batches += 1

    results_text = (
        f"--- Evaluation Results: {mode.upper()} ---\nRun Name: {run_name}\n----------------------------------------\n"
        f"[Embedding 1]\nL1 Distance:       {total_l1_1 / num_batches:.6f}\nCosine Similarity: {total_cosine_1 / num_batches:.6f}\n\n"
        f"[Embedding 2]\nL1 Distance:       {total_l1_2 / num_batches:.6f}\nCosine Similarity: {total_cosine_2 / num_batches:.6f}\n"
    )
    print("\n" + results_text)
    with open(out_file_path, "w") as f: f.write(results_text)

if __name__ == "__main__":
    BASE_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    
    train_cold_demorph(diffae_path_str=BASE_PATH, run_name="diffae_default", num_timesteps=200)
    evaluate_cold_demorph(diffae_path_str=BASE_PATH, run_name="diffae_default", num_timesteps=200, mode='one_shot')
    evaluate_cold_demorph(diffae_path_str=BASE_PATH, run_name="diffae_default", num_timesteps=200, mode='iterative')
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
class CoupledLatentDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, epoch_size: int = 1000000, deterministic: bool = False):
        self.embeddings = embeddings
        self.num_samples = len(embeddings)
        
        if deterministic:
            np.random.seed(42)
            
        # Sample heavily to ensure we can filter down to 'epoch_size' unique pairs
        idx1 = np.random.randint(0, self.num_samples, size=epoch_size * 3)
        idx2 = np.random.randint(0, self.num_samples, size=epoch_size * 3)
        
        pairs = np.stack([idx1, idx2], axis=-1)
        pairs = np.sort(pairs, axis=-1) # Enforces idx1 < idx2, eliminating inverted pairs
        pairs = pairs[pairs[:, 0] != pairs[:, 1]] # Eliminates self pairs
        pairs = np.unique(pairs, axis=0) # Eliminates duplicate pairs
        
        np.random.shuffle(pairs)
        self.pairs = pairs[:epoch_size]
        self.epoch_size = len(self.pairs)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
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
# 2. Network Architecture (DiffAE Latent DDIM)
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

class LatentDDIMBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, time_emb_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.silu = nn.SiLU()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, hidden_dim * 2)
        )

    def forward(self, x, t_emb):
        h = self.linear(x)
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)
        return self.silu(self.norm(h) * (1 + scale) + shift)

class CoupledLatentDDIMNet(nn.Module):
    def __init__(self, x_dim=1024, c_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        self.base_dim = x_dim + c_dim
        self.blocks = nn.ModuleList()
        
        # First block only takes the base input
        self.blocks.append(LatentDDIMBlock(self.base_dim, hidden_dim, time_emb_dim))
        
        # Subsequent blocks take concatenated (base input + previous hidden output)
        for _ in range(num_layers - 1):
            self.blocks.append(LatentDDIMBlock(self.base_dim + hidden_dim, hidden_dim, time_emb_dim))
            
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t, c, t):
        t_emb = self.time_mlp(t)
        base_input = torch.cat([x_t, c], dim=-1)
        
        h = base_input
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, t_emb)
            else:
                h = block(torch.cat([base_input, h], dim=-1), t_emb)
                
        return self.final_linear(h)

# ==========================================
# 3. Hot Gaussian Diffusion Process
# ==========================================
class HotLatentDiffusion(nn.Module):
    def __init__(self, model, num_train_timesteps=1000):
        super().__init__()
        self.model = model
        self.num_timesteps = num_train_timesteps
        
        # DiffAE uses a constant 0.008 beta schedule for the Latent DDIM
        betas = torch.full((num_train_timesteps,), 0.008, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def compute_loss(self, z1, z2):
        b = z1.shape[0]
        c = (z1 + z2) / 2.0
        x_0 = torch.cat([z1, z2], dim=-1) # Coupled state
        
        t = torch.randint(0, self.num_timesteps, (b,), device=z1.device).long()
        noise = torch.randn_like(x_0)
        
        alpha_bar_t = self.alphas_cumprod[t].unsqueeze(-1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        
        pred_noise = self.model(x_t, c, t)
        
        # DiffAE Latent DDIM utilizes L1 loss for predicting noise
        loss = F.l1_loss(pred_noise, noise, reduction='none').mean()
        return loss

    @torch.no_grad()
    def ddim_sample_loop(self, c, sample_steps=100):
        device = c.device
        b = c.shape[0]
        x_t = torch.randn((b, 1024), device=device)
        
        step_size = self.num_timesteps // sample_steps
        timesteps = torch.arange(self.num_timesteps - 1, -1, -step_size, device=device).long()
        
        for i, t in enumerate(tqdm(timesteps, desc='DDIM Sampling', leave=False)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            pred_noise = self.model(x_t, c, t_batch)
            
            alpha_bar_t = self.alphas_cumprod[t]
            if i < len(timesteps) - 1:
                alpha_bar_t_prev = self.alphas_cumprod[t - step_size]
            else:
                alpha_bar_t_prev = torch.tensor(1.0, device=device)
                
            # Standard DDIM reverse process equation (sigma = 0)
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * pred_noise
            
            x_t = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt
            
        final_z1, final_z2 = x_t.chunk(2, dim=-1)
        return final_z1, final_z2

# ==========================================
# 4. Training & Validation Mechanics
# ==========================================
def train_hot_demorph(diffae_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Train_L1_Noise_Loss", "Val_L1_Noise_Loss", "Val_DDIM_Reconstruct_L1"])
            
    train_embs = load_split_and_normalize(diffae_path_str, "train")
    val_embs = load_split_and_normalize(diffae_path_str, "val")
    
    train_loader = DataLoader(CoupledLatentDataset(train_embs, epoch_size=1_000_000), batch_size=15_000, shuffle=True, num_workers=8)
    val_loader = DataLoader(CoupledLatentDataset(val_embs, epoch_size=10_000, deterministic=True), batch_size=10_000, shuffle=False, num_workers=4)

    net = CoupledLatentDDIMNet().to(device)
    diffusion = HotLatentDiffusion(net, num_train_timesteps=1000).to(device)
    
    # DiffAE utilizes AdamW with weight decay 0.01 for the latent DDIM
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4, "batch_size": 15_000, "num_layers": 10, "hidden_dim": 2048, "num_train_timesteps": 1000
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
        val_noise_loss = 0.0
        val_reconstruct_loss = None
        
        with torch.no_grad():
            for batch_z1, batch_z2 in val_loader:
                val_noise_loss += diffusion.compute_loss(batch_z1.to(device), batch_z2.to(device)).item()
        avg_val_noise_loss = val_noise_loss / len(val_loader)

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_reconstruct_loss_total = 0.0
            with torch.no_grad():
                for batch_z1, batch_z2 in val_loader:
                    batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                    batch_c = (batch_z1 + batch_z2) / 2.0
                    pred_z1, pred_z2 = diffusion.ddim_sample_loop(batch_c, sample_steps=100)
                    
                    # Permutation invariant evaluation
                    dist_a = F.l1_loss(pred_z1, batch_z1, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z2, reduction='none').mean(dim=1)
                    dist_b = F.l1_loss(pred_z1, batch_z2, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z1, reduction='none').mean(dim=1)
                    val_reconstruct_loss_total += torch.min(dist_a, dist_b).mean().item()
            val_reconstruct_loss = val_reconstruct_loss_total / len(val_loader)
            
        log_dict = {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_noise_loss": avg_val_noise_loss}
        if val_reconstruct_loss is not None: log_dict["val_ddim_reconstruct_l1"] = val_reconstruct_loss
        wandb.log(log_dict)
        
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_noise_loss:.6f}", f"{val_reconstruct_loss:.6f}" if val_reconstruct_loss is not None else "N/A"])
            
        checkpoint_data = {'epoch': epoch + 1, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint_data, ckpt_dir / "last.pt")
        
        if avg_val_noise_loss < best_val_loss:
            best_val_loss = avg_val_noise_loss
            torch.save(checkpoint_data, ckpt_dir / "best.pt")
            
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Noise Loss: {avg_val_noise_loss:.4f}" + (f" | Val DDIM L1: {val_reconstruct_loss:.4f}" if val_reconstruct_loss is not None else ""))
    wandb.finish()

# ==========================================
# 5. Partition Testing Evaluation Script
# ==========================================
def evaluate_hot_demorph(diffae_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / "eval_ddim.txt"
    
    test_embs = load_split_and_normalize(diffae_path_str, "test")
    test_loader = DataLoader(CoupledLatentDataset(test_embs, epoch_size=10_000, deterministic=True), batch_size=10_000, shuffle=False, num_workers=4)

    net = CoupledLatentDDIMNet().to(device)
    diffusion = HotLatentDiffusion(net, num_train_timesteps=1000).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    net.eval()

    total_l1_1, total_l1_2, total_cosine_1, total_cosine_2 = 0.0, 0.0, 0.0, 0.0
    total_cos_z1_z2, total_cos_z1_c, total_cos_z2_c, total_cos_pred1_pred2 = 0.0, 0.0, 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in test_loader:
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = (batch_z1 + batch_z2) / 2.0
            
            pred_z1, pred_z2 = diffusion.ddim_sample_loop(batch_c, sample_steps=100)

            dist_a = F.l1_loss(pred_z1, batch_z1, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z2, reduction='none').mean(dim=1)
            dist_b = F.l1_loss(pred_z1, batch_z2, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z1, reduction='none').mean(dim=1)
            mask_a = (dist_a <= dist_b).unsqueeze(-1)
            
            aligned_pred_z1 = torch.where(mask_a, pred_z1, pred_z2)
            aligned_pred_z2 = torch.where(mask_a, pred_z2, pred_z1)

            total_l1_1 += F.l1_loss(aligned_pred_z1, batch_z1).item()
            total_l1_2 += F.l1_loss(aligned_pred_z2, batch_z2).item()
            total_cosine_1 += F.cosine_similarity(aligned_pred_z1, batch_z1, dim=-1).mean().item()
            total_cosine_2 += F.cosine_similarity(aligned_pred_z2, batch_z2, dim=-1).mean().item()
            
            total_cos_z1_z2 += F.cosine_similarity(batch_z1, batch_z2, dim=-1).mean().item()
            total_cos_z1_c += F.cosine_similarity(batch_z1, batch_c, dim=-1).mean().item()
            total_cos_z2_c += F.cosine_similarity(batch_z2, batch_c, dim=-1).mean().item()
            total_cos_pred1_pred2 += F.cosine_similarity(aligned_pred_z1, aligned_pred_z2, dim=-1).mean().item()
            
            num_batches += 1

    results_text = (
        f"--- Evaluation Results: DDIM (100 Steps) ---\nRun Name: {run_name}\n----------------------------------------\n"
        f"Base Reference Angular Properties:\n"
        f"  - CosSim(True Baseline z1, True Baseline z2): {total_cos_z1_z2 / num_batches:.6f}\n"
        f"  - CosSim(True Baseline z1, Average mixture c): {total_cos_z1_c / num_batches:.6f}\n"
        f"  - CosSim(True Baseline z2, Average mixture c): {total_cos_z2_c / num_batches:.6f}\n\n"
        f"[Embedding 1 Performance Alignment]\n"
        f"  - L1 Distance:                       {total_l1_1 / num_batches:.6f}\n"
        f"  - Cosine Similarity to Target:       {total_cosine_1 / num_batches:.6f}\n\n"
        f"[Embedding 2 Performance Alignment]\n"
        f"  - L1 Distance:                       {total_l1_2 / num_batches:.6f}\n"
        f"  - Cosine Similarity to Target:       {total_cosine_2 / num_batches:.6f}\n\n"
        f"Generated Outputs Inter-Relationship:\n"
        f"  - CosSim(Aligned Pred z1, Aligned Pred z2):  {total_cos_pred1_pred2 / num_batches:.6f}\n"
    )
    print("\n" + results_text)
    with open(out_file_path, "w") as f: f.write(results_text)

if __name__ == "__main__":
    BASE_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    RUN_NAME = "diffae_hot_latent_ddim"
    
    train_hot_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME)
    evaluate_hot_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME)

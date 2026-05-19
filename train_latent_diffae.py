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
    """
    Dynamically generates pairs of Diff-AE semantic embeddings and their average.
    """
    def __init__(self, embeddings: np.ndarray, epoch_size: int = 1000000, deterministic: bool = False):
        self.embeddings = embeddings
        self.epoch_size = epoch_size
        self.deterministic = deterministic
        self.num_samples = len(embeddings)
        
        if self.deterministic:
            np.random.seed(42)
            self.pairs = np.random.randint(0, self.num_samples, size=(epoch_size, 2))

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if self.deterministic:
            idx1, idx2 = self.pairs[idx]
        else:
            idx1, idx2 = np.random.randint(0, self.num_samples, size=2)

        z1 = self.embeddings[idx1]
        z2 = self.embeddings[idx2]

        return torch.tensor(z1, dtype=torch.float32), torch.tensor(z2, dtype=torch.float32)

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
        h = self.norm(h) * (1 + scale) + shift
        return self.silu(h)

class ColdDemorphNet(nn.Module):
    """
    Predicts ONE 512D clean embedding given a degraded mixture and the average condition.
    """
    def __init__(self, x_dim=512, c_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        cond_dim = time_emb_dim + c_dim
        
        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(x_dim, hidden_dim, cond_dim))
        
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + x_dim, hidden_dim, cond_dim))
            
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, t, c):
        t_emb = self.time_mlp(t)
        cond = torch.cat([t_emb, c], dim=-1)
        
        h = x
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, cond)
            else:
                h = block(torch.cat([h, x], dim=-1), cond)
                
        return self.final_linear(h)

# ==========================================
# 3. Cold Demorph Diffusion Process
# ==========================================
class DeterministicColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=10):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, z1, z2, t):
        """
        Forward degradation: Mixes z1 and z2.
        t=0: alpha=1.0 -> purely z1
        t=T: alpha=0.5 -> purely (z1+z2)/2 (the average)
        """
        alpha = 1.0 - 0.5 * (t / self.num_timesteps).view(-1, 1).float()
        return alpha * z1 + (1.0 - alpha) * z2

    def compute_loss(self, z1, z2):
        b = z1.shape[0]
        c = (z1 + z2) / 2.0
        
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=z1.device).long()
        x_t = self.degrade(z1, z2, t)
        pred_z = self.model(x_t, t, c)
        
        loss_1 = F.l1_loss(pred_z, z1, reduction='none').mean(dim=1)
        loss_2 = F.l1_loss(pred_z, z2, reduction='none').mean(dim=1)
        
        return torch.min(loss_1, loss_2).mean()

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        device = c.device
        b = c.shape[0]
        
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = c.clone()
        
        for t in tqdm(timesteps, desc='TACOs Sampling', leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            pred_z1 = self.model(x_t, t_batch, c)
            pred_z2 = 2.0 * c - pred_z1
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            
            deg_t = self.degrade(pred_z1, pred_z2, t_batch)
            deg_t_prev = self.degrade(pred_z1, pred_z2, t_prev_batch)
            
            x_t = x_t - deg_t + deg_t_prev
            
        final_z1 = x_t
        final_z2 = 2.0 * c - final_z1
        
        return final_z1, final_z2

# ==========================================
# 4. Evaluation & Training Loop
# ==========================================
def load_and_zscore_diffae(diffae_path_str: str) -> np.ndarray:
    diffae_path = Path(diffae_path_str).resolve()
    print(f"Loading DiffAE embeddings from: {diffae_path}")
    raw_diffae = np.load(diffae_path).astype(np.float32)
    
    mean_path = diffae_path.parent / f"{diffae_path.stem}_mean.npy"
    std_path = diffae_path.parent / f"{diffae_path.stem}_std.npy"
    
    if mean_path.exists() and std_path.exists():
        zscore_mean = np.load(mean_path).astype(np.float32)
        zscore_std = np.load(std_path).astype(np.float32)
        diffae_embs = (raw_diffae - zscore_mean) / zscore_std
        print("Z-score normalization applied successfully.")
        return diffae_embs
    else:
        raise FileNotFoundError(f"Missing Z-score statistics: {mean_path.name} or {std_path.name}")

def train_cold_demorph(diffae_path_str: str, run_name: str, num_timesteps: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Directory Setup
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_L1", "Val_Cheap_L1", "Val_TACOs_Reconstruct_L1"])
            
    # Data Loading
    diffae_embs = load_and_zscore_diffae(diffae_path_str)

    split_idx = int(len(diffae_embs) * 0.9)
    train_embs, val_embs = diffae_embs[:split_idx], diffae_embs[split_idx:]
    
    train_dataset = ColdDiffAEDemorphDataset(train_embs, epoch_size=1_000_000, deterministic=False)
    val_dataset = ColdDiffAEDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16_384, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16_384, shuffle=False, num_workers=4)

    # Model Setup
    net = ColdDemorphNet(x_dim=512, c_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4,
        "batch_size": 16_384,
        "num_layers": 10,
        "hidden_dim": 2048,
        "num_timesteps": num_timesteps,
        "latent_space": "Diff-AE"
    })

    epochs = 50 
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_z1, batch_z2 in pbar:
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            
            optimizer.zero_grad()
            loss = diffusion.compute_loss(z1=batch_z1, z2=batch_z2)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"L1_loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop (Cheap Metric)
        net.eval()
        val_cheap_loss = 0.0
        val_tacos_loss = None
        
        with torch.no_grad():
            for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val Cheap]"):
                batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                loss_cheap = diffusion.compute_loss(z1=batch_z1, z2=batch_z2)
                val_cheap_loss += loss_cheap.item()
                
        avg_val_cheap_loss = val_cheap_loss / len(val_loader)

        # Validation Loop (Expensive TACOs Metric)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_tacos_loss_total = 0.0
            
            with torch.no_grad():
                for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val TACOs]"):
                    batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                    batch_c = (batch_z1 + batch_z2) / 2.0
                    
                    pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c)
                    
                    dist_a = F.l1_loss(pred_z1, batch_z1, reduction='none').mean(dim=1) + \
                             F.l1_loss(pred_z2, batch_z2, reduction='none').mean(dim=1)
                             
                    dist_b = F.l1_loss(pred_z1, batch_z2, reduction='none').mean(dim=1) + \
                             F.l1_loss(pred_z2, batch_z1, reduction='none').mean(dim=1)
                             
                    loss = torch.min(dist_a, dist_b).mean()
                    val_tacos_loss_total += loss.item()
                    
            val_tacos_loss = val_tacos_loss_total / len(val_loader)
            
        # Logging & Checkpointing
        log_dict = {
            "epoch": epoch + 1,
            "train_l1": avg_train_loss,
            "val_cheap_l1": avg_val_cheap_loss
        }
        if val_tacos_loss is not None:
            log_dict["val_tacos_reconstruct_l1"] = val_tacos_loss
            
        wandb.log(log_dict)
        
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{avg_train_loss:.6f}", 
                f"{avg_val_cheap_loss:.6f}", 
                f"{val_tacos_loss:.6f}" if val_tacos_loss is not None else "N/A"
            ])
            
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_cheap_loss': avg_val_cheap_loss,
        }
        
        torch.save(checkpoint_data, ckpt_dir / "last.pt")
        
        if avg_val_cheap_loss < best_val_loss:
            best_val_loss = avg_val_cheap_loss
            torch.save(checkpoint_data, ckpt_dir / "best.pt")
            print(f"--> Saved new best checkpoint based on Cheap Val to {ckpt_dir / 'best.pt'}")
            
        print(f"Epoch {epoch+1} | Train L1: {avg_train_loss:.4f} | Val Cheap L1: {avg_val_cheap_loss:.4f}", end="")
        if val_tacos_loss is not None:
            print(f" | Val TACOs L1: {val_tacos_loss:.4f}")
        else:
            print()

    wandb.finish()


# ==========================================
# 5. Independent Evaluation Script
# ==========================================
def evaluate_cold_demorph(diffae_path_str: str, run_name: str, num_timesteps: int = 10, mode: str = 'iterative'):
    assert mode in ['iterative', 'one_shot'], "Mode must be 'iterative' or 'one_shot'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    out_file_path = exp_dir / f"eval_{mode}.txt"
    
    # Data Loading
    print(f"Loading DiffAE embeddings for {mode} evaluation...")
    diffae_embs = load_and_zscore_diffae(diffae_path_str)
    
    split_idx = int(len(diffae_embs) * 0.9)
    val_embs = diffae_embs[split_idx:]
    
    val_dataset = ColdDiffAEDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)

    # Model Setup
    net = ColdDemorphNet(x_dim=512, c_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # Evaluation Loop
    total_l1_1, total_l1_2 = 0.0, 0.0
    total_cosine_1, total_cosine_2 = 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Evaluating ({mode})"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = (batch_z1 + batch_z2) / 2.0
            b = batch_z1.shape[0]
            
            # --- 1. Generation ---
            if mode == 'iterative':
                pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c)
            else: # one_shot
                t_batch = torch.full((b,), num_timesteps, device=device, dtype=torch.long)
                pred_z1 = net(batch_c, t_batch, batch_c)
                pred_z2 = 2.0 * batch_c - pred_z1

            # --- 2. Permutation Alignment ---
            dist_a = F.l1_loss(pred_z1, batch_z1, reduction='none').mean(dim=1) + \
                     F.l1_loss(pred_z2, batch_z2, reduction='none').mean(dim=1)
                     
            dist_b = F.l1_loss(pred_z1, batch_z2, reduction='none').mean(dim=1) + \
                     F.l1_loss(pred_z2, batch_z1, reduction='none').mean(dim=1)
                     
            mask_a = (dist_a <= dist_b).unsqueeze(-1)
            
            aligned_pred_z1 = torch.where(mask_a, pred_z1, pred_z2)
            aligned_pred_z2 = torch.where(mask_a, pred_z2, pred_z1)

            # --- 3. Compute Metrics directly in the standard space ---
            l1_1 = F.l1_loss(aligned_pred_z1, batch_z1)
            l1_2 = F.l1_loss(aligned_pred_z2, batch_z2)
            
            cos_sim_1 = F.cosine_similarity(aligned_pred_z1, batch_z1, dim=-1).mean()
            cos_sim_2 = F.cosine_similarity(aligned_pred_z2, batch_z2, dim=-1).mean()
            
            total_l1_1 += l1_1.item()
            total_l1_2 += l1_2.item()
            total_cosine_1 += cos_sim_1.item()
            total_cosine_2 += cos_sim_2.item()
            num_batches += 1

    avg_l1_1 = total_l1_1 / num_batches
    avg_l1_2 = total_l1_2 / num_batches
    avg_cosine_1 = total_cosine_1 / num_batches
    avg_cosine_2 = total_cosine_2 / num_batches

    results_text = (
        f"--- Evaluation Results: {mode.upper()} ---\n"
        f"Run Name: {run_name}\n"
        f"Validation Pairs: {len(val_dataset)}\n"
        f"----------------------------------------\n"
        f"[Embedding 1 (Primary Predicted)]\n"
        f"L1 Distance:       {avg_l1_1:.6f}\n"
        f"Cosine Similarity: {avg_cosine_1:.6f}\n\n"
        f"[Embedding 2 (Mathematically Extracted)]\n"
        f"L1 Distance:       {avg_l1_2:.6f}\n"
        f"Cosine Similarity: {avg_cosine_2:.6f}\n"
    )
    
    print("\n" + results_text)
    
    with open(out_file_path, "w") as f:
        f.write(results_text)
        
    print(f"Saved evaluation results to {out_file_path}")

if __name__ == "__main__":
    train_cold_demorph(
        diffae_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy",
        run_name="avg_diffae",
        num_timesteps=10
    )

    evaluate_cold_demorph(
        diffae_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy",
        run_name="avg_diffae",
        num_timesteps=10,
        mode='one_shot'
    )
    
    evaluate_cold_demorph(
        diffae_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy",
        run_name="avg_diffae",
        num_timesteps=10,
        mode='iterative'
    )

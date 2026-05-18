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
class TranslationDataset(Dataset):
    """
    1-to-1 Mapping Dataset.
    Now optimized for ArcFace -> Diff-AE translation.
    """
    def __init__(self, diffae_embs: np.ndarray, arcface_embs: np.ndarray):
        self.diffae_embs = diffae_embs
        
        # Scale ArcFace by sqrt(512) so its magnitude matches the Z-scored Diff-AE space.
        # This keeps the interpolation mathematically stable.
        self.arcface_embs = arcface_embs * np.sqrt(512)
        
    def __len__(self):
        return len(self.diffae_embs)

    def __getitem__(self, idx):
        # Return (target_x0, condition_xT)
        # Target: Diff-AE | Condition (Terminal): ArcFace
        return (
            torch.tensor(self.diffae_embs[idx], dtype=torch.float32), 
            torch.tensor(self.arcface_embs[idx], dtype=torch.float32)
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
        h = self.norm(h) * (1 + scale) + shift
        return self.silu(h)

class ColdTranslationNet(nn.Module):
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
# 3. Cold Diffusion Process
# ==========================================
class DeterministicColdDiffusion(nn.Module):
    def __init__(self, model, num_timesteps=10):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, x_0, x_cond, t):
        """
        Deterministic linear interpolation.
        t=0: Pure Diff-AE (x_0)
        t=T: Pure ArcFace (x_cond)
        """
        s = (t / self.num_timesteps).view(-1, 1).float()
        return (1.0 - s) * x_0 + s * x_cond

    def compute_loss(self, x_0, x_cond):
        b = x_0.shape[0]
        # Uniformly sample t from {1, 2, ..., T}
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=x_0.device).long()
        
        x_t = self.degrade(x_0, x_cond, t)
        pred_x_0 = self.model(x_t, t, x_cond)
        
        return F.l1_loss(pred_x_0, x_0)

    @torch.no_grad()
    def tacos_sample_loop(self, x_cond):
        device = x_cond.device
        b = x_cond.shape[0]
        
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        
        # Start sampling from the terminal degraded state (ArcFace condition)
        x_t = x_cond.clone()
        
        for t in tqdm(timesteps, desc='TACOs Sampling', leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # 1. Predict clean x_0 (Diff-AE)
            pred_x_0 = self.model(x_t, t_batch, x_cond)
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            
            # 2. Re-apply degradation to t and t-1
            deg_t = self.degrade(pred_x_0, x_cond, t_batch)
            deg_t_prev = self.degrade(pred_x_0, x_cond, t_prev_batch)
            
            # 3. TACOs update step
            x_t = x_t - deg_t + deg_t_prev
            
        return x_t

# ==========================================
# 4. Evaluation & Training Loop
# ==========================================
def train_cold_translation(diffae_path_str: str, arcface_path_str: str, run_name: str, num_timesteps: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------------------------------
    # Directory Setup
    # ------------------------------------------
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_L1", "Val_Cheap_L1", "Val_TACOs_Reconstruct_L1"])
            
    # ------------------------------------------
    # Data Loading & Normalization
    # ------------------------------------------
    print("Loading DiffAE embeddings...")
    diffae_path = Path(diffae_path_str).resolve()
    raw_diffae = np.load(diffae_path).astype(np.float32)
    
    mean_path = diffae_path.parent / f"{diffae_path.stem}_mean.npy"
    std_path = diffae_path.parent / f"{diffae_path.stem}_std.npy"
    
    if mean_path.exists() and std_path.exists():
        zscore_mean = np.load(mean_path).astype(np.float32)
        zscore_std = np.load(std_path).astype(np.float32)
        diffae_embs = (raw_diffae - zscore_mean) / zscore_std
    else:
        raise FileNotFoundError("Missing DiffAE Z-score statistics.")
        
    print("Loading ArcFace embeddings...")
    arcface_path = Path(arcface_path_str).resolve()
    arcface_embs = np.load(arcface_path).astype(np.float32)
    
    assert len(diffae_embs) == len(arcface_embs), "Dataset lengths must match perfectly."

    # ------------------------------------------
    # Dataset Splits
    # ------------------------------------------
    split_idx = int(len(diffae_embs) * 0.9)
    
    train_dataset = TranslationDataset(diffae_embs[:split_idx], arcface_embs[:split_idx])
    val_dataset = TranslationDataset(diffae_embs[split_idx:], arcface_embs[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)

    # ------------------------------------------
    # Model Setup
    # ------------------------------------------
    net = ColdTranslationNet(x_dim=512, c_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDiffusion(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4,
        "batch_size": 512,
        "num_layers": 10,
        "hidden_dim": 2048,
        "num_timesteps": num_timesteps,
        "direction": "arcface_to_diffae"
    })

    epochs = 50
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        # Now extracting batch_target (Diff-AE) and batch_cond (ArcFace)
        for batch_target, batch_cond in pbar:
            batch_target, batch_cond = batch_target.to(device), batch_cond.to(device)
            
            optimizer.zero_grad()
            loss = diffusion.compute_loss(x_0=batch_target, x_cond=batch_cond)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"L1_loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # ------------------------------------------
        # Validation Loop (Cheap Metric)
        # ------------------------------------------
        net.eval()
        val_cheap_loss = 0.0
        val_tacos_loss = None
        
        with torch.no_grad():
            for batch_target, batch_cond in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val Cheap]"):
                batch_target, batch_cond = batch_target.to(device), batch_cond.to(device)
                
                loss_cheap = diffusion.compute_loss(x_0=batch_target, x_cond=batch_cond)
                val_cheap_loss += loss_cheap.item()
                
        avg_val_cheap_loss = val_cheap_loss / len(val_loader)

        # ------------------------------------------
        # Validation Loop (Expensive TACOs Metric)
        # ------------------------------------------
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_tacos_loss_total = 0.0
            
            with torch.no_grad():
                for batch_target, batch_cond in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val TACOs]"):
                    batch_target, batch_cond = batch_target.to(device), batch_cond.to(device)
                    
                    # Generate Diff-AE from ArcFace
                    sampled_target = diffusion.tacos_sample_loop(batch_cond)
                    
                    # Compute L1 Loss directly in the Z-scored Diff-AE space
                    loss = F.l1_loss(sampled_target, batch_target)
                    val_tacos_loss_total += loss.item()
                    
            val_tacos_loss = val_tacos_loss_total / len(val_loader)
            
        # ------------------------------------------
        # Logging & Checkpointing
        # ------------------------------------------
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

if __name__ == "__main__":
    train_cold_translation(
        diffae_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy",
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="arcface_to_diffae_100ts"
    )

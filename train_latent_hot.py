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
class DemorphEmbeddingDataset(Dataset):
    """
    Dynamically generates pairs of semantic embeddings and their average.
    Magnitude ordering has been removed; permutation invariance is handled natively in the loss.
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

        z_avg = (z1 + z2) / 2.0
        
        # Target is the concatenated 1024D vector
        y = np.concatenate([z1, z2], axis=0)

        return torch.tensor(y, dtype=torch.float32), torch.tensor(z_avg, dtype=torch.float32)

# ==========================================
# 2. Network Architecture (Latent DDIM)
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

class LatentDemorphNet(nn.Module):
    def __init__(self, x_dim=1024, c_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
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
# 3. Diffusion Process
# ==========================================
class GaussianDiffusion(nn.Module):
    def __init__(self, model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, cond):
        """
        Computes the permutation-invariant L1 loss purely in the epsilon (noise) domain.
        """
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()
        
        # Baseline noise (Path A)
        noise_a = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise_a)
        predicted_noise = self.model(x_noisy, t, cond)
        
        # Loss A: Target is the original arrangement
        loss_a = F.l1_loss(predicted_noise, noise_a, reduction='none').mean(dim=1)
        
        # Construct the swapped target (Path B)
        z1, z2 = x_start.chunk(2, dim=1)
        x_start_b = torch.cat([z2, z1], dim=1)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        # Calculate the theoretical noise that would have generated the SAME x_noisy
        # if the underlying clean data was actually the swapped pair (x_start_b)
        noise_b = (x_noisy - sqrt_alphas_cumprod_t * x_start_b) / sqrt_one_minus_alphas_cumprod_t
        
        # Loss B: Target is the swapped arrangement
        loss_b = F.l1_loss(predicted_noise, noise_b, reduction='none').mean(dim=1)
        
        # Permutation invariance: let the network choose the easiest path
        return torch.min(loss_a, loss_b).mean()

    @torch.no_grad()
    def ddim_sample_loop(self, cond, shape, sampling_timesteps=50, eta=0.0):
        device = self.betas.device
        b = shape[0]
        
        step_ratio = self.num_timesteps // sampling_timesteps
        timesteps = (torch.arange(0, sampling_timesteps, device=device) * step_ratio).long()
        timesteps = torch.flip(timesteps, dims=(0,))
        
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(tqdm(timesteps, desc='DDIM Sampling', leave=False)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            predicted_noise = self.model(x, t_batch, cond)
            
            alpha = self.alphas_cumprod[t]
            t_prev = t - step_ratio
            alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            
            pred_x0 = (x - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
            dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * predicted_noise
            
            noise = torch.randn_like(x) if t_prev >= 0 else torch.zeros_like(x)
            
            x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma * noise
            
        return x

# ==========================================
# 4. Evaluation & Training Loop
# ==========================================
def compute_permutation_invariant_loss(pred_y, target_y):
    pred_z1, pred_z2 = pred_y.chunk(2, dim=1)
    tgt_z1, tgt_z2 = target_y.chunk(2, dim=1)
    
    dist_a = F.l1_loss(pred_z1, tgt_z1, reduction='none').mean(dim=1) + \
             F.l1_loss(pred_z2, tgt_z2, reduction='none').mean(dim=1)
             
    dist_b = F.l1_loss(pred_z1, tgt_z2, reduction='none').mean(dim=1) + \
             F.l1_loss(pred_z2, tgt_z1, reduction='none').mean(dim=1)
             
    return torch.min(dist_a, dist_b).mean()

def train_latent_demorph(embeddings_path_str: str, run_name: str, sampling_timesteps: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Directory Setup
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_L1_Noise", "Val_L1_Noise", "Val_Perm_Inv_Reconstruct_L1"])
    
    # Data Loading & Normalization
    embeddings_path = Path(embeddings_path_str).resolve()
    print(f"Loading raw embeddings from: {embeddings_path}")
    raw_embeddings = np.load(embeddings_path).astype(np.float32)
    
    mean_path = embeddings_path.parent / f"{embeddings_path.stem}_mean.npy"
    std_path = embeddings_path.parent / f"{embeddings_path.stem}_std.npy"
    
    if mean_path.exists() and std_path.exists():
        print("Found Z-score statistics. Applying normalization...")
        zscore_mean = np.load(mean_path).astype(np.float32)
        zscore_std = np.load(std_path).astype(np.float32)
        
        embeddings = (raw_embeddings - zscore_mean) / zscore_std
        print("Z-score normalization applied successfully.")
    else:
        raise FileNotFoundError(f"Missing Z-score statistics. Please ensure {mean_path.name} and {std_path.name} exist.")
    
    split_idx = int(len(embeddings) * 0.9)
    train_embs, val_embs = embeddings[:split_idx], embeddings[split_idx:]
    
    train_dataset = DemorphEmbeddingDataset(train_embs, epoch_size=1_000_000, deterministic=False)
    val_dataset = DemorphEmbeddingDataset(val_embs, epoch_size=10_000, deterministic=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16384, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16384, shuffle=False, num_workers=4)
    
    # Model Setup
    net = LatentDemorphNet(x_dim=1024, c_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = GaussianDiffusion(net, num_timesteps=1000).to(device)
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4,
        "batch_size": 16384,
        "num_layers": 10,
        "hidden_dim": 2048,
        "train_timesteps": 1000,
        "sampling_timesteps": sampling_timesteps
    })

    epochs = 50 
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_y, batch_cond in pbar:
            batch_y, batch_cond = batch_y.to(device), batch_cond.to(device)
            
            optimizer.zero_grad()
            loss = diffusion.compute_loss(batch_y, batch_cond)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"L1_noise_loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop (Cheap Metric)
        net.eval()
        val_noise_loss = 0.0
        val_perm_inv_loss = None
        
        with torch.no_grad():
            for batch_y, batch_cond in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val Noise]"):
                batch_y, batch_cond = batch_y.to(device), batch_cond.to(device)
                
                loss_noise = diffusion.compute_loss(batch_y, batch_cond)
                val_noise_loss += loss_noise.item()
                
        avg_val_noise_loss = val_noise_loss / len(val_loader)
        
        # Validation Loop (Expensive Metric)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_perm_inv_loss_total = 0.0
            
            with torch.no_grad():
                for batch_y, batch_cond in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val DDIM]"):
                    batch_y, batch_cond = batch_y.to(device), batch_cond.to(device)
                    
                    sampled_y = diffusion.ddim_sample_loop(
                        batch_cond, 
                        shape=batch_y.shape, 
                        sampling_timesteps=sampling_timesteps
                    )
                    loss = compute_permutation_invariant_loss(sampled_y, batch_y)
                    val_perm_inv_loss_total += loss.item()
                    
            val_perm_inv_loss = val_perm_inv_loss_total / len(val_loader)
        
        # Logging & Checkpointing
        log_dict = {
            "epoch": epoch + 1,
            "train_noise_l1": avg_train_loss,
            "val_noise_l1": avg_val_noise_loss
        }
        if val_perm_inv_loss is not None:
            log_dict["val_reconstruction_perm_inv_l1"] = val_perm_inv_loss
            
        wandb.log(log_dict)
        
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{avg_train_loss:.6f}", 
                f"{avg_val_noise_loss:.6f}", 
                f"{val_perm_inv_loss:.6f}" if val_perm_inv_loss is not None else "N/A"
            ])
            
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_noise_loss': avg_val_noise_loss,
        }
        
        torch.save(checkpoint_data, ckpt_dir / "last.pt")
        
        if avg_val_noise_loss < best_val_loss:
            best_val_loss = avg_val_noise_loss
            torch.save(checkpoint_data, ckpt_dir / "best.pt")
            print(f"--> Saved new best checkpoint based on Val Noise to {ckpt_dir / 'best.pt'}")
            
        print(f"Epoch {epoch+1} | Train L1: {avg_train_loss:.4f} | Val Noise L1: {avg_val_noise_loss:.4f}", end="")
        if val_perm_inv_loss is not None:
            print(f" | Val Perm-Inv L1: {val_perm_inv_loss:.4f}")
        else:
            print()

    wandb.finish()

if __name__ == "__main__":
    train_latent_demorph(
        embeddings_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy",
        run_name="avg_diffae_hot_pit"
    )

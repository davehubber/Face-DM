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
class ColdDiffAEDemorphTrainDataset(Dataset):
    """Generates random on-the-fly training pairs from the training matrix."""
    def __init__(self, embeddings: np.ndarray, epoch_size: int = 1_000_000):
        self.embeddings = embeddings
        self.epoch_size = epoch_size
        self.num_samples = len(embeddings)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        idx1 = np.random.randint(0, self.num_samples)
        offset = np.random.randint(1, self.num_samples)
        idx2 = (idx1 + offset) % self.num_samples
            
        return (torch.tensor(self.embeddings[idx1], dtype=torch.float32), 
                torch.tensor(self.embeddings[idx2], dtype=torch.float32))

class ColdDiffAEDemorphTestPairsDataset(Dataset):
    """Loads pre-paired evaluation arrays directly from disk."""
    def __init__(self, paired_embeddings: np.ndarray):
        self.pairs = paired_embeddings

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        z1 = self.pairs[idx, 0]
        z2 = self.pairs[idx, 1]
        return torch.tensor(z1, dtype=torch.float32), torch.tensor(z2, dtype=torch.float32)

def load_split_and_normalize(base_path_str: str, split: str) -> np.ndarray:
    base_path = Path(base_path_str).resolve()
    parent = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_test_pairs", "")
    
    mean_path = parent / f"{stem}_train_mean.npy"
    std_path = parent / f"{stem}_train_std.npy"
    
    if split == "train":
        split_path = parent / f"{stem}_train.npy"
    elif split == "test":
        split_path = parent / f"{stem}_test_pairs.npy"
    else:
        raise ValueError(f"Unknown split type requested: {split}")
        
    if not split_path.exists():
        raise FileNotFoundError(f"Split file missing at: {split_path}")
        
    data = np.load(split_path).astype(np.float32)
    
    if mean_path.exists() and std_path.exists():
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        
        if split == "test":
            mean = mean[np.newaxis, :]
            std = std[np.newaxis, :]
            
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
            
        self.final_linear = nn.Linear(hidden_dim, x_dim * 2)

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
# 3. Cold Diffusion Process (Dual-Target PIT)
# ==========================================
class DeterministicColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=300):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.SQRT_2 = math.sqrt(2.0)

    def degrade(self, z1, z2, t):
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        w1 = torch.sqrt(1.0 - 0.5 * gamma)
        w2 = torch.sqrt(0.5 * gamma)
        return w1 * z1 + w2 * z2

    def compute_loss(self, z1, z2, t=None):
        b = z1.shape[0]
        if t is None:
            t = torch.randint(1, self.num_timesteps + 1, (b,), device=z1.device).long()
        
        x_t = self.degrade(z1, z2, t)
        pred = self.model(x_t, t)
        
        pred_z1_raw, pred_z2_raw = pred.chunk(2, dim=-1)
        
        # 1. Standard L1 PIT Loss
        loss_A = F.l1_loss(pred_z1_raw, z1, reduction='none').mean(dim=-1) + \
                 F.l1_loss(pred_z2_raw, z2, reduction='none').mean(dim=-1)
                 
        loss_B = F.l1_loss(pred_z1_raw, z2, reduction='none').mean(dim=-1) + \
                 F.l1_loss(pred_z2_raw, z1, reduction='none').mean(dim=-1)
        
        pit_loss = torch.min(loss_A, loss_B)
        
        # 2. The Spread Penalty (Magnitude Matching)
        true_spread = torch.norm(z1 - z2, p=2, dim=-1)
        pred_spread = torch.norm(pred_z1_raw - pred_z2_raw, p=2, dim=-1)
        
        spread_loss = F.mse_loss(pred_spread, true_spread, reduction='none')
        
        # Conditional step: activate spread_loss ONLY at the last timestep (t == T)
        is_last_t = (t == self.num_timesteps).float()
        lambda_spread = 0.1
        
        # 4. Total Loss Calculation
        total_loss = pit_loss + (lambda_spread * spread_loss * is_last_t)
        
        return total_loss.mean()

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        device = c.device
        b = c.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = c.clone()
        
        prev_pred_z1 = None
        for t in tqdm(timesteps, desc='TACOs Sampling', leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            pred_raw = self.model(x_t, t_batch)
            pred_raw_1, pred_raw_2 = pred_raw.chunk(2, dim=-1)
            
            if prev_pred_z1 is None:
                pred_z1 = pred_raw_1
            else:
                dist_1 = F.l1_loss(pred_raw_1, prev_pred_z1, reduction='none').mean(dim=-1)
                dist_2 = F.l1_loss(pred_raw_2, prev_pred_z1, reduction='none').mean(dim=-1)
                
                swap_mask = dist_2 < dist_1
                pred_z1 = torch.where(swap_mask.unsqueeze(-1), pred_raw_2, pred_raw_1)
            
            pred_z2 = self.SQRT_2 * c - pred_z1
            prev_pred_z1 = pred_z1
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            deg_t = self.degrade(pred_z1, pred_z2, t_batch)
            deg_t_prev = self.degrade(pred_z1, pred_z2, t_prev_batch)
            
            x_t = x_t - deg_t + deg_t_prev
            
        final_z1 = x_t
        final_z2 = self.SQRT_2 * c - final_z1
        return final_z1, final_z2

# ==========================================
# 4. Early Stopping Tracker Engine
# ==========================================
class EarlyStopping:
    """Monitors validation improvement metrics and stops training when stuck."""
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0  
        else:
            self.counter += 1
            print(f" EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# ==========================================
# 5. Training & Validation Mechanics
# ==========================================
def train_cold_demorph(diffae_path_str: str, run_name: str, num_timesteps: int = 300, epochs: int = 150, patience: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Train_Loss", "Val_Cheap_Loss", "Val_Last_T_Loss", "Val_TACOs_Reconstruct_L1"])
            
    train_embs = load_split_and_normalize(diffae_path_str, "train")
    test_pairs_embs = load_split_and_normalize(diffae_path_str, "test")
    
    train_loader = DataLoader(ColdDiffAEDemorphTrainDataset(train_embs, epoch_size=1_000_000), batch_size=16_384, shuffle=True, num_workers=8)
    val_loader = DataLoader(ColdDiffAEDemorphTestPairsDataset(test_pairs_embs), batch_size=1000, shuffle=False, num_workers=4)

    net = ColdDemorphNet().to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    early_stopper = EarlyStopping(patience=patience, min_delta=1e-5)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4, "batch_size": 16_384, "num_layers": 10, "hidden_dim": 2048, "num_timesteps": num_timesteps, "early_stop_patience": patience
    })

    best_val_loss = float("inf")
    SQRT_05 = math.sqrt(0.5)
    
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
        val_last_t_loss = 0.0
        val_tacos_loss = None
        
        with torch.no_grad():
            for batch_z1, batch_z2 in val_loader:
                batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                val_cheap_loss += diffusion.compute_loss(batch_z1, batch_z2).item()
                
                fixed_t = torch.full((batch_z1.shape[0],), diffusion.num_timesteps, device=device, dtype=torch.long)
                val_last_t_loss += diffusion.compute_loss(batch_z1, batch_z2, t=fixed_t).item()

        avg_val_cheap_loss = val_cheap_loss / len(val_loader)
        avg_val_last_t_loss = val_last_t_loss / len(val_loader)

        if (epoch + 1) % 25 == 0 or (epoch + 1) == epochs:
            val_tacos_loss_total = 0.0
            with torch.no_grad():
                for batch_z1, batch_z2 in val_loader:
                    batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                    batch_c_vp = SQRT_05 * batch_z1 + SQRT_05 * batch_z2
                    pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c_vp)
                    
                    dist_a = F.l1_loss(pred_z1, batch_z1, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z2, reduction='none').mean(dim=1)
                    dist_b = F.l1_loss(pred_z1, batch_z2, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z1, reduction='none').mean(dim=1)
                    val_tacos_loss_total += torch.min(dist_a, dist_b).mean().item()
            val_tacos_loss = val_tacos_loss_total / len(val_loader)
            
        log_dict = {
            "epoch": epoch + 1, 
            "train_loss": avg_train_loss, 
            "val_cheap_loss": avg_val_cheap_loss,
            "val_last_t_loss": avg_val_last_t_loss
        }
        if val_tacos_loss is not None: log_dict["val_tacos_reconstruct_l1"] = val_tacos_loss
        wandb.log(log_dict)
        
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, 
                f"{avg_train_loss:.6f}", 
                f"{avg_val_cheap_loss:.6f}", 
                f"{avg_val_last_t_loss:.6f}",
                f"{val_tacos_loss:.6f}" if val_tacos_loss is not None else "N/A"
            ])
            
        checkpoint_data = {'epoch': epoch + 1, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint_data, ckpt_dir / "last.pt")
        
        if avg_val_cheap_loss < best_val_loss:
            best_val_loss = avg_val_cheap_loss
            torch.save(checkpoint_data, ckpt_dir / "best.pt")
            
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Cheap Loss: {avg_val_cheap_loss:.4f} | Val Last-T Loss: {avg_val_last_t_loss:.4f}" + 
            (f" | Val TACOs L1: {val_tacos_loss:.4f}" if val_tacos_loss is not None else "")
        )

        if early_stopper(avg_val_cheap_loss):
            print(f"\n[EARLY STOPPING TRIGGERED] Validation profile plateaued for {patience} epochs. Terminating run.")
            break

    wandb.finish()

# ==========================================
# 6. Evaluation Script
# ==========================================
def evaluate_cold_demorph(diffae_path_str: str, run_name: str, num_timesteps: int = 300, mode: str = 'iterative'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / f"eval_{mode}.txt"
    
    test_pairs_embs = load_split_and_normalize(diffae_path_str, "test")
    test_loader = DataLoader(ColdDiffAEDemorphTestPairsDataset(test_pairs_embs), batch_size=1000, shuffle=False, num_workers=4)

    net = ColdDemorphNet().to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    net.eval()

    SQRT_05 = math.sqrt(0.5)
    SQRT_2 = math.sqrt(2.0)

    with torch.no_grad():
        for batch_z1, batch_z2 in test_loader:
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            
            batch_c_vp = SQRT_05 * batch_z1 + SQRT_05 * batch_z2 
            batch_c_true = 0.5 * (batch_z1 + batch_z2)           
            
            if mode == 'iterative':
                pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c_vp)
            else:
                pred = net(batch_c_vp, torch.full((batch_z1.shape[0],), num_timesteps, device=device).long())
                pred_raw_1, _ = pred.chunk(2, dim=-1)
                
                pred_z1 = pred_raw_1
                pred_z2 = SQRT_2 * batch_c_vp - pred_z1

            # --- Permutation-Invariant Alignment Logic ---
            dist_a = F.l1_loss(pred_z1, batch_z1, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z2, reduction='none').mean(dim=1)
            dist_b = F.l1_loss(pred_z1, batch_z2, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z1, reduction='none').mean(dim=1)
            mask_a = (dist_a <= dist_b).unsqueeze(-1)
            
            aligned_pred_z1 = torch.where(mask_a, pred_z1, batch_z2)
            aligned_pred_z2 = torch.where(mask_a, pred_z2, batch_z1)

            # 1. Predictions vs Ground-Truth Alignment
            l1_gt_1 = F.l1_loss(aligned_pred_z1, batch_z1).item()
            l1_gt_2 = F.l1_loss(aligned_pred_z2, batch_z2).item()
            cos_gt_1 = F.cosine_similarity(aligned_pred_z1, batch_z1, dim=-1).mean().item()
            cos_gt_2 = F.cosine_similarity(aligned_pred_z2, batch_z2, dim=-1).mean().item()

            # 2. Alignment to True Average Mixture (c_true) Metrics
            l1_pred1_to_c = F.l1_loss(aligned_pred_z1, batch_c_true).item()
            l1_pred2_to_c = F.l1_loss(aligned_pred_z2, batch_c_true).item()
            cos_pred1_to_c = F.cosine_similarity(aligned_pred_z1, batch_c_true, dim=-1).mean().item()
            cos_pred2_to_c = F.cosine_similarity(aligned_pred_z2, batch_c_true, dim=-1).mean().item()
            
            ref_l1_z1_to_c = F.l1_loss(batch_z1, batch_c_true).item()
            ref_l1_z2_to_c = F.l1_loss(batch_z2, batch_c_true).item()
            ref_cos_z1_to_c = F.cosine_similarity(batch_z1, batch_c_true, dim=-1).mean().item()
            ref_cos_z2_to_c = F.cosine_similarity(batch_z2, batch_c_true, dim=-1).mean().item()

            # 3. Inter-relationship Spread Metrics (p1 vs p2)
            l1_inter_pred = F.l1_loss(aligned_pred_z1, aligned_pred_z2).item()
            cos_inter_pred = F.cosine_similarity(aligned_pred_z1, aligned_pred_z2, dim=-1).mean().item()
            
            ref_l1_inter_gt = F.l1_loss(batch_z1, batch_z2).item()
            ref_cos_inter_gt = F.cosine_similarity(batch_z1, batch_z2, dim=-1).mean().item()

    results_text = (
        f"--- Evaluation Results: {mode.upper()} ---\nRun Name: {run_name}\n----------------------------------------\n"
        f"[1. Prediction to Ground-Truth Alignment]\n"
        f"  - Embedding 1 -> L1 Distance: {l1_gt_1:.6f} | Cosine Similarity: {cos_gt_1:.6f}\n"
        f"  - Embedding 2 -> L1 Distance: {l1_gt_2:.6f} | Cosine Similarity: {cos_gt_2:.6f}\n\n"
        f"[2. Proximity to True Average Mixture (c)]\n"
        f"  - Predicted 1 to c -> L1 Distance: {l1_pred1_to_c:.6f} | Cosine Similarity: {cos_pred1_to_c:.6f}\n"
        f"  - Baseline GT 1 to c -> L1 Distance: {ref_l1_z1_to_c:.6f} | Cosine Similarity: {ref_cos_z1_to_c:.6f}\n"
        f"  - Predicted 2 to c -> L1 Distance: {l1_pred2_to_c:.6f} | Cosine Similarity: {cos_pred2_to_c:.6f}\n"
        f"  - Baseline GT 2 to c -> L1 Distance: {ref_l1_z2_to_c:.6f} | Cosine Similarity: {ref_cos_z2_to_c:.6f}\n\n"
        f"[3. Predicted Outputs Inter-Relationship / Spread]\n"
        f"  - Inter-Predicted (p1 vs p2) -> L1: {l1_inter_pred:.6f} | Cosine Similarity: {cos_inter_pred:.6f}\n"
        f"  - Inter-Ground-Truth (z1 vs z2) -> L1: {ref_l1_inter_gt:.6f} | Cosine Similarity: {ref_cos_inter_gt:.6f}\n"
    )
    
    print("\n" + results_text)
    with open(out_file_path, "w") as f: 
        f.write(results_text)

if __name__ == "__main__":
    BASE_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    RUN_NAME = "diffae_spreadLoss0.1_finalTS"
    
    train_cold_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME, num_timesteps=300, epochs=150, patience=20)
    evaluate_cold_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME, num_timesteps=300, mode='one_shot')
    evaluate_cold_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME, num_timesteps=300, mode='iterative')

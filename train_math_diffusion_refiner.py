import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. Dataset & Data Loading
# ==========================================
class ConditionalMorphDataset(Dataset):
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

        # Symmetric Training
        if torch.rand(1).item() > 0.5:
            condition, target = A, B
        else:
            condition, target = B, A

        return (
            torch.tensor(M, dtype=torch.float32),
            torch.tensor(condition, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )

# ==========================================
# 2. Diffusion Network Architecture
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
    def __init__(self, in_dim, hidden_dim, cond_dim):
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
        joint_cond_dim = time_emb_dim + x_dim
        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(x_dim, hidden_dim, joint_cond_dim))
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + x_dim, hidden_dim, joint_cond_dim))
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t, t, v_cond):
        t_emb = self.time_mlp(t)
        joint_cond = torch.cat([t_emb, v_cond], dim=-1) # v_cond will be the Morph M
        h = x_t
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, joint_cond)
            else:
                h = block(torch.cat([h, x_t], dim=-1), joint_cond)
        return self.final_linear(h)

# ==========================================
# 3. Math-Baseline 10-Step Cold Diffusion
# ==========================================
class MathBaselineColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=10):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def compute_loss(self, M, condition, target, t=None):
        b = M.shape[0]
        if t is None:
            t = torch.randint(1, self.num_timesteps + 1, (b,), device=M.device).long()

        # 1. Simple Inverse Linear Interpolation to get x_T anchor
        z_coarse = (2.0 * M) - condition
        
        # 2. Degrade target towards z_coarse (the faulty prediction)
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        x_t = gamma * z_coarse + (1 - gamma) * target
        
        # 3. Predict target using x_t, conditioned on the original Morph M
        pred_target = self.model(x_t, t, M)
        
        return F.mse_loss(pred_target, target)

    @torch.no_grad()
    def sample_loop(self, M, condition):
        device = M.device
        b = M.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        
        # 1. Start the denoising process exactly at the math-based faulty prediction
        z_coarse = (2.0 * M) - condition
        x_t = z_coarse.clone()

        for t in timesteps:
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Predict clean target using the Morph M as the guiding condition
            pred_target = self.model(x_t, t_batch, M)
            
            # Cold diffusion step update along the z_coarse -> target trajectory
            gamma_t = (t_batch / self.num_timesteps).view(-1, 1).float()
            gamma_prev = ((t_batch - 1) / self.num_timesteps).view(-1, 1).float()
            
            deg_t = gamma_t * z_coarse + (1 - gamma_t) * pred_target
            deg_t_prev = gamma_prev * z_coarse + (1 - gamma_prev) * pred_target

            x_t = x_t - deg_t + deg_t_prev
            
        return x_t

# ==========================================
# 4. Early Stopping
# ==========================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# ==========================================
# 5. Training Loop
# ==========================================
def train_math_diffusion(data_dir: str, run_name: str, num_timesteps=10, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    root = Path(data_dir)

    print("Loading data...")
    train_bf = np.load(root / "train_bonafide_zsem.npy").astype(np.float32)
    train_morphs = np.load(root / "train_morph_zsem.npy").astype(np.float32)
    
    global_mean = train_bf.mean(axis=0, keepdims=True)
    global_std = train_bf.std(axis=0, keepdims=True) + 1e-8
    
    train_bf = (train_bf - global_mean) / global_std
    train_morphs = (train_morphs - global_mean) / global_std
    
    eval_bf = np.load(root / "eval_bonafide_zsem.npy").astype(np.float32)
    eval_morphs = np.load(root / "eval_morph_zsem.npy").astype(np.float32)
    eval_bf = (eval_bf - global_mean) / global_std
    eval_morphs = (eval_morphs - global_mean) / global_std

    train_loader = DataLoader(ConditionalMorphDataset(train_morphs, train_bf, root / "train_morph_metadata.csv"), batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(ConditionalMorphDataset(eval_morphs, eval_bf, root / "eval_morph_metadata.csv"), batch_size=1000, shuffle=False, num_workers=4)

    net = ColdDemorphNet(num_layers=10).to(device)
    diffusion = MathBaselineColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    early_stopper = EarlyStopping(patience=20)

    wandb.init(project="Face-DM-Conditional", name=run_name, config={"timesteps": num_timesteps})
    best_val_loss = float("inf")

    for epoch in range(150):
        net.train()
        train_loss = 0.0
        for M, cond, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            M, cond, tgt = M.to(device), cond.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            loss = diffusion.compute_loss(M, cond, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        net.eval()
        val_loss, sample_l1 = 0.0, 0.0
        with torch.no_grad():
            for M, cond, tgt in val_loader:
                M, cond, tgt = M.to(device), cond.to(device), tgt.to(device)
                val_loss += diffusion.compute_loss(M, cond, tgt).item()
                
            # Sample first batch to track true quality
            b_M, b_cond, b_tgt = next(iter(val_loader))
            b_M, b_cond, b_tgt = b_M.to(device), b_cond.to(device), b_tgt.to(device)
            pred_tgt = diffusion.sample_loop(b_M, b_cond)
            sample_l1 = F.l1_loss(pred_tgt, b_tgt).item()

        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_sample_l1": sample_l1})
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Sample L1: {sample_l1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), ckpt_dir / "math_refiner_best.pt")

        if early_stopper(avg_val_loss):
            print("Early stopping triggered.")
            break

    wandb.finish()

if __name__ == "__main__":
    DATA_DIR = "/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings"
    train_math_diffusion(DATA_DIR, "math_baseline_10step_refiner")

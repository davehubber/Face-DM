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

# --- Include the ConditionalMorphDataset and EarlyStopping classes here (same as Phase 1) ---
from phase1_train_predictor import ConditionalMorphDataset, EarlyStopping, CoarsePredictorNet

# ==========================================
# 1. Diffusion Network Architecture
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
        joint_cond = torch.cat([t_emb, v_cond], dim=-1)
        h = x_t
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, joint_cond)
            else:
                h = block(torch.cat([h, x_t], dim=-1), joint_cond)
        return self.final_linear(h)

# ==========================================
# 2. 10-Step Cold Diffusion Logic
# ==========================================
class DeterministicColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=10): # Reduced to 10!
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, M, target, t):
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        return gamma * M + (1 - gamma) * target

    def compute_loss(self, M, z_coarse, target, t=None):
        b = M.shape[0]
        if t is None:
            t = torch.randint(1, self.num_timesteps + 1, (b,), device=M.device).long()

        # Degrade target towards Morph M
        x_t = self.degrade(M, target, t)
        
        # Predict target using x_t AND the coarse prediction as condition
        pred_target = self.model(x_t, t, z_coarse)
        return F.mse_loss(pred_target, target)

    @torch.no_grad()
    def sample_loop(self, M, z_coarse):
        device = M.device
        b = M.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = M.clone()

        for t in timesteps:
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            pred_target = self.model(x_t, t_batch, z_coarse)
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            deg_t = self.degrade(M, pred_target, t_batch)
            deg_t_prev = self.degrade(M, pred_target, t_prev_batch)

            x_t = x_t - deg_t + deg_t_prev
        return x_t

# ==========================================
# 3. Training Loop
# ==========================================
def train_phase2(data_dir: str, predictor_ckpt: str, run_name: str, num_timesteps=10, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    root = Path(data_dir)

    print("Loading data...")
    # --- Add the same data loading and normalization logic as Phase 1 here ---
    train_bf = np.load(root / "train_bonafide_zsem.npy").astype(np.float32)
    train_morphs = np.load(root / "train_morph_zsem.npy").astype(np.float32)
    global_mean = train_bf.mean(axis=0, keepdims=True)
    global_std = train_bf.std(axis=0, keepdims=True) + 1e-8
    train_bf = (train_bf - global_mean) / global_std
    train_morphs = (train_morphs - global_mean) / global_std
    
    eval_bf = np.load(root / "eval_bonafide_zsem.npy").astype(np.float32)
    eval_morphs = np.load(root / "eval_morph_zsem_1000.npy").astype(np.float32)
    eval_bf = (eval_bf - global_mean) / global_std
    eval_morphs = (eval_morphs - global_mean) / global_std

    train_loader = DataLoader(ConditionalMorphDataset(train_morphs, train_bf, root / "train_morph_metadata.csv"), batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(ConditionalMorphDataset(eval_morphs, eval_bf, root / "eval_morph_metadata_1000.csv"), batch_size=1000, shuffle=False, num_workers=4)

    # Load Frozen Predictor
    predictor = CoarsePredictorNet().to(device)
    predictor.load_state_dict(torch.load(predictor_ckpt, map_location=device))
    predictor.eval() # Freeze it

    net = ColdDemorphNet(num_layers=8).to(device) # Slightly smaller since the predictor did the heavy lifting
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    early_stopper = EarlyStopping(patience=15)

    wandb.init(project="Face-DM-Phase2", name=run_name, config={"timesteps": num_timesteps})
    best_val_loss = float("inf")

    for epoch in range(150):
        net.train()
        train_loss = 0.0
        for M, cond, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            M, cond, tgt = M.to(device), cond.to(device), tgt.to(device)
            
            # Step 1: Get coarse prediction (No gradients tracked here)
            with torch.no_grad():
                z_coarse = predictor(M, cond)

            # Step 2: Train diffusion model
            optimizer.zero_grad()
            loss = diffusion.compute_loss(M, z_coarse, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        net.eval()
        val_loss, sample_l1 = 0.0, 0.0
        with torch.no_grad():
            for M, cond, tgt in val_loader:
                M, cond, tgt = M.to(device), cond.to(device), tgt.to(device)
                z_coarse = predictor(M, cond)
                val_loss += diffusion.compute_loss(M, z_coarse, tgt).item()
                
            # Sample first batch to track true quality
            b_M, b_cond, b_tgt = next(iter(val_loader))
            b_M, b_cond, b_tgt = b_M.to(device), b_cond.to(device), b_tgt.to(device)
            b_z_coarse = predictor(b_M, b_cond)
            pred_tgt = diffusion.sample_loop(b_M, b_z_coarse)
            sample_l1 = F.l1_loss(pred_tgt, b_tgt).item()

        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_sample_l1": sample_l1})
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Sample L1: {sample_l1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), ckpt_dir / "refiner_best.pt")

        if early_stopper(avg_val_loss):
            print("Early stopping triggered.")
            break

    wandb.finish()

if __name__ == "__main__":
    DATA_DIR = "/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings"
    PREDICTOR_CKPT = "experiments/phase1_coarse_predictor/checkpoints/predictor_best.pt"
    train_phase2(DATA_DIR, PREDICTOR_CKPT, "phase2_10step_refiner")

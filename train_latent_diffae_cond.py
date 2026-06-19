import math
from pathlib import Path
import csv
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
class ConditionalMorphDataset(Dataset):
    """
    Loads Real Morphs and their Bona Fide components.
    Implements symmetric training (randomly swapping condition and target).
    """
    def __init__(self, morph_embs: np.ndarray, bf_embs: np.ndarray, metadata_path: Path):
        self.morph_embs = morph_embs
        self.bf_embs = bf_embs
        self.records = []
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx_m = int(row['embedding_index'])
                
                # Depending on whether this is train or eval, the column name changes slightly
                col_a = 'source_idx_A_in_train_bf' if 'source_idx_A_in_train_bf' in row else 'source_idx_A_in_eval_bf'
                col_b = 'source_idx_B_in_train_bf' if 'source_idx_B_in_train_bf' in row else 'source_idx_B_in_eval_bf'
                
                idx_a = int(row[col_a])
                idx_b = int(row[col_b])
                
                # Only include morphs where both bona fide sources exist in the dataset
                if idx_a >= 0 and idx_b >= 0:
                    self.records.append((idx_m, idx_a, idx_b))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        idx_m, idx_a, idx_b = self.records[idx]
        
        M = self.morph_embs[idx_m]
        A = self.bf_embs[idx_a]
        B = self.bf_embs[idx_b]

        # SYMMETRIC TRAINING: 50% chance to condition on A and target B, or vice versa.
        # This replaces the old "Smiling Score" hack entirely.
        if torch.rand(1).item() > 0.5:
            condition = A
            target = B
        else:
            condition = B
            target = A

        return (
            torch.tensor(M, dtype=torch.float32),
            torch.tensor(condition, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


# ==========================================
# 2. Network Architecture (Joint AdaLN)
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
        # cond_dim is now (time_emb_dim + x_dim). It projects to hidden_dim * 2 (for scale & shift)
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

        # The global joint condition dimension
        joint_cond_dim = time_emb_dim + x_dim

        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(x_dim, hidden_dim, joint_cond_dim))
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + x_dim, hidden_dim, joint_cond_dim))

        # We now only predict ONE embedding (the target identity)
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t, t, v_cond):
        t_emb = self.time_mlp(t)
        
        # GLOBAL CONCATENATION: Combine time and reference vector once
        joint_cond = torch.cat([t_emb, v_cond], dim=-1)
        
        h = x_t
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, joint_cond)
            else:
                h = block(torch.cat([h, x_t], dim=-1), joint_cond)
                
        return self.final_linear(h)


# ==========================================
# 3. Cold Diffusion Process (Real-Morph Trajectory)
# ==========================================
class DeterministicColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=300):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, M, target, t):
        """
        Interpolates between the clean target (t=0) and the real morphed mixture (t=T).
        x_T = M
        x_0 = target
        """
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        return gamma * M + (1 - gamma) * target

    def compute_loss(self, M, condition, target, t=None):
        b = M.shape[0]
        if t is None:
            t = torch.randint(1, self.num_timesteps + 1, (b,), device=M.device).long()

        # Step 1: Degrade target towards Morph M
        x_t = self.degrade(M, target, t)
        
        # Step 2: Predict target using x_t and the condition identity
        pred_target = self.model(x_t, t, condition)

        return F.l1_loss(pred_target, target)

    @torch.no_grad()
    def sample_loop(self, M, condition):
        """Iterative denoising to pull the target out of the Morph."""
        device = M.device
        b = M.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        
        # Start at fully degraded state (the Morph)
        x_t = M.clone()

        for t in tqdm(timesteps, desc="Sampling", leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Predict clean target
            pred_target = self.model(x_t, t_batch, condition)

            # Cold diffusion step update
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            deg_t = self.degrade(M, pred_target, t_batch)
            deg_t_prev = self.degrade(M, pred_target, t_prev_batch)

            x_t = x_t - deg_t + deg_t_prev

        # x_0 is the recovered target
        return x_t


# ==========================================
# 4. Early Stopping
# ==========================================
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
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
# 5. Training Loop
# ==========================================
def train_conditional_demorph(
    data_dir: str,
    run_name: str,
    num_timesteps: int = 300,
    epochs: int = 150,
    patience: int = 20,
    batch_size: int = 8192
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    root = Path(data_dir)

    print("Loading datasets into memory...")
    # Load Training Data
    train_bf = np.load(root / "train_bonafide_zsem.npy").astype(np.float32)
    train_morphs = np.load(root / "train_morph_zsem.npy").astype(np.float32)
    
    # Compute normalizers dynamically from Bona Fide distribution
    global_mean = train_bf.mean(axis=0, keepdims=True)
    global_std = train_bf.std(axis=0, keepdims=True) + 1e-8
    
    train_bf = (train_bf - global_mean) / global_std
    train_morphs = (train_morphs - global_mean) / global_std
    
    # Load Evaluation Data
    eval_bf = np.load(root / "eval_bonafide_zsem.npy").astype(np.float32)
    eval_morphs = np.load(root / "eval_morph_zsem_1000.npy").astype(np.float32)
    
    eval_bf = (eval_bf - global_mean) / global_std
    eval_morphs = (eval_morphs - global_mean) / global_std

    train_dataset = ConditionalMorphDataset(train_morphs, train_bf, root / "train_morph_metadata.csv")
    eval_dataset = ConditionalMorphDataset(eval_morphs, eval_bf, root / "eval_morph_metadata_1000.csv")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(eval_dataset, batch_size=1000, shuffle=False, num_workers=4)

    net = ColdDemorphNet(x_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    early_stopper = EarlyStopping(patience=patience)

    wandb.init(
        project="Face-DM-Conditional",
        name=run_name,
        dir=str(exp_dir),
        config={"batch_size": batch_size, "num_layers": 10, "timesteps": num_timesteps}
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0

        for batch_M, batch_cond, batch_tgt in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            batch_M = batch_M.to(device)
            batch_cond = batch_cond.to(device)
            batch_tgt = batch_tgt.to(device)

            optimizer.zero_grad()
            loss = diffusion.compute_loss(batch_M, batch_cond, batch_tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_M, batch_cond, batch_tgt in val_loader:
                batch_M = batch_M.to(device)
                batch_cond = batch_cond.to(device)
                batch_tgt = batch_tgt.to(device)
                
                # Validation loss with randomly drawn timesteps
                val_loss += diffusion.compute_loss(batch_M, batch_cond, batch_tgt).item()

        avg_val_loss = val_loss / len(val_loader)
        
        # Periodic full sample evaluation
        sample_l1 = 0.0
        if (epoch + 1) % 25 == 0 or (epoch + 1) == epochs:
            with torch.no_grad():
                # We only sample the first batch to save time during train loop
                b_M, b_cond, b_tgt = next(iter(val_loader))
                b_M, b_cond, b_tgt = b_M.to(device), b_cond.to(device), b_tgt.to(device)
                pred_tgt = diffusion.sample_loop(b_M, b_cond)
                sample_l1 = F.l1_loss(pred_tgt, b_tgt).item()

        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_sample_L1": sample_l1})

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} " + 
              (f"| Sample L1: {sample_l1:.4f}" if sample_l1 > 0 else ""))

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
# 6. Evaluation Script
# ==========================================
def evaluate_conditional_demorph(data_dir: str, run_name: str, num_timesteps: int = 300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    
    root = Path(data_dir)
    train_bf = np.load(root / "train_bonafide_zsem.npy").astype(np.float32)
    global_mean = train_bf.mean(axis=0, keepdims=True)
    global_std = train_bf.std(axis=0, keepdims=True) + 1e-8

    eval_bf = np.load(root / "eval_bonafide_zsem.npy").astype(np.float32)
    eval_morphs = np.load(root / "eval_morph_zsem_1000.npy").astype(np.float32)
    
    eval_bf = (eval_bf - global_mean) / global_std
    eval_morphs = (eval_morphs - global_mean) / global_std

    records = []
    with open(root / "eval_morph_metadata_1000.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_m = int(row['embedding_index'])
            # FIXED: Column names updated to match the new encoding pipeline
            idx_a = int(row['source_idx_A_in_train_bf'])
            idx_b = int(row['source_idx_B_in_train_bf'])
            if idx_a >= 0 and idx_b >= 0:
                records.append((idx_m, idx_a, idx_b))

    net = ColdDemorphNet().to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    net.eval()
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)

    total_l1 = 0
    total_cos = 0
    
    with torch.no_grad():
        # Iterate manually to compute exact metrics
        batch_size = 500
        for i in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
            batch_records = records[i:i+batch_size]
            
            b_M = torch.tensor(np.array([eval_morphs[r[0]] for r in batch_records]), device=device)
            # Conditioning on A, targeting B
            b_cond_A = torch.tensor(np.array([eval_bf[r[1]] for r in batch_records]), device=device)
            b_tgt_B = torch.tensor(np.array([eval_bf[r[2]] for r in batch_records]), device=device)
            
            pred_B = diffusion.sample_loop(b_M, b_cond_A)
            
            total_l1 += F.l1_loss(pred_B, b_tgt_B, reduction="sum").item()
            total_cos += F.cosine_similarity(pred_B, b_tgt_B, dim=-1).sum().item()

    avg_l1 = total_l1 / (len(records) * 512) if len(records) > 0 else float('inf')
    avg_cos = total_cos / len(records) if len(records) > 0 else 0.0

    results = (
        f"--- CONDITIONAL EVALUATION ---\n"
        f"Recovered Target L1 Distance : {avg_l1:.6f}\n"
        f"Recovered Target Cosine Sim  : {avg_cos:.6f}\n"
    )
    print("\n" + results)
    with open(exp_dir / "eval_metrics.txt", "w") as f:
        f.write(results)


if __name__ == "__main__":
    DATA_DIR = "/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings"
    RUN_NAME = "diffae_conditional_baseline"

    # train_conditional_demorph(data_dir=DATA_DIR, run_name=RUN_NAME)
    evaluate_conditional_demorph(data_dir=DATA_DIR, run_name=RUN_NAME)

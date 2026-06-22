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

        # Symmetric Training (Combinatorial Augmentation)
        if torch.rand(1).item() > 0.5:
            condition, target = A, B
        else:
            condition, target = B, A

        return (
            torch.tensor(M, dtype=torch.float32),
            torch.tensor(condition, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


def resolve_existing_path(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of these files were found: " + ", ".join(str(root / name) for name in names)
    )

# ==========================================
# 2. UNCONDITIONED Diffusion Network
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
        
        # The condition dimension is now STRICTLY the time embedding
        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(x_dim, hidden_dim, time_emb_dim))
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + x_dim, hidden_dim, time_emb_dim))
            
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t, t):
        t_emb = self.time_mlp(t)
        h = x_t
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, t_emb)
            else:
                h = block(torch.cat([h, x_t], dim=-1), t_emb)
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
        
        # 2. Degrade target towards z_coarse using variance-preserving trajectory
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        x_t = gamma * z_coarse + (1 - gamma) * target
        
        # 3. Predict target using x_t, unconditioned (only time)
        pred_target = self.model(x_t, t)
        
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
            
            # Predict clean target unconditioned
            pred_target = self.model(x_t, t_batch)
            
            # Cold diffusion step update using variance-preserving trajectory
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
def train_math_diffusion_unconditioned(data_dir: str, run_name: str, num_timesteps=10, batch_size=256):
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
    val_loader = DataLoader(ConditionalMorphDataset(eval_morphs, eval_bf, root / "eval_morph_metadata.csv"), batch_size=batch_size, shuffle=False, num_workers=4)

    net = ColdDemorphNet(num_layers=10).to(device)
    diffusion = MathBaselineColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    early_stopper = EarlyStopping(patience=20)

    wandb.init(project="Face-DM-Conditional", name=run_name, config={"timesteps": num_timesteps, "batch_size": batch_size})
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
                
            b_M, b_cond, b_tgt = next(iter(val_loader))
            b_M, b_cond, b_tgt = b_M.to(device), b_cond.to(device), b_tgt.to(device)
            pred_tgt = diffusion.sample_loop(b_M, b_cond)
            sample_l1 = F.l1_loss(pred_tgt, b_tgt).item()

        avg_val_loss = val_loss / len(val_loader)
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_sample_l1": sample_l1})
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Sample L1: {sample_l1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), ckpt_dir / "best.pt")

        if early_stopper(avg_val_loss):
            print("Early stopping triggered.")
            break

    wandb.finish()

# ==========================================
# 6. Evaluation Script
# ==========================================
def evaluate_math_diffusion_unconditioned(data_dir: str, run_name: str, num_timesteps: int = 10, mode: str = 'iterative'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / f"eval_metrics_{mode}.txt"
    
    root = Path(data_dir)
    train_bf = np.load(root / "train_bonafide_zsem.npy").astype(np.float32)
    global_mean = train_bf.mean(axis=0, keepdims=True)
    global_std = train_bf.std(axis=0, keepdims=True) + 1e-8

    eval_bf = np.load(root / "eval_bonafide_zsem.npy").astype(np.float32)
    eval_morphs = np.load(resolve_existing_path(root, "eval_morph_zsem.npy", "eval_morph_zsem_1000.npy")).astype(np.float32)
    
    eval_bf = (eval_bf - global_mean) / global_std
    eval_morphs = (eval_morphs - global_mean) / global_std

    records = []
    with open(resolve_existing_path(root, "eval_morph_metadata.csv", "eval_morph_metadata_1000.csv"), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_m = int(row['embedding_index'])
            col_a = 'source_idx_A_in_train_bf' if 'source_idx_A_in_train_bf' in row else 'source_idx_A_in_eval_bf'
            col_b = 'source_idx_B_in_train_bf' if 'source_idx_B_in_train_bf' in row else 'source_idx_B_in_eval_bf'
            idx_a = int(row[col_a])
            idx_b = int(row[col_b])
            if idx_a >= 0 and idx_b >= 0:
                records.append((idx_m, idx_a, idx_b))

    net = ColdDemorphNet(num_layers=10).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    diffusion = MathBaselineColdDemorph(net, num_timesteps=num_timesteps).to(device)

    total_l1_a_to_b = 0
    total_cos_a_to_b = 0
    total_l1_b_to_a = 0
    total_cos_b_to_a = 0
    
    total_S_a_to_b = 0
    total_S_b_to_a = 0
    
    with torch.no_grad():
        batch_size = 500
        for i in tqdm(range(0, len(records), batch_size), desc=f"Evaluating ({mode.upper()})"):
            batch_records = records[i:i+batch_size]
            
            b_M = torch.tensor(np.array([eval_morphs[r[0]] for r in batch_records]), device=device)
            b_A = torch.tensor(np.array([eval_bf[r[1]] for r in batch_records]), device=device)
            b_B = torch.tensor(np.array([eval_bf[r[2]] for r in batch_records]), device=device)
            
            if mode == 'iterative':
                pred_B = diffusion.sample_loop(b_M, b_A)
                pred_A = diffusion.sample_loop(b_M, b_B)
            else:  # 'one_shot'
                t_max = torch.full((b_M.shape[0],), num_timesteps, device=device).long()
                # Unconditioned execution requires structural anchor trajectory initialization
                z_coarse_B = (2.0 * b_M) - b_A
                z_coarse_A = (2.0 * b_M) - b_B
                pred_B = net(z_coarse_B, t_max)
                pred_A = net(z_coarse_A, t_max)
            
            total_l1_a_to_b += F.l1_loss(pred_B, b_B, reduction="sum").item()
            total_cos_a_to_b += F.cosine_similarity(pred_B, b_B, dim=-1).sum().item()
            total_l1_b_to_a += F.l1_loss(pred_A, b_A, reduction="sum").item()
            total_cos_b_to_a += F.cosine_similarity(pred_A, b_A, dim=-1).sum().item()

            # --- Success Rate of Reversal (%S) Metrics ---
            cos_B_gt = F.cosine_similarity(pred_B, b_B, dim=-1)
            cos_B_M = F.cosine_similarity(pred_B, b_M, dim=-1)
            total_S_a_to_b += (cos_B_gt > cos_B_M).float().sum().item()

            cos_A_gt = F.cosine_similarity(pred_A, b_A, dim=-1)
            cos_A_M = F.cosine_similarity(pred_A, b_M, dim=-1)
            total_S_b_to_a += (cos_A_gt > cos_A_M).float().sum().item()

    avg_l1_a_to_b = total_l1_a_to_b / (len(records) * 512) if len(records) > 0 else float('inf')
    avg_cos_a_to_b = total_cos_a_to_b / len(records) if len(records) > 0 else 0.0
    pct_S_a_to_b = (total_S_a_to_b / len(records)) * 100 if len(records) > 0 else 0.0

    avg_l1_b_to_a = total_l1_b_to_a / (len(records) * 512) if len(records) > 0 else float('inf')
    avg_cos_b_to_a = total_cos_b_to_a / len(records) if len(records) > 0 else 0.0
    pct_S_b_to_a = (total_S_b_to_a / len(records)) * 100 if len(records) > 0 else 0.0

    avg_l1 = 0.5 * (avg_l1_a_to_b + avg_l1_b_to_a)
    avg_cos = 0.5 * (avg_cos_a_to_b + avg_cos_b_to_a)
    pct_S_comb = 0.5 * (pct_S_a_to_b + pct_S_b_to_a)

    results = (
        f"--- MATH BASELINE UNCONDITIONED EVALUATION ({mode.upper()}) ---\n"
        f"Condition A -> Recover B L1 Distance : {avg_l1_a_to_b:.6f}\n"
        f"Condition A -> Recover B Cosine Sim  : {avg_cos_a_to_b:.6f}\n"
        f"Condition A -> Recover B %S1         : {pct_S_a_to_b:.2f}%\n"
        f"Condition B -> Recover A L1 Distance : {avg_l1_b_to_a:.6f}\n"
        f"Condition B -> Recover A Cosine Sim  : {avg_cos_b_to_a:.6f}\n"
        f"Condition B -> Recover A %S2         : {pct_S_b_to_a:.2f}%\n"
        f"Mean Conditional L1 Distance         : {avg_l1:.6f}\n"
        f"Mean Conditional Cosine Sim          : {avg_cos:.6f}\n"
        f"Combined Total %S                    : {pct_S_comb:.2f}%\n"
    )
    print("\n" + results)
    with open(out_file_path, "w") as f:
        f.write(results)


if __name__ == "__main__":
    DATA_DIR = "/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings_v2"
    RUN_NAME = "diffae_conditional_refiner"
    TIMESTEPS = 10

    train_math_diffusion_unconditioned(DATA_DIR, RUN_NAME, num_timesteps=TIMESTEPS)
    evaluate_math_diffusion_unconditioned(data_dir=DATA_DIR, run_name=RUN_NAME, num_timesteps=TIMESTEPS, mode='one_shot')
    evaluate_math_diffusion_unconditioned(data_dir=DATA_DIR, run_name=RUN_NAME, num_timesteps=TIMESTEPS, mode='iterative')

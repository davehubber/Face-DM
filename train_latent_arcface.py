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
class ColdArcFaceDemorphDataset(Dataset):
    """
    Dynamically generates pairs of distinct ArcFace embeddings. 
    Operates purely in L2 normalized space without arbitrary scaling.
    """
    def __init__(self, embeddings: np.ndarray, epoch_size: int = 1000000, deterministic: bool = False):
        self.embeddings = embeddings
        self.epoch_size = epoch_size
        self.deterministic = deterministic
        self.num_samples = len(embeddings)
        
        if self.deterministic:
            np.random.seed(42)
            pairs = set()
            while len(pairs) < epoch_size:
                idx1 = np.random.randint(0, self.num_samples)
                idx2 = np.random.randint(0, self.num_samples)
                if idx1 != idx2:
                    pairs.add((idx1, idx2))
            self.pairs = np.array(list(pairs))

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if self.deterministic:
            idx1, idx2 = self.pairs[idx]
        else:
            idx1 = np.random.randint(0, self.num_samples)
            idx2 = np.random.randint(0, self.num_samples)
            while idx1 == idx2:
                idx2 = np.random.randint(0, self.num_samples)

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
    Predicts both clean embeddings simultaneously (output dim = 1024).
    For t < T, the first slot is trained to natively output the dominant embedding.
    """
    def __init__(self, x_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        cond_dim = time_emb_dim
        
        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(x_dim, hidden_dim, cond_dim))
        
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + x_dim, hidden_dim, cond_dim))
            
        self.final_linear = nn.Linear(hidden_dim, x_dim * 2)

    def forward(self, x, t):
        cond = self.time_mlp(t)
        
        h = x
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, cond)
            else:
                h = block(torch.cat([h, x], dim=-1), cond)
                
        pred_z1, pred_z2 = self.final_linear(h).chunk(2, dim=-1)
        return pred_z1, pred_z2

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
        Forward degradation: Reverts to linear interpolation.
        t=0: alpha=1.0 -> purely z1
        t=T: alpha=0.50 -> perfect 50/50 average mixture
        """
        alpha = 1.0 - 0.50 * (t / self.num_timesteps).view(-1, 1).float()
        return alpha * z1 + (1.0 - alpha) * z2

    def compute_loss(self, z1, z2):
        b = z1.shape[0]
        
        # 50% of the batch targets the final symmetric step t=T; the rest targets t < T
        is_terminal = torch.rand(b, device=z1.device) < 0.5
        t_non_terminal = torch.randint(1, self.num_timesteps, (b,), device=z1.device).long()
        t = torch.where(is_terminal, torch.tensor(self.num_timesteps, device=z1.device), t_non_terminal)
        
        x_t = self.degrade(z1, z2, t)
        pred_z1, pred_z2 = self.model(x_t, t)
        
        # --- Loss for Terminal Timestep t=T (Permutation Invariant) ---
        loss_opt_A = F.l1_loss(pred_z1, z1, reduction='none').mean(dim=-1) + F.l1_loss(pred_z2, z2, reduction='none').mean(dim=-1)
        loss_opt_B = F.l1_loss(pred_z1, z2, reduction='none').mean(dim=-1) + F.l1_loss(pred_z2, z1, reduction='none').mean(dim=-1)
        loss_terminal = torch.min(loss_opt_A, loss_opt_B) / 2.0
        
        # --- Loss for Intermediate Timesteps t < T (Strictly targeting z1) ---
        loss_non_terminal = F.l1_loss(pred_z1, z1, reduction='none').mean(dim=-1)
        
        loss = torch.where(is_terminal, loss_terminal, loss_non_terminal).mean()
        return loss

    @torch.no_grad()
    def tacos_sample_loop(self, c_avg, z1=None, z2=None):
        """
        Decoupled dual-path TACOs sampling protocol. Generates two parallel,
        independent tracks optimized by their respective trajectory orientation.
        """
        device = c_avg.device
        b = c_avg.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        
        # Initial step at t=T (Symmetric problem)
        t_T = torch.full((b,), self.num_timesteps, device=device, dtype=torch.long)
        out_z1, out_z2 = self.model(c_avg, t_T)
        
        # Use validation targets to accurately align path references for tracking metrics
        if z1 is not None and z2 is not None:
            match_A = F.l1_loss(out_z1, z1, reduction='none').mean(dim=-1) + F.l1_loss(out_z2, z2, reduction='none').mean(dim=-1)
            match_B = F.l1_loss(out_z1, z2, reduction='none').mean(dim=-1) + F.l1_loss(out_z2, z1, reduction='none').mean(dim=-1)
            swap = (match_B < match_A).unsqueeze(-1)
            z1_init = torch.where(swap, out_z2, out_z1)
            z2_init = torch.where(swap, out_z1, out_z2)
        else:
            z1_init, z2_init = out_z1, out_z2

        # Initialize parallel path states
        x_t = c_avg.clone()  # Path 1: tracking z1
        y_t = c_avg.clone()  # Path 2: tracking z2
        
        for t in timesteps:
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            
            # --- Path 1: Dominant target is z1 ---
            if t == self.num_timesteps:
                p1, p2 = z1_init, z2_init
            else:
                p1, _ = self.model(x_t, t_batch)
                p2 = 2.0 * c_avg - p1
                
            deg_t_1 = self.degrade(p1, p2, t_batch)
            deg_t_prev_1 = self.degrade(p1, p2, t_prev_batch)
            x_t = x_t - deg_t_1 + deg_t_prev_1
            
            # --- Path 2: Dominant target is z2 ---
            if t == self.num_timesteps:
                q2, q1 = z2_init, z1_init
            else:
                # Queries the first slot because z2 acts as the dominant component of this tracking frame
                q2, _ = self.model(y_t, t_batch)
                q1 = 2.0 * c_avg - q2
                
            deg_t_2 = self.degrade(q2, q1, t_batch)
            deg_t_prev_2 = self.degrade(q2, q1, t_prev_batch)
            y_t = y_t - deg_t_2 + deg_t_prev_2
            
        final_z1 = x_t
        final_z2 = y_t
        return final_z1, final_z2

# ==========================================
# 4. Evaluation & Training Loop
# ==========================================
def train_cold_demorph(arcface_path_str: str, run_name: str, num_timesteps: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_L1", "Val_Cheap_L1", "Val_TACOs_z1_L1", "Val_TACOs_z2_L1"])
            
    print("Loading ArcFace embeddings...")
    arcface_path = Path(arcface_path_str).resolve()
    scale_factor = np.sqrt(512)
    arcface_embs = (np.load(arcface_path).astype(np.float32)) * scale_factor

    split_idx = int(len(arcface_embs) * 0.9)
    train_embs, val_embs = arcface_embs[:split_idx], arcface_embs[split_idx:]
    
    train_dataset = ColdArcFaceDemorphDataset(train_embs, epoch_size=1_000_000, deterministic=False)
    val_dataset = ColdArcFaceDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16_384, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=2_048, shuffle=False, num_workers=4)

    net = ColdDemorphNet(x_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 3e-4,
        "batch_size": 16_384,
        "num_layers": 10,
        "hidden_dim": 2048,
        "num_timesteps": num_timesteps,
        "latent_space": "ArcFace",
        "mixture": "Asymmetric training (50% at t=T with PI, 50% at t<T targeting z1), Decoupled Dual-Path Sampling"
    })

    epochs = 100 
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
        
        # ------------------------------------------
        # Validation Loop (Cheap Metric)
        # ------------------------------------------
        net.eval()
        val_cheap_loss = 0.0
        val_tacos_z1_loss = None
        val_tacos_z2_loss = None
        
        with torch.no_grad():
            for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val Cheap]"):
                batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                loss_cheap = diffusion.compute_loss(z1=batch_z1, z2=batch_z2)
                val_cheap_loss += loss_cheap.item()
                
        avg_val_cheap_loss = val_cheap_loss / len(val_loader)

        # ------------------------------------------
        # Validation Loop (Expensive TACOs Metric)
        # ------------------------------------------
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_z1_total = 0.0
            val_z2_total = 0.0
            
            with torch.no_grad():
                for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val TACOs]"):
                    batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                    batch_c_avg = (batch_z1 + batch_z2) / 2.0
                    
                    # Generate predictions using separate independent tracking channels
                    pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c_avg, batch_z1, batch_z2)
                    
                    val_z1_total += F.l1_loss(pred_z1, batch_z1).item()
                    val_z2_total += F.l1_loss(pred_z2, batch_z2).item()
                    
            val_tacos_z1_loss = val_z1_total / len(val_loader)
            val_tacos_z2_loss = val_z2_total / len(val_loader)
            
        # ------------------------------------------
        # Logging & Checkpointing
        # ------------------------------------------
        log_dict = {
            "epoch": epoch + 1,
            "train_l1": avg_train_loss,
            "val_cheap_l1": avg_val_cheap_loss
        }
        if val_tacos_z1_loss is not None:
            log_dict["val_tacos_reconstruct_z1_l1"] = val_tacos_z1_loss
            log_dict["val_tacos_reconstruct_z2_l1"] = val_tacos_z2_loss
            
        wandb.log(log_dict)
        
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{avg_train_loss:.6f}", 
                f"{avg_val_cheap_loss:.6f}", 
                f"{val_tacos_z1_loss:.6f}" if val_tacos_z1_loss is not None else "N/A",
                f"{val_tacos_z2_loss:.6f}" if val_tacos_z2_loss is not None else "N/A"
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
        if val_tacos_z1_loss is not None:
            print(f" | TACOs z1 L1: {val_tacos_z1_loss:.4f} | TACOs z2 L1: {val_tacos_z2_loss:.4f}")
        else:
            print()

    wandb.finish()


def evaluate_cold_demorph(arcface_path_str: str, run_name: str, num_timesteps: int = 10, mode: str = 'iterative'):
    """
    Evaluates both decoupled sampling paths independently, recording isolated metrics for z1 and z2.
    """
    assert mode in ['iterative', 'one_shot'], "Mode must be 'iterative' or 'one_shot'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    out_file_path = exp_dir / f"eval_{mode}.txt"
    
    print(f"Loading ArcFace embeddings for {mode} evaluation...")
    scale_factor = math.sqrt(512)
    arcface_embs = (np.load(Path(arcface_path_str).resolve()).astype(np.float32)) * scale_factor
    split_idx = int(len(arcface_embs) * 0.9)
    val_embs = arcface_embs[split_idx:]
    
    val_dataset = ColdArcFaceDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    val_loader = DataLoader(val_dataset, batch_size=2_048, shuffle=False, num_workers=4)

    net = ColdDemorphNet(x_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # Track isolated performance metrics for each path
    metrics = {
        'z1': {'l1_scaled': 0.0, 'l1_raw': 0.0, 'cosine': 0.0},
        'z2': {'l1_scaled': 0.0, 'l1_raw': 0.0, 'cosine': 0.0}
    }
    num_batches = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Evaluating ({mode})"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c_avg = (batch_z1 + batch_z2) / 2.0
            b = batch_z1.shape[0]
            
            # --- 1. Generation ---
            if mode == 'iterative':
                pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c_avg, batch_z1, batch_z2)
            else: # one_shot
                t_T = torch.full((b,), num_timesteps, device=device, dtype=torch.long)
                out_z1, out_z2 = net(batch_c_avg, t_T)
                # Symmetrical alignment matching for one-shot evaluation baseline
                match_A = F.l1_loss(out_z1, batch_z1, reduction='none').mean(dim=-1) + F.l1_loss(out_z2, batch_z2, reduction='none').mean(dim=-1)
                match_B = F.l1_loss(out_z1, batch_z2, reduction='none').mean(dim=-1) + F.l1_loss(out_z2, batch_z1, reduction='none').mean(dim=-1)
                swap = (match_B < match_A).unsqueeze(-1)
                pred_z1 = torch.where(swap, out_z2, out_z1)
                pred_z2 = torch.where(swap, out_z1, out_z2)

            # --- 2. Compute Target-Isolated Performance Indicators ---
            # Normalized Hypersphere Conversions
            pred_z1_raw = F.normalize(pred_z1 / scale_factor, p=2, dim=-1)
            pred_z2_raw = F.normalize(pred_z2 / scale_factor, p=2, dim=-1)
            true_z1_raw = F.normalize(batch_z1 / scale_factor, p=2, dim=-1)
            true_z2_raw = F.normalize(batch_z2 / scale_factor, p=2, dim=-1)
            
            # Accumulate Path 1 (z1) Metrics
            metrics['z1']['l1_scaled'] += F.l1_loss(pred_z1, batch_z1).item()
            metrics['z1']['l1_raw'] += F.l1_loss(pred_z1_raw, true_z1_raw).item()
            metrics['z1']['cosine'] += F.cosine_similarity(pred_z1_raw, true_z1_raw, dim=-1).mean().item()
            
            # Accumulate Path 2 (z2) Metrics
            metrics['z2']['l1_scaled'] += F.l1_loss(pred_z2, batch_z2).item()
            metrics['z2']['l1_raw'] += F.l1_loss(pred_z2_raw, true_z2_raw).item()
            metrics['z2']['cosine'] += F.cosine_similarity(pred_z2_raw, true_z2_raw, dim=-1).mean().item()
            
            num_batches += 1

    # Average metrics
    for target in ['z1', 'z2']:
        for metric in metrics[target]:
            metrics[target][metric] /= num_batches

    results_text = (
        f"--- Evaluation Results (Decoupled Parallel Tracking): {mode.upper()} ---\n"
        f"Run Name: {run_name}\n"
        f"Validation Pairs: 10,000\n"
        f"----------------------------------------------------\n"
        f"PATH 1 RECOVERY METRICS (Targeting z1):\n"
        f"  L1 Distance (Scaled space): {metrics['z1']['l1_scaled']:.6f}\n"
        f"  L1 Distance (Raw ArcFace):  {metrics['z1']['l1_raw']:.6f}\n"
        f"  Cosine Similarity:          {metrics['z1']['cosine']:.6f}\n"
        f"----------------------------------------------------\n"
        f"PATH 2 RECOVERY METRICS (Targeting z2):\n"
        f"  L1 Distance (Scaled space): {metrics['z2']['l1_scaled']:.6f}\n"
        f"  L1 Distance (Raw ArcFace):  {metrics['z2']['l1_raw']:.6f}\n"
        f"  Cosine Similarity:          {metrics['z2']['cosine']:.6f}\n"
    )
    
    print("\n" + results_text)
    
    with open(out_file_path, "w") as f:
        f.write(results_text)
        
    print(f"Saved evaluation results to {out_file_path}")


if __name__ == "__main__":
    RUN_NAME = "arcface_decoupled_dual_path_deaveraging"
    EMB_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy"

    train_cold_demorph(
        arcface_path_str=EMB_PATH,
        run_name=RUN_NAME,
        num_timesteps=5,
    )

    evaluate_cold_demorph(
        arcface_path_str=EMB_PATH,
        run_name=RUN_NAME,
        num_timesteps=5,
        mode='one_shot',
    )
    
    evaluate_cold_demorph(
        arcface_path_str=EMB_PATH,
        run_name=RUN_NAME,
        num_timesteps=5,
        mode='iterative',
    )
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
            
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, t):
        cond = self.time_mlp(t)
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
    def __init__(self, model, num_timesteps=20):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, z1, z2, t):
        """
        Forward degradation: RESTORED TO UNIFORM LINEAR SCHEDULE.
        Training operates exactly as it was originally designed.
        """
        scale_factor = math.sqrt(512)
        zm = F.normalize(z1 + z2, p=2, dim=-1) * scale_factor
        
        alpha = 1.0 - (t / self.num_timesteps).view(-1, 1).float()
        return alpha * z1 + (1.0 - alpha) * zm

    def compute_loss(self, z1, z2):
        b = z1.shape[0]
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=z1.device).long()
        
        x_t = self.degrade(z1, z2, t)
        pred_z1 = self.model(x_t, t)
        
        # Reconstruct twin via hypersphere mirror geometry
        scale_factor = math.sqrt(512)
        c = F.normalize(z1 + z2, p=2, dim=-1) * scale_factor
        kappa = 2.0 * torch.sum(c * pred_z1, dim=-1, keepdim=True) / 512.0
        pred_z2 = kappa * c - pred_z1
        
        # Project everything into raw L2 space for angular optimization
        p1_raw = F.normalize(pred_z1, p=2, dim=-1)
        p2_raw = F.normalize(pred_z2, p=2, dim=-1)
        z1_raw = F.normalize(z1, p=2, dim=-1)
        z2_raw = F.normalize(z2, p=2, dim=-1)
        
        # 1. Permutation Invariant Target Loss
        cos_to_z1 = F.cosine_similarity(p1_raw, z1_raw, dim=-1)
        cos_to_z2 = F.cosine_similarity(p1_raw, z2_raw, dim=-1)
        loss_target = 1.0 - torch.max(cos_to_z1, cos_to_z2).mean()
        
        # 2. Explicit Inter-Prediction Orthogonality Loss
        cos_real_baseline = F.cosine_similarity(z1_raw, z2_raw, dim=-1)
        cos_predicted_inter = F.cosine_similarity(p1_raw, p2_raw, dim=-1)
        loss_ortho = F.mse_loss(cos_predicted_inter, cos_real_baseline)
        
        return loss_target + 0.5 * loss_ortho

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        """
        Permutation-Aware TACOs sampling with a custom inference trajectory.
        Takes large jumps early on, and switches to 1-by-1 fine-tuning near t=0.
        """
        device = c.device
        b = c.shape[0]
        
        # Dynamically build the sub-sampled inference sequence
        # For T=20, this generates exactly: [20, 15, 11, 8, 6, 4, 3, 2, 1]
        sampled_ticks = []
        t_curr = self.num_timesteps
        step = 5
        while t_curr > 0:
            sampled_ticks.append(t_curr)
            if t_curr <= 5:
                step = 1
            elif t_curr <= 8:
                step = 2
            elif t_curr <= 12:
                step = 3
            elif t_curr <= 16:
                step = 4
            t_curr -= step

        x_t = c.clone()
        
        for i, t in enumerate(tqdm(sampled_ticks, desc='TACOs Sub-Sampling', leave=False)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # 1. Predict one clean embedding component
            pred_z = self.model(x_t, t_batch)
            
            # 2. Recover the exact complementary component from the hypersphere geometry
            kappa = 2.0 * torch.sum(c * pred_z, dim=-1, keepdim=True) / 512.0
            comp_z = kappa * c - pred_z
            
            # 3. Formulate the two possible branch orientations
            opt1_z1, opt1_z2 = pred_z, comp_z
            opt2_z1, opt2_z2 = comp_z, pred_z
            
            # 4. Determine which trajectory matches current x_t using Cosine Similarity
            deg_t_opt1 = self.degrade(opt1_z1, opt1_z2, t_batch)
            deg_t_opt2 = self.degrade(opt2_z1, opt2_z2, t_batch)
            
            cos_opt1 = F.cosine_similarity(deg_t_opt1, x_t, dim=-1)
            cos_opt2 = F.cosine_similarity(deg_t_opt2, x_t, dim=-1)
            
            is_opt1 = (cos_opt1 >= cos_opt2).unsqueeze(-1)
            
            z1_est = torch.where(is_opt1, opt1_z1, opt2_z1)
            z2_est = torch.where(is_opt1, opt1_z2, opt2_z2)
            
            # Look up the next step in our sequence. If we are on the final element, next is 0.
            t_prev = sampled_ticks[i+1] if (i + 1) < len(sampled_ticks) else 0
            t_prev_batch = torch.full((b,), t_prev, device=device, dtype=torch.long)
            
            # 5. Get trajectory updates
            deg_t = self.degrade(z1_est, z2_est, t_batch)
            deg_t_prev = self.degrade(z1_est, z2_est, t_prev_batch)
            
            # 6. TACOs adjustment step bridging custom intervals
            x_t = x_t - deg_t + deg_t_prev
            
        # Extract both final un-morphed outputs cleanly
        final_z1 = x_t
        kappa_final = 2.0 * torch.sum(c * final_z1, dim=-1, keepdim=True) / 512.0
        final_z2 = kappa_final * c - final_z1
        
        return final_z1, final_z2

# ==========================================
# 4. Evaluation & Training Loop
# ==========================================
def train_cold_demorph(arcface_path_str: str, run_name: str, num_timesteps: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Cheap_Loss", "Val_TACOs_Reconstruct_Raw_Cos"])
            
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
        "mixture": "Angular PIT Loss + Orthogonality Penalty Constraint",
        "scheduler": "Linear Training + Sub-sampled Custom Trajectory Inference",
        "scaled_by_sqrt_512": True
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
            pbar.set_postfix({"Loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop (Cheap Metric)
        net.eval()
        val_cheap_loss = 0.0
        val_tacos_cos = None
        
        with torch.no_grad():
            for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val Cheap]"):
                batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                loss_cheap = diffusion.compute_loss(z1=batch_z1, z2=batch_z2)
                val_cheap_loss += loss_cheap.item()
                
        avg_val_cheap_loss = val_cheap_loss / len(val_loader)

        # Validation Loop (Expensive TACOs Metric)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_tacos_cos_total = 0.0
            
            with torch.no_grad():
                for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val TACOs]"):
                    batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                    batch_c = F.normalize(batch_z1 + batch_z2, p=2, dim=-1) * scale_factor
                    
                    pred_z1, _ = diffusion.tacos_sample_loop(batch_c)
                    
                    pred_z1_raw = F.normalize(pred_z1, p=2, dim=-1)
                    true_z1_raw = F.normalize(batch_z1, p=2, dim=-1)
                    true_z2_raw = F.normalize(batch_z2, p=2, dim=-1)
                    
                    cos_z1 = F.cosine_similarity(pred_z1_raw, true_z1_raw, dim=-1)
                    cos_z2 = F.cosine_similarity(pred_z1_raw, true_z2_raw, dim=-1)
                    val_tacos_cos_total += torch.max(cos_z1, cos_z2).mean().item()
                    
            val_tacos_cos = val_tacos_cos_total / len(val_loader)
            
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_cheap_loss": avg_val_cheap_loss
        }
        if val_tacos_cos is not None:
            log_dict["val_tacos_reconstruct_cos"] = val_tacos_cos
            
        wandb.log(log_dict)
        
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{avg_train_loss:.6f}", 
                f"{avg_val_cheap_loss:.6f}", 
                f"{val_tacos_cos:.6f}" if val_tacos_cos is not None else "N/A"
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
            
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Cheap Loss: {avg_val_cheap_loss:.4f}", end="")
        if val_tacos_cos is not None:
            print(f" | Val TACOs Cos: {val_tacos_cos:.4f}")
        else:
            print()

    wandb.finish()


def evaluate_cold_demorph(arcface_path_str: str, run_name: str, num_timesteps: int = 20, mode: str = 'iterative'):
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

    # Metrics accumulators
    metrics = {
        "l1_scaled_z1": 0.0, "l1_scaled_z2": 0.0,
        "l1_raw_z1": 0.0, "l1_raw_z2": 0.0,
        "cos_z1": 0.0, "cos_z2": 0.0,
        "cos_real_z1_z2": 0.0,
        "cos_inter_pred": 0.0
    }
    num_batches = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Evaluating ({mode})"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = F.normalize(batch_z1 + batch_z2, p=2, dim=-1) * scale_factor
            b = batch_z1.shape[0]
            
            # --- 1. Generation of Both Components ---
            if mode == 'iterative':
                pred_z1_scaled, pred_z2_scaled = diffusion.tacos_sample_loop(batch_c)
            else: # one_shot
                t_batch = torch.full((b,), num_timesteps, device=device, dtype=torch.long)
                pred_z1_scaled = net(batch_c, t_batch)
                kappa = 2.0 * torch.sum(batch_c * pred_z1_scaled, dim=-1, keepdim=True) / 512.0
                pred_z2_scaled = kappa * batch_c - pred_z1_scaled

            # --- 2. Raw Space Conversion (L2 Normalized Hypersphere) ---
            p1_raw = F.normalize(pred_z1_scaled, p=2, dim=-1)
            p2_raw = F.normalize(pred_z2_scaled, p=2, dim=-1)
            true_z1_raw = F.normalize(batch_z1, p=2, dim=-1)
            true_z2_raw = F.normalize(batch_z2, p=2, dim=-1)
            
            # --- 3. Dynamic Permutation Alignment via Cosine Similarity ---
            cos_p1_z1 = F.cosine_similarity(p1_raw, true_z1_raw, dim=-1)
            cos_p1_z2 = F.cosine_similarity(p1_raw, true_z2_raw, dim=-1)
            
            is_p1_to_z1 = (cos_p1_z1 >= cos_p1_z2).unsqueeze(-1)
            
            z1_pred_scaled = torch.where(is_p1_to_z1, pred_z1_scaled, pred_z2_scaled)
            z2_pred_scaled = torch.where(is_p1_to_z1, pred_z2_scaled, pred_z1_scaled)
            
            z1_pred_raw = torch.where(is_p1_to_z1, p1_raw, p2_raw)
            z2_pred_raw = torch.where(is_p1_to_z1, p2_raw, p1_raw)

            # --- 4. Compute Metrics ---
            m_l1_scaled_z1 = F.l1_loss(z1_pred_scaled, batch_z1, reduction='none').mean(dim=-1).mean()
            m_l1_raw_z1 = F.l1_loss(z1_pred_raw, true_z1_raw, reduction='none').mean(dim=-1).mean()
            m_cos_z1 = F.cosine_similarity(z1_pred_raw, true_z1_raw, dim=-1).mean()
            
            m_l1_scaled_z2 = F.l1_loss(z2_pred_scaled, batch_z2, reduction='none').mean(dim=-1).mean()
            m_l1_raw_z2 = F.l1_loss(z2_pred_raw, true_z2_raw, reduction='none').mean(dim=-1).mean()
            m_cos_z2 = F.cosine_similarity(z2_pred_raw, true_z2_raw, dim=-1).mean()
            
            m_cos_real = F.cosine_similarity(true_z1_raw, true_z2_raw, dim=-1).mean()
            m_cos_inter = F.cosine_similarity(p1_raw, p2_raw, dim=-1).mean()
            
            # Accumulate
            metrics["l1_scaled_z1"] += m_l1_scaled_z1.item()
            metrics["l1_scaled_z2"] += m_l1_scaled_z2.item()
            metrics["l1_raw_z1"] += m_l1_raw_z1.item()
            metrics["l1_raw_z2"] += m_l1_raw_z2.item()
            metrics["cos_z1"] += m_cos_z1.item()
            metrics["cos_z2"] += m_cos_z2.item()
            metrics["cos_real_z1_z2"] += m_cos_real.item()
            metrics["cos_inter_pred"] += m_cos_inter.item()
            
            num_batches += 1

    for k in metrics:
        metrics[k] /= num_batches

    results_text = (
        f"==================================================\n"
        f"DIFFAE/ARCFACE DEMORPH EVALUATION REPORT ({mode.upper()})\n"
        f"==================================================\n"
        f"Run Name:         {run_name}\n"
        f"Validation Pairs: 10,000\n"
        f"Target Space:     ArcFace 512-D (L2-Normalized Base)\n"
        f"--------------------------------------------------\n\n"
        f"1. GROUND TRUTH Z1 RECONSTRUCTION\n"
        f"--------------------------------------------------\n"
        f"  - L1 Distance (Scaled space): {metrics['l1_scaled_z1']:.6f}\n"
        f"  - L1 Distance (Raw ArcFace):  {metrics['l1_raw_z1']:.6f}\n"
        f"  - Cosine Similarity vs Z1:    {metrics['cos_z1']:.6f}\n\n"
        f"2. GROUND TRUTH Z2 RECONSTRUCTION\n"
        f"--------------------------------------------------\n"
        f"  - L1 Distance (Scaled space): {metrics['l1_scaled_z2']:.6f}\n"
        f"  - L1 Distance (Raw ArcFace):  {metrics['l1_raw_z2']:.6f}\n"
        f"  - Cosine Similarity vs Z2:    {metrics['cos_z2']:.6f}\n\n"
        f"3. GEOMETRIC ALIGNMENT & ORTHOGONALITY COMPARISON\n"
        f"--------------------------------------------------\n"
        f"  - Baseline Cosine Sim (True Z1 vs True Z2):  {metrics['cos_real_z1_z2']:.6f}\n"
        f"  - Generated Cosine Sim (Pred_Z1 vs Pred_Z2): {metrics['cos_inter_pred']:.6f}\n\n"
        f"  Interpretation:\n"
        f"  The Baseline metric proves how uncorrelated the target parents naturally\n"
        f"  are (expecting ~0.0673 based on global dataset tests). For a perfect\n"
        f"  separation, the Generated Cosine Similarity should converge closely to\n"
        f"  or match that exact baseline window.\n"
        f"==================================================\n"
    )
    
    print("\n" + results_text)
    with open(out_file_path, "w") as f:
        f.write(results_text)
        
    print(f"Saved evaluation results to {out_file_path}")


if __name__ == "__main__":
    #train_cold_demorph(
    #    arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
    #    run_name="ortho_loss_run_20ts",
    #    num_timesteps=20,
    #)

    evaluate_cold_demorph(
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="angular_ortho_loss_run_20ts",
        num_timesteps=20,
        mode='one_shot'
    )
    
    evaluate_cold_demorph(
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="angular_ortho_loss_run_20ts",
        num_timesteps=20,
        mode='iterative'
    )

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
import os

# ==========================================
# 1. Dataset & Data Loading
# ==========================================
class ColdArcFaceDemorphDataset(Dataset):
    """
    Dynamically generates pairs of ArcFace embeddings and their average.
    """
    def __init__(self, embeddings: np.ndarray, epoch_size: int = 1000000, deterministic: bool = False):
        # Scale ArcFace by sqrt(512) to keep network inputs O(1)
        self.embeddings = embeddings * np.sqrt(512)
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
    Inputs a 512D mixture and 512D average.
    Outputs a 1024D vector representing the full predicted clean pair.
    """
    def __init__(self, in_dim=512, out_dim=1024, c_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        cond_dim = time_emb_dim + c_dim
        
        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(in_dim, hidden_dim, cond_dim))
        
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + in_dim, hidden_dim, cond_dim))
            
        # Final layer maps to 1024 dimensions
        self.final_linear = nn.Linear(hidden_dim, out_dim)

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

    def compute_loss(self, z1, z2, threshold=9, terminal_weight=9.0):
        b = z1.shape[0]
        c = (z1 + z2) / 2.0
        
        # ---------------------------------------------------------
        # Timestep Importance Sampling
        # ---------------------------------------------------------
        # Create base probabilities (weight 1.0) for all timesteps
        weights = torch.ones(self.num_timesteps, device=z1.device)
        
        # Boost the weight of the terminal timestep (T)
        # If terminal_weight = 9.0, it will be sampled 50% of the time.
        weights[-1] = terminal_weight
        
        # Sample timesteps according to the weighted distribution
        t_indices = torch.multinomial(weights, b, replacement=True)
        t = t_indices + 1 # Shift from 0-indexed to 1-indexed (1 to T)
        
        # ---------------------------------------------------------
        x_t = self.degrade(z1, z2, t)
        
        pred_y = self.model(x_t, t, c)
        pred_z_a, pred_z_b = pred_y.chunk(2, dim=-1)
        
        # 1. Joint Loss (For high timesteps)
        loss_path_a = F.l1_loss(pred_z_a, z1, reduction='none').mean(dim=1) + \
                      F.l1_loss(pred_z_b, z2, reduction='none').mean(dim=1)
                      
        loss_path_b = F.l1_loss(pred_z_a, z2, reduction='none').mean(dim=1) + \
                      F.l1_loss(pred_z_b, z1, reduction='none').mean(dim=1)
                      
        joint_loss_per_item = torch.min(loss_path_a, loss_path_b)
        
        # 2. Dominant-Only Loss (For low timesteps)
        loss_z1_a = F.l1_loss(pred_z_a, z1, reduction='none').mean(dim=1)
        loss_z1_b = F.l1_loss(pred_z_b, z1, reduction='none').mean(dim=1)
        dominant_loss_per_item = torch.min(loss_z1_a, loss_z1_b)
        
        # 3. Dynamic Masking 
        mask_high_t = (t > threshold).float()
        mask_low_t = 1.0 - mask_high_t
        
        final_loss = (mask_high_t * joint_loss_per_item) + (mask_low_t * dominant_loss_per_item)
        
        return final_loss.mean()

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        device = c.device
        b = c.shape[0]
        
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = c.clone()
        
        for t in tqdm(timesteps, desc='TACOs Sampling', leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # 1. Predict BOTH embeddings
            pred_y = self.model(x_t, t_batch, c)
            pred_z_a, pred_z_b = pred_y.chunk(2, dim=-1)
            
            # 2. Select the dominant embedding (the one closest to x_t)
            # Since alpha_t > 0.5, x_t is biased toward the primary embedding z1.
            dist_a = F.l1_loss(pred_z_a, x_t, reduction='none').mean(dim=1, keepdim=True)
            dist_b = F.l1_loss(pred_z_b, x_t, reduction='none').mean(dim=1, keepdim=True)
            
            pred_z1 = torch.where(dist_a <= dist_b, pred_z_a, pred_z_b)
            
            # 3. Mathematically extract the secondary embedding (ignoring the network's prediction for it)
            pred_z2 = 2.0 * c - pred_z1
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            
            # 4. Re-apply degradation
            deg_t = self.degrade(pred_z1, pred_z2, t_batch)
            deg_t_prev = self.degrade(pred_z1, pred_z2, t_prev_batch)
            
            # 5. TACOs update
            x_t = x_t - deg_t + deg_t_prev
            
        final_z1 = x_t
        final_z2 = 2.0 * c - final_z1
        
        return final_z1, final_z2

# ==========================================
# 4. Evaluation & Training Loop
# ==========================================
def train_cold_demorph(arcface_path_str: str, run_name: str, num_timesteps: int = 10):
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
    print("Loading ArcFace embeddings...")
    arcface_path = Path(arcface_path_str).resolve()
    arcface_embs = np.load(arcface_path).astype(np.float32)

    split_idx = int(len(arcface_embs) * 0.9)
    train_embs, val_embs = arcface_embs[:split_idx], arcface_embs[split_idx:]
    
    train_dataset = ColdArcFaceDemorphDataset(train_embs, epoch_size=1_000_000, deterministic=False)
    val_dataset = ColdArcFaceDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16_384, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16_384, shuffle=False, num_workers=4)

    # Model Setup
    net = ColdDemorphNet(in_dim=512, out_dim=1024, c_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4,
        "batch_size": 16_384,
        "num_layers": 10,
        "hidden_dim": 2048,
        "num_timesteps": num_timesteps,
        "latent_space": "ArcFace",
        "output_type": "Joint Pair (1024D)"
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
                    
                    # Ensure strict normalization for metric checking
                    pred_z1 = F.normalize(pred_z1, p=2, dim=-1)
                    pred_z2 = F.normalize(pred_z2, p=2, dim=-1)
                    true_z1 = F.normalize(batch_z1, p=2, dim=-1)
                    true_z2 = F.normalize(batch_z2, p=2, dim=-1)
                    
                    dist_a = F.l1_loss(pred_z1, true_z1, reduction='none').mean(dim=1) + \
                             F.l1_loss(pred_z2, true_z2, reduction='none').mean(dim=1)
                             
                    dist_b = F.l1_loss(pred_z1, true_z2, reduction='none').mean(dim=1) + \
                             F.l1_loss(pred_z2, true_z1, reduction='none').mean(dim=1)
                             
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
def evaluate_cold_demorph(arcface_path_str: str, run_name: str, num_timesteps: int = 10, mode: str = 'iterative'):
    assert mode in ['iterative', 'one_shot'], "Mode must be 'iterative' or 'one_shot'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    out_file_path = exp_dir / f"eval_{mode}.txt"
    
    print(f"Loading ArcFace embeddings for {mode} evaluation...")
    arcface_embs = np.load(Path(arcface_path_str).resolve()).astype(np.float32)
    split_idx = int(len(arcface_embs) * 0.9)
    val_embs = arcface_embs[split_idx:]
    
    val_dataset = ColdArcFaceDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)

    net = ColdDemorphNet(in_dim=512, out_dim=1024, c_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    total_l1_scaled_1, total_l1_scaled_2 = 0.0, 0.0
    total_l1_raw_1, total_l1_raw_2 = 0.0, 0.0
    total_cosine_1, total_cosine_2 = 0.0, 0.0
    num_batches = 0
    scale_factor = math.sqrt(512)

    with torch.no_grad():
        for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Evaluating ({mode})"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = (batch_z1 + batch_z2) / 2.0
            b = batch_z1.shape[0]
            
            if mode == 'iterative':
                pred_z1_scaled, pred_z2_scaled = diffusion.tacos_sample_loop(batch_c)
            else: # one_shot
                t_batch = torch.full((b,), num_timesteps, device=device, dtype=torch.long)
                pred_y = net(batch_c, t_batch, batch_c)
                pred_z_a, pred_z_b = pred_y.chunk(2, dim=-1)
                
                # In one-shot from pure average, select an arbitrary slot for extraction
                pred_z1_scaled = pred_z_a 
                pred_z2_scaled = 2.0 * batch_c - pred_z1_scaled

            # Permutation Alignment
            dist_a = F.l1_loss(pred_z1_scaled, batch_z1, reduction='none').mean(dim=1) + \
                     F.l1_loss(pred_z2_scaled, batch_z2, reduction='none').mean(dim=1)
                     
            dist_b = F.l1_loss(pred_z1_scaled, batch_z2, reduction='none').mean(dim=1) + \
                     F.l1_loss(pred_z2_scaled, batch_z1, reduction='none').mean(dim=1)
                     
            mask_a = (dist_a <= dist_b).unsqueeze(-1)
            
            aligned_pred_z1_scaled = torch.where(mask_a, pred_z1_scaled, pred_z2_scaled)
            aligned_pred_z2_scaled = torch.where(mask_a, pred_z2_scaled, pred_z1_scaled)

            # 3. Compute Metrics
            l1_scaled_1 = F.l1_loss(aligned_pred_z1_scaled, batch_z1)
            l1_scaled_2 = F.l1_loss(aligned_pred_z2_scaled, batch_z2)
            
            pred_z1_raw = F.normalize(aligned_pred_z1_scaled / scale_factor, p=2, dim=-1)
            pred_z2_raw = F.normalize(aligned_pred_z2_scaled / scale_factor, p=2, dim=-1)
            true_z1_raw = F.normalize(batch_z1 / scale_factor, p=2, dim=-1)
            true_z2_raw = F.normalize(batch_z2 / scale_factor, p=2, dim=-1)
            
            l1_raw_1 = F.l1_loss(pred_z1_raw, true_z1_raw)
            l1_raw_2 = F.l1_loss(pred_z2_raw, true_z2_raw)
            
            cos_sim_1 = F.cosine_similarity(pred_z1_raw, true_z1_raw, dim=-1).mean()
            cos_sim_2 = F.cosine_similarity(pred_z2_raw, true_z2_raw, dim=-1).mean()
            
            total_l1_scaled_1 += l1_scaled_1.item()
            total_l1_scaled_2 += l1_scaled_2.item()
            total_l1_raw_1 += l1_raw_1.item()
            total_l1_raw_2 += l1_raw_2.item()
            total_cosine_1 += cos_sim_1.item()
            total_cosine_2 += cos_sim_2.item()
            num_batches += 1

    avg_l1_scaled_1 = total_l1_scaled_1 / num_batches
    avg_l1_scaled_2 = total_l1_scaled_2 / num_batches
    avg_l1_raw_1 = total_l1_raw_1 / num_batches
    avg_l1_raw_2 = total_l1_raw_2 / num_batches
    avg_cosine_1 = total_cosine_1 / num_batches
    avg_cosine_2 = total_cosine_2 / num_batches

    results_text = (
        f"--- Evaluation Results: {mode.upper()} ---\n"
        f"Run Name: {run_name}\n"
        f"Validation Pairs: {len(val_dataset)}\n"
        f"----------------------------------------\n"
        f"[Embedding 1 (Primary Predicted)]\n"
        f"L1 Distance (Scaled space): {avg_l1_scaled_1:.6f}\n"
        f"L1 Distance (Raw ArcFace):  {avg_l1_raw_1:.6f}\n"
        f"Cosine Similarity:          {avg_cosine_1:.6f}\n\n"
        f"[Embedding 2 (Mathematically Extracted)]\n"
        f"L1 Distance (Scaled space): {avg_l1_scaled_2:.6f}\n"
        f"L1 Distance (Raw ArcFace):  {avg_l1_raw_2:.6f}\n"
        f"Cosine Similarity:          {avg_cosine_2:.6f}\n"
    )
    
    print("\n" + results_text)
    
    with open(out_file_path, "w") as f:
        f.write(results_text)
        
    print(f"Saved evaluation results to {out_file_path}")

if __name__ == "__main__":
    train_cold_demorph(
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="avg_arcface_joint_threshold_focus",
        num_timesteps=10
    )

    evaluate_cold_demorph(
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="avg_arcface_joint_threshold_focus",
        num_timesteps=10,
        mode='one_shot'
    )
    
    evaluate_cold_demorph(
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="avg_arcface_joint_threshold_focus",
        num_timesteps=10,
        mode='iterative'
    )

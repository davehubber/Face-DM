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
    Predicts ONE 512D clean embedding given a degraded mixture and the average condition.
    """
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

    def compute_loss(self, z1, z2):
        b = z1.shape[0]
        c = (z1 + z2) / 2.0
        
        # Sample random timestep
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=z1.device).long()
        
        # Construct degraded state with z1 as the dominant component
        x_t = self.degrade(z1, z2, t)
        
        # Predict one of the clean embeddings
        pred_z = self.model(x_t, t, c)
        
        # Permutation invariant L1 loss
        loss_1 = F.l1_loss(pred_z, z1, reduction='none').mean(dim=1)
        loss_2 = F.l1_loss(pred_z, z2, reduction='none').mean(dim=1)
        
        return torch.min(loss_1, loss_2).mean()

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        """
        TACOs sampling for Demorphing.
        Returns BOTH decomposed embeddings (pred_z1, pred_z2).
        """
        device = c.device
        b = c.shape[0]
        
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        
        # Start sampling from the terminal degraded state (the average)
        x_t = c.clone()
        
        for t in tqdm(timesteps, desc='TACOs Sampling', leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # 1. Predict one clean embedding
            pred_z1 = self.model(x_t, t_batch, c)
            
            # 2. Mathematically extract the other embedding
            # Since c = (z1 + z2) / 2, we have z2 = 2c - z1
            pred_z2 = 2.0 * c - pred_z1
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            
            # 3. Re-apply degradation to construct t and t-1
            deg_t = self.degrade(pred_z1, pred_z2, t_batch)
            deg_t_prev = self.degrade(pred_z1, pred_z2, t_prev_batch)
            
            # 4. TACOs update step
            x_t = x_t - deg_t + deg_t_prev
            
        # x_t eventually arrives at x_0, which maps cleanly to z1.
        # We output the final state of x_t and its mathematically paired equivalent.
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

    # Dataset Splits (e.g. 90/10)
    split_idx = int(len(arcface_embs) * 0.9)
    train_embs, val_embs = arcface_embs[:split_idx], arcface_embs[split_idx:]
    
    train_dataset = ColdArcFaceDemorphDataset(train_embs, epoch_size=1_000_000, deterministic=False)
    val_dataset = ColdArcFaceDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16_384, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=16_384, shuffle=False, num_workers=4)

    # Model Setup
    net = ColdDemorphNet(x_dim=512, c_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4,
        "batch_size": 16_384,
        "num_layers": 10,
        "hidden_dim": 2048,
        "num_timesteps": num_timesteps,
        "latent_space": "ArcFace"
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
        
        # ------------------------------------------
        # Validation Loop (Cheap Metric)
        # ------------------------------------------
        net.eval()
        val_cheap_loss = 0.0
        val_tacos_loss = None
        
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
            val_tacos_loss_total = 0.0
            
            with torch.no_grad():
                for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val TACOs]"):
                    batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                    batch_c = (batch_z1 + batch_z2) / 2.0
                    
                    # Generate the decomposed pair using TACOs
                    pred_z1, pred_z2 = diffusion.tacos_sample_loop(batch_c)
                    
                    # Collapse back down to L2 Norm = 1 sphere for true rigid evaluation
                    pred_z1 = F.normalize(pred_z1, p=2, dim=-1)
                    pred_z2 = F.normalize(pred_z2, p=2, dim=-1)
                    true_z1 = F.normalize(batch_z1, p=2, dim=-1)
                    true_z2 = F.normalize(batch_z2, p=2, dim=-1)
                    
                    # Permutation Invariant evaluation on the joint pair
                    dist_a = F.l1_loss(pred_z1, true_z1, reduction='none').mean(dim=1) + \
                             F.l1_loss(pred_z2, true_z2, reduction='none').mean(dim=1)
                             
                    dist_b = F.l1_loss(pred_z1, true_z2, reduction='none').mean(dim=1) + \
                             F.l1_loss(pred_z2, true_z1, reduction='none').mean(dim=1)
                             
                    loss = torch.min(dist_a, dist_b).mean()
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

def evaluate_cold_demorph(arcface_path_str: str, run_name: str, num_timesteps: int = 10, mode: str = 'iterative'):
    """
    Evaluates the trained Cold Diffusion Demorph model on the validation set.
    
    Args:
        arcface_path_str: Path to the raw ArcFace embeddings.
        run_name: The name of the experiment folder.
        num_timesteps: Number of timesteps used during training.
        mode: 'iterative' for TACOs sampling, 'one_shot' for a single direct prediction.
    """
    assert mode in ['iterative', 'one_shot'], "Mode must be 'iterative' or 'one_shot'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------------------------------------
    # Directory & Checkpoint Setup
    # ------------------------------------------
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    out_file_path = exp_dir / f"eval_{mode}.txt"
    
    # ------------------------------------------
    # Data Loading (Exact same split as training)
    # ------------------------------------------
    print(f"Loading ArcFace embeddings for {mode} evaluation...")
    arcface_embs = np.load(Path(arcface_path_str).resolve()).astype(np.float32)
    split_idx = int(len(arcface_embs) * 0.9)
    val_embs = arcface_embs[split_idx:]
    
    # deterministic=True ensures we evaluate on the exact same pairs 
    val_dataset = ColdArcFaceDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)

    # ------------------------------------------
    # Model Setup
    # ------------------------------------------
    net = ColdDemorphNet(x_dim=512, c_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # ------------------------------------------
    # Evaluation Loop
    # ------------------------------------------
    total_l1_scaled = 0.0
    total_l1_raw = 0.0
    total_cosine = 0.0
    num_batches = 0
    
    scale_factor = math.sqrt(512)

    with torch.no_grad():
        for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Evaluating ({mode})"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = (batch_z1 + batch_z2) / 2.0
            b = batch_z1.shape
            
            # --- 1. Generation ---
            if mode == 'iterative':
                pred_z1_scaled, pred_z2_scaled = diffusion.tacos_sample_loop(batch_c)
            else: # one_shot
                # Predict directly from the terminal timestep T
                t_batch = torch.full((b,), num_timesteps, device=device, dtype=torch.long)
                pred_z1_scaled = net(batch_c, t_batch, batch_c)
                pred_z2_scaled = 2.0 * batch_c - pred_z1_scaled

            # --- 2. Permutation Alignment ---
            # We calculate L1 distances in the scaled space to determine the correct pairing path
            dist_a = F.l1_loss(pred_z1_scaled, batch_z1, reduction='none').mean(dim=1) + \
                     F.l1_loss(pred_z2_scaled, batch_z2, reduction='none').mean(dim=1)
                     
            dist_b = F.l1_loss(pred_z1_scaled, batch_z2, reduction='none').mean(dim=1) + \
                     F.l1_loss(pred_z2_scaled, batch_z1, reduction='none').mean(dim=1)
                     
            # Create masks to dynamically swap predictions based on the lowest distance
            mask_a = (dist_a <= dist_b).unsqueeze(-1)
            mask_b = ~mask_a
            
            # Aligned predictions (Scaled Space)
            aligned_pred_z1_scaled = torch.where(mask_a, pred_z1_scaled, pred_z2_scaled)
            aligned_pred_z2_scaled = torch.where(mask_a, pred_z2_scaled, pred_z1_scaled)

            # --- 3. Compute Metrics ---
            # Metric 1: Scaled L1 (Matching the training loss magnitude)
            l1_scaled = (F.l1_loss(aligned_pred_z1_scaled, batch_z1) + 
                         F.l1_loss(aligned_pred_z2_scaled, batch_z2)) / 2.0
            
            # Map back to strictly L2-normalized ArcFace domain
            pred_z1_raw = F.normalize(aligned_pred_z1_scaled / scale_factor, p=2, dim=-1)
            pred_z2_raw = F.normalize(aligned_pred_z2_scaled / scale_factor, p=2, dim=-1)
            true_z1_raw = F.normalize(batch_z1 / scale_factor, p=2, dim=-1)
            true_z2_raw = F.normalize(batch_z2 / scale_factor, p=2, dim=-1)
            
            # Metric 2: Raw ArcFace L1
            l1_raw = (F.l1_loss(pred_z1_raw, true_z1_raw) + 
                      F.l1_loss(pred_z2_raw, true_z2_raw)) / 2.0
            
            # Metric 3: Cosine Similarity
            cos_sim_1 = F.cosine_similarity(pred_z1_raw, true_z1_raw, dim=-1).mean()
            cos_sim_2 = F.cosine_similarity(pred_z2_raw, true_z2_raw, dim=-1).mean()
            cos_sim = (cos_sim_1 + cos_sim_2) / 2.0
            
            total_l1_scaled += l1_scaled.item()
            total_l1_raw += l1_raw.item()
            total_cosine += cos_sim.item()
            num_batches += 1

    # ------------------------------------------
    # Finalize & Save
    # ------------------------------------------
    avg_l1_scaled = total_l1_scaled / num_batches
    avg_l1_raw = total_l1_raw / num_batches
    avg_cosine = total_cosine / num_batches

    results_text = (
        f"--- Evaluation Results: {mode.upper()} ---\n"
        f"Run Name: {run_name}\n"
        f"Validation Pairs: 10,000\n"
        f"----------------------------------------\n"
        f"L1 Distance (Scaled space): {avg_l1_scaled:.6f}\n"
        f"L1 Distance (Raw ArcFace):  {avg_l1_raw:.6f}\n"
        f"Cosine Similarity:          {avg_cosine:.6f}\n"
    )
    
    print("\n" + results_text)
    
    with open(out_file_path, "w") as f:
        f.write(results_text)
        
    print(f"Saved evaluation results to {out_file_path}")

if __name__ == "__main__":
    # train_cold_demorph(
    #     arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
    #     run_name="avg_arcface",
    #     num_timesteps=10
    # )

    evaluate_cold_demorph(
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="avg_arcface",
        num_timesteps=10,
        mode='one_shot'
    )
    
    evaluate_cold_demorph(
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="avg_arcface",
        num_timesteps=10,
        mode='iterative'
    )
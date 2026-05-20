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
    Loads pre-computed morph pairs from an .npz file.
    Couples z1 and z2 into a single 1024-dimensional vector.
    """
    def __init__(self, npz_path: str | Path):
        print(f"Loading dataset from {npz_path}...")
        data = np.load(Path(npz_path))
        
        # Scale ArcFace by sqrt(512) to keep network inputs O(1)
        scale_factor = np.sqrt(512)
        z1 = data['z1'] * scale_factor
        z2 = data['z2'] * scale_factor
        self.c = data['c'] * scale_factor
        
        # Couple z1 and z2 into a 1024D vector
        self.z12 = np.concatenate([z1, z2], axis=-1)
        self.num_samples = len(self.c)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.tensor(self.z12[idx], dtype=torch.float32),
            torch.tensor(self.c[idx], dtype=torch.float32)
        )

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
    Predicts the FULL 1024D coupled clean state [z1, z2] given a degraded state and morph condition c.
    """
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
# 3. Cold Demorph Diffusion Process
# ==========================================
class DeterministicColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=10):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, z12, c, t):
        """
        Forward degradation: Both z1 and z2 halves interpolate independently towards c.
        t=0: alpha=1.0 -> purely [z1, z2]
        t=T: alpha=0.0 -> purely [c, c]
        """
        alpha = 1.0 - (t / self.num_timesteps).view(-1, 1).float()
        
        z1, z2 = z12.chunk(2, dim=-1)
        
        deg_z1 = alpha * z1 + (1.0 - alpha) * c
        deg_z2 = alpha * z2 + (1.0 - alpha) * c
        
        return torch.cat([deg_z1, deg_z2], dim=-1)

    def compute_loss(self, z12, c):
        b = z12.shape[0]
        
        # Sample random timestep
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=z12.device).long()
        
        # Construct degraded state along the clean trajectory from [z1, z2] to [c, c]
        x_t = self.degrade(z12, c, t)
        
        # Predict the full 1024D clean state [z1, z2]
        pred_z12 = self.model(x_t, t, c)
        
        return F.l1_loss(pred_z12, z12, reduction='mean')

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        """
        TACOs sampling for Demorphing.
        Traverses back from [c, c] (t=T) to [z1, z2] (t=0).
        """
        device = c.device
        b = c.shape[0]
        
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        
        # Start sampling exactly from the terminal state: [c, c]
        x_t = torch.cat([c, c], dim=-1)
        
        for t in tqdm(timesteps, desc='TACOs Sampling', leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # 1. Predict the full coupled target embeddings [z1, z2]
            pred_z12 = self.model(x_t, t_batch, c)
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            
            # 2. Re-apply degradation operators to construct steps t and t-1
            deg_t = self.degrade(pred_z12, c, t_batch)
            deg_t_prev = self.degrade(pred_z12, c, t_prev_batch)
            
            # 3. TACOs update step
            x_t = x_t - deg_t + deg_t_prev
            
        return x_t

# ==========================================
# 4. Evaluation & Training Loop
# ==========================================
def train_cold_demorph(train_npz: str, val_npz: str, run_name: str, num_timesteps: int = 10):
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
    train_dataset = ColdArcFaceDemorphDataset(train_npz)
    val_dataset = ColdArcFaceDemorphDataset(val_npz)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)

    # Model Setup (x_dim updated to 1024)
    net = ColdDemorphNet(x_dim=1024, c_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4,
        "batch_size": 512,
        "num_layers": 10,
        "hidden_dim": 2048,
        "num_timesteps": num_timesteps,
        "latent_space": "ArcFace",
        "description": "Direct interpolation trajectory between coupled [z1, z2] and terminal [c, c] state."
    })

    epochs = 50 
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_z12, batch_c in pbar:
            batch_z12, batch_c = batch_z12.to(device), batch_c.to(device)
            
            optimizer.zero_grad()
            loss = diffusion.compute_loss(z12=batch_z12, c=batch_c)
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
        val_tacos_raw_loss = None
        
        with torch.no_grad():
            for batch_z12, batch_c in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val Cheap]"):
                batch_z12, batch_c = batch_z12.to(device), batch_c.to(device)
                
                loss_cheap = diffusion.compute_loss(z12=batch_z12, c=batch_c)
                val_cheap_loss += loss_cheap.item()
                
        avg_val_cheap_loss = val_cheap_loss / len(val_loader)

        # ------------------------------------------
        # Validation Loop (Expensive TACOs Metric)
        # ------------------------------------------
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_tacos_loss_total = 0.0
            val_tacos_raw_loss_total = 0.0
            
            with torch.no_grad():
                for batch_z12, batch_c in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val TACOs]"):
                    batch_z12, batch_c = batch_z12.to(device), batch_c.to(device)
                    
                    # Generate the coupled vector using TACOs
                    pred_z12 = diffusion.tacos_sample_loop(batch_c)
                    
                    # 1. Comparable Scaled Loss
                    loss_scaled = F.l1_loss(pred_z12, batch_z12, reduction='mean')
                    val_tacos_loss_total += loss_scaled.item()
                    
                    # 2. Raw Hypersphere Space Loss
                    pred_z1, pred_z2 = pred_z12.chunk(2, dim=-1)
                    batch_z1, batch_z2 = batch_z12.chunk(2, dim=-1)
                    
                    pred_z1_raw = F.normalize(pred_z1 / math.sqrt(512), p=2, dim=-1)
                    true_z1_raw = F.normalize(batch_z1 / math.sqrt(512), p=2, dim=-1)
                    pred_z2_raw = F.normalize(pred_z2 / math.sqrt(512), p=2, dim=-1)
                    true_z2_raw = F.normalize(batch_z2 / math.sqrt(512), p=2, dim=-1)
                    
                    loss_raw = (F.l1_loss(pred_z1_raw, true_z1_raw, reduction='mean') + 
                                F.l1_loss(pred_z2_raw, true_z2_raw, reduction='mean')) / 2.0
                    val_tacos_raw_loss_total += loss_raw.item()
                    
            val_tacos_loss = val_tacos_loss_total / len(val_loader)
            val_tacos_raw_loss = val_tacos_raw_loss_total / len(val_loader)
            
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
            log_dict["val_tacos_reconstruct_raw_l1"] = val_tacos_raw_loss
            
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

def evaluate_cold_demorph(train_npz: str, val_npz: str, run_name: str, num_timesteps: int = 10, mode: str = 'iterative'):
    """
    Evaluates the trained model on the structured morph validation set for both decoupled targets.
    """
    assert mode in ['iterative', 'one_shot'], "Mode must be 'iterative' or 'one_shot'"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    out_file_path = exp_dir / f"eval_{mode}.txt"
    
    val_dataset = ColdArcFaceDemorphDataset(val_npz)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)

    net = ColdDemorphNet(x_dim=1024, c_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    total_l1_scaled = 0.0
    total_l1_raw_z1, total_l1_raw_z2 = 0.0, 0.0
    total_cosine_z1, total_cosine_z2 = 0.0, 0.0
    num_batches = 0
    
    scale_factor = math.sqrt(512)

    with torch.no_grad():
        for batch_z12, batch_c in tqdm(val_loader, desc=f"Evaluating ({mode})"):
            batch_z12, batch_c = batch_z12.to(device), batch_c.to(device)
            b = batch_z12.shape[0]
            
            # --- 1. Generation ---
            if mode == 'iterative':
                pred_z12_scaled = diffusion.tacos_sample_loop(batch_c)
            else: # one_shot
                t_batch = torch.full((b,), num_timesteps, device=device, dtype=torch.long)
                # At t=T, terminal state is [c, c]
                terminal_state = torch.cat([batch_c, batch_c], dim=-1)
                pred_z12_scaled = net(terminal_state, t_batch, batch_c)

            # --- 2. Compute Metrics ---
            l1_scaled = F.l1_loss(pred_z12_scaled, batch_z12).item()
            
            # Unpack into separate components
            pred_z1, pred_z2 = pred_z12_scaled.chunk(2, dim=-1)
            batch_z1, batch_z2 = batch_z12.chunk(2, dim=-1)
            
            # Transform variables back into normalized spaces
            pred_z1_raw = F.normalize(pred_z1 / scale_factor, p=2, dim=-1)
            true_z1_raw = F.normalize(batch_z1 / scale_factor, p=2, dim=-1)
            
            pred_z2_raw = F.normalize(pred_z2 / scale_factor, p=2, dim=-1)
            true_z2_raw = F.normalize(batch_z2 / scale_factor, p=2, dim=-1)
            
            l1_raw_z1 = F.l1_loss(pred_z1_raw, true_z1_raw).item()
            cos_sim_z1 = F.cosine_similarity(pred_z1_raw, true_z1_raw, dim=-1).mean().item()
            
            l1_raw_z2 = F.l1_loss(pred_z2_raw, true_z2_raw).item()
            cos_sim_z2 = F.cosine_similarity(pred_z2_raw, true_z2_raw, dim=-1).mean().item()
            
            total_l1_scaled += l1_scaled
            total_l1_raw_z1 += l1_raw_z1
            total_l1_raw_z2 += l1_raw_z2
            total_cosine_z1 += cos_sim_z1
            total_cosine_z2 += cos_sim_z2
            num_batches += 1

    avg_l1_scaled = total_l1_scaled / num_batches
    avg_l1_raw_z1 = total_l1_raw_z1 / num_batches
    avg_l1_raw_z2 = total_l1_raw_z2 / num_batches
    avg_cosine_z1 = total_cosine_z1 / num_batches
    avg_cosine_z2 = total_cosine_z2 / num_batches

    results_text = (
        f"--- Evaluation Results (Coupled Model): {mode.upper()} ---\n"
        f"Run Name: {run_name}\n"
        f"Validation Pairs: {len(val_dataset)}\n"
        f"----------------------------------------\n"
        f"Total Shared L1 Distance (Scaled space): {avg_l1_scaled:.6f}\n"
        f"Target z1 L1 Distance (Raw ArcFace):     {avg_l1_raw_z1:.6f}\n"
        f"Target z2 L1 Distance (Raw ArcFace):     {avg_l1_raw_z2:.6f}\n"
        f"Target z1 Cosine Similarity:             {avg_cosine_z1:.6f}\n"
        f"Target z2 Cosine Similarity:             {avg_cosine_z2:.6f}\n"
    )
    
    print("\n" + results_text)
    with open(out_file_path, "w") as f:
        f.write(results_text)

if __name__ == "__main__":
    TRAIN_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/morphed_dataset/train_morphs.npz"
    VAL_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/morphed_dataset/val_morphs.npz"
    RUN_ID = "coupled_arcface_img"

    train_cold_demorph(
        train_npz=TRAIN_PATH,
        val_npz=VAL_PATH,
        run_name=RUN_ID,
        num_timesteps=10
    )

    evaluate_cold_demorph(
        train_npz=TRAIN_PATH,
        val_npz=VAL_PATH,
        run_name=RUN_ID,
        num_timesteps=10,
        mode='one_shot'
    )
    
    evaluate_cold_demorph(
        train_npz=TRAIN_PATH,
        val_npz=VAL_PATH,
        run_name=RUN_ID,
        num_timesteps=10,
        mode='iterative'
    )

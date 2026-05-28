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
# 2. Transformer Network Architecture (DiT-style)
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

class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block using Adaptive Layer Normalization (AdaLN-Modulation)
    to dynamically scale and shift feature configurations based on the timestep.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Generates scale, shift, and gate parameters for both MHA and MLP blocks
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model)
        )

    def forward(self, x, cond):
        # x: [B, N, d_model], cond: [B, cond_dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_mod(cond).chunk(6, dim=-1)
        
        # Unsqueeze conditioning vectors to broadcast across sequence length dimension (N)
        shift_msa, scale_msa, gate_msa = shift_msa.unsqueeze(1), scale_msa.unsqueeze(1), gate_msa.unsqueeze(1)
        shift_mlp, scale_mlp, gate_mlp = shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)
        
        # Attention Branch
        h1 = self.norm1(x) * (1 + scale_msa) + shift_msa
        h1, _ = self.attn(h1, h1, h1)
        x = x + gate_msa * h1
        
        # Feed-Forward Branch
        h2 = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        h2 = self.mlp(h2)
        x = x + gate_mlp * h2
        
        return x

class ColdDemorphTransformer(nn.Module):
    """
    Transformer backbone that chunks a 512-dim ArcFace embedding into a token sequence,
    allowing self-attention to identify anomalous feature compositions.
    """
    def __init__(self, x_dim=512, num_tokens=16, d_model=512, nhead=8, dim_feedforward=2048, num_layers=8, time_emb_dim=512):
        super().__init__()
        assert x_dim % num_tokens == 0, f"Embedding dimension {x_dim} must be divisible by num_tokens {num_tokens}"
        
        self.num_tokens = num_tokens
        self.token_dim = x_dim // num_tokens
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        # Project vector chunks into Transformer hidden space
        self.input_proj = nn.Linear(self.token_dim, d_model)
        
        # Learnable spatial position embeddings for the 1D token sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, dim_feedforward, cond_dim=time_emb_dim)
            for _ in range(num_layers)
        ])
        
        # Final output layer norm and modulation projection
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * d_model)
        )
        self.output_proj = nn.Linear(d_model, self.token_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Zero-initialize the AdaLN linear projections so the network starts off acting close to an identity mapping
        for block in self.blocks:
            nn.init.constant_(block.adaLN_mod[-1].weight, 0)
            nn.init.constant_(block.adaLN_mod[-1].bias, 0)
        nn.init.constant_(self.final_adaLN_mod[-1].weight, 0)
        nn.init.constant_(self.final_adaLN_mod[-1].bias, 0)

    def forward(self, x, t):
        # x: [B, 512], t: [B]
        B = x.shape[0]
        cond = self.time_mlp(t)
        
        # Chunk 512-dim vector into sequence: [B, 16, 32]
        x_seq = x.view(B, self.num_tokens, self.token_dim)
        
        # Linear project and add positional info
        h = self.input_proj(x_seq) + self.pos_embed
        
        # Process through DiT layers
        for block in self.blocks:
            h = block(h, cond)
            
        # Apply final modulated layer norm
        scale, shift = self.final_adaLN_mod(cond).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        # Map back to token space and flatten back to [B, 512]
        out_seq = self.output_proj(h)
        return out_seq.view(B, -1)

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
        Forward degradation: Interpolates between z1 and the hypersphere-normalized mixture zm.
        """
        scale_factor = math.sqrt(512)
        zm = F.normalize(z1 + z2, p=2, dim=-1) * scale_factor
        
        alpha = 1.0 - (t / self.num_timesteps).view(-1, 1).float()
        return alpha * z1 + (1.0 - alpha) * zm

    def compute_loss(self, z1, z2):
        b = z1.shape[0]
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=z1.device).long()
        
        x_t = self.degrade(z1, z2, t)
        pred_z = self.model(x_t, t)
        
        loss_z1 = F.l1_loss(pred_z, z1, reduction='none').mean(dim=-1)
        loss_z2 = F.l1_loss(pred_z, z2, reduction='none').mean(dim=-1)
        loss = torch.min(loss_z1, loss_z2).mean()
        
        return loss

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        """
        Permutation-Aware TACOs sampling for Hypersphere Demorphing.
        """
        device = c.device
        b = c.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = c.clone()
        
        for t in tqdm(timesteps, desc='TACOs Sampling', leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # 1. Predict one clean embedding component
            pred_z = self.model(x_t, t_batch)
            
            # 2. Recover the exact complementary component from the hypersphere geometry
            kappa = 2.0 * torch.sum(c * pred_z, dim=-1, keepdim=True) / 512.0
            comp_z = kappa * c - pred_z
            
            # 3. Formulate the two possible branch orientations
            opt1_z1, opt1_z2 = pred_z, comp_z
            opt2_z1, opt2_z2 = comp_z, pred_z
            
            # 4. Determine which trajectory configuration matches current x_t
            deg_t_opt1 = self.degrade(opt1_z1, opt1_z2, t_batch)
            deg_t_opt2 = self.degrade(opt2_z1, opt2_z2, t_batch)
            
            dist_opt1 = F.l1_loss(deg_t_opt1, x_t, reduction='none').mean(dim=-1)
            dist_opt2 = F.l1_loss(deg_t_opt2, x_t, reduction='none').mean(dim=-1)
            
            is_opt1 = (dist_opt1 <= dist_opt2).unsqueeze(-1)
            
            z1_est = torch.where(is_opt1, opt1_z1, opt2_z1)
            z2_est = torch.where(is_opt1, opt1_z2, opt2_z2)
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            
            # 5. Get trajectory updates
            deg_t = self.degrade(z1_est, z2_est, t_batch)
            deg_t_prev = self.degrade(z1_est, z2_est, t_prev_batch)
            
            # 6. TACOs adjustment step
            x_t = x_t - deg_t + deg_t_prev
            
        # Extract both final un-morphed outputs cleanly
        final_z1 = x_t
        kappa_final = 2.0 * torch.sum(c * final_z1, dim=-1, keepdim=True) / 512.0
        final_z2 = kappa_final * c - final_z1
        
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
            writer.writerow(["Epoch", "Train_L1", "Val_Cheap_L1", "Val_TACOs_Reconstruct_L1"])
            
    print("Loading ArcFace embeddings...")
    arcface_path = Path(arcface_path_str).resolve()
    scale_factor = np.sqrt(512)
    arcface_embs = (np.load(arcface_path).astype(np.float32)) * scale_factor

    split_idx = int(len(arcface_embs) * 0.9)
    train_embs, val_embs = arcface_embs[:split_idx], arcface_embs[split_idx:]
    
    train_dataset = ColdArcFaceDemorphDataset(train_embs, epoch_size=1_000_000, deterministic=False)
    val_dataset = ColdArcFaceDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    
    train_loader = DataLoader(train_dataset, batch_size=1_024, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)

    # Initializing the Transformer model instead of the simple MLP network
    net = ColdDemorphTransformer(
        x_dim=512, 
        num_tokens=16, 
        d_model=512, 
        nhead=8, 
        dim_feedforward=2048, 
        num_layers=8
    ).to(device)
    
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 3e-4,
        "batch_size": 1_024,
        "num_layers": 8,
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 2048,
        "num_timesteps": num_timesteps,
        "latent_space": "ArcFace",
        "architecture": "Diffusion Transformer (DiT-1D Chunks)",
        "mixture": "Hypersphere Projected Normal Mixture (Permutation-aware TACOs)",
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
            
            optimizer.zero_zero_grad() if hasattr(optimizer, "zero_zero_grad") else optimizer.zero_grad()
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
        val_tacos_raw_loss = None
        
        with torch.no_grad():
            for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val Cheap]"):
                batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                loss_cheap = diffusion.compute_loss(z1=batch_z1, z2=batch_z2)
                val_cheap_loss += loss_cheap.item()
                
        avg_val_cheap_loss = val_cheap_loss / len(val_loader)

        # Validation Loop (Expensive TACOs Metric)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            val_tacos_loss_total = 0.0
            val_tacos_raw_loss_total = 0.0
            
            with torch.no_grad():
                for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val TACOs]"):
                    batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                    
                    batch_c = F.normalize(batch_z1 + batch_z2, p=2, dim=-1) * math.sqrt(512)
                    
                    pred_z1, _ = diffusion.tacos_sample_loop(batch_c)
                    
                    loss_scaled_z1 = F.l1_loss(pred_z1, batch_z1, reduction='none').mean(dim=-1)
                    loss_scaled_z2 = F.l1_loss(pred_z1, batch_z2, reduction='none').mean(dim=-1)
                    loss_scaled = torch.min(loss_scaled_z1, loss_scaled_z2).mean()
                    val_tacos_loss_total += loss_scaled.item()
                    
                    pred_z1_raw = F.normalize(pred_z1 / math.sqrt(512), p=2, dim=-1)
                    true_z1_raw = F.normalize(batch_z1 / math.sqrt(512), p=2, dim=-1)
                    true_z2_raw = F.normalize(batch_z2 / math.sqrt(512), p=2, dim=-1)
                    
                    loss_raw_z1 = F.l1_loss(pred_z1_raw, true_z1_raw, reduction='none').mean(dim=-1)
                    loss_raw_z2 = F.l1_loss(pred_z1_raw, true_z2_raw, reduction='none').mean(dim=-1)
                    loss_raw = torch.min(loss_raw_z1, loss_raw_z2).mean()
                    val_tacos_raw_loss_total += loss_raw.item()
                    
            val_tacos_loss = val_tacos_loss_total / len(val_loader)
            val_tacos_raw_loss = val_tacos_raw_loss_total / len(val_loader)
            
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


def evaluate_cold_demorph(arcface_path_str: str, run_name: str, num_timesteps: int = 10, mode: str = 'iterative'):
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
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)

    net = ColdDemorphTransformer(
        x_dim=512, 
        num_tokens=16, 
        d_model=512, 
        nhead=8, 
        dim_feedforward=2048, 
        num_layers=8
    ).to(device)
    
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    total_l1_scaled = 0.0
    total_l1_raw = 0.0
    total_cosine = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Evaluating ({mode})"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = F.normalize(batch_z1 + batch_z2, p=2, dim=-1) * scale_factor
            b = batch_z1.shape[0]
            
            # --- 1. Generation ---
            if mode == 'iterative':
                pred_z_scaled, _ = diffusion.tacos_sample_loop(batch_c)
            else: # one_shot
                t_batch = torch.full((b,), num_timesteps, device=device, dtype=torch.long)
                pred_z_scaled = net(batch_c, t_batch)

            # --- 2. Compute Permutation-Invariant Metrics ---
            l1_scaled_z1 = F.l1_loss(pred_z_scaled, batch_z1, reduction='none').mean(dim=-1)
            l1_scaled_z2 = F.l1_loss(pred_z_scaled, batch_z2, reduction='none').mean(dim=-1)
            l1_scaled = torch.min(l1_scaled_z1, l1_scaled_z2).mean()
            
            pred_z_raw = F.normalize(pred_z_scaled / scale_factor, p=2, dim=-1)
            true_z1_raw = F.normalize(batch_z1 / scale_factor, p=2, dim=-1)
            true_z2_raw = F.normalize(batch_z2 / scale_factor, p=2, dim=-1)
            
            l1_raw_z1 = F.l1_loss(pred_z_raw, true_z1_raw, reduction='none').mean(dim=-1)
            l1_raw_z2 = F.l1_loss(pred_z_raw, true_z2_raw, reduction='none').mean(dim=-1)
            l1_raw = torch.min(l1_raw_z1, l1_raw_z2).mean()
            
            cos_sim_z1 = F.cosine_similarity(pred_z_raw, true_z1_raw, dim=-1)
            cos_sim_z2 = F.cosine_similarity(pred_z_raw, true_z2_raw, dim=-1)
            cos_sim = torch.max(cos_sim_z1, cos_sim_z2).mean()
            
            total_l1_scaled += l1_scaled.item()
            total_l1_raw += l1_raw.item()
            total_cosine += cos_sim.item()
            num_batches += 1

    avg_l1_scaled = total_l1_scaled / num_batches
    avg_l1_raw = total_l1_raw / num_batches
    avg_cosine = total_cosine / num_batches

    results_text = (
        f"--- Evaluation Results (Permutation Invariant & Spherical TACOs): {mode.upper()} ---\n"
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
    train_cold_demorph(
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="avg_arcface_50_50_pit_dit",
        num_timesteps=5,
    )

    evaluate_cold_demorph(
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="avg_arcface_50_50_pit_dit",
        num_timesteps=5,
        mode='one_shot'
    )
    
    evaluate_cold_demorph(
        arcface_path_str="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        run_name="avg_arcface_50_50_pit_dit",
        num_timesteps=5,
        mode='iterative'
    )

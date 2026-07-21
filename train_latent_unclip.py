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
# 1. Dataset & Cross-Domain Data Loading
# ==========================================
class CrossDomainDemorphTrainDataset(Dataset):
    """Generates random on-the-fly face-flower pairs for training."""
    def __init__(self, face_embeddings: np.ndarray, flower_embeddings: np.ndarray, epoch_size: int = 1_000_000):
        self.face_embs = face_embeddings
        self.flower_embs = flower_embeddings
        self.epoch_size = epoch_size
        self.num_faces = len(face_embeddings)
        self.num_flowers = len(flower_embeddings)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        idx_face = np.random.randint(0, self.num_faces)
        idx_flower = np.random.randint(0, self.num_flowers)
            
        return (torch.tensor(self.face_embs[idx_face], dtype=torch.float32), 
                torch.tensor(self.flower_embs[idx_flower], dtype=torch.float32))

class CrossDomainDemorphTestPairsDataset(Dataset):
    """Loads statically paired evaluation arrays for cross-domain unmixing."""
    def __init__(self, paired_embeddings: np.ndarray):
        self.pairs = paired_embeddings

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        z_face = self.pairs[idx, 0]
        z_flower = self.pairs[idx, 1]
        return torch.tensor(z_face, dtype=torch.float32), torch.tensor(z_flower, dtype=torch.float32)

def load_cross_domain_split(base_path_str: str, split: str) -> np.ndarray:
    """Loads embeddings and scales them by the square root of their dimensionality."""
    split_path = Path(base_path_str).resolve()
    if not split_path.exists():
        raise FileNotFoundError(f"Embedding file missing at: {split_path}")
        
    data = np.load(split_path).astype(np.float32)
    
    # Dynamically scale by sqrt(dim) to adjust to unCLIP feature sizes automatically
    dim = data.shape[-1]
    data = data * math.sqrt(dim)
    return data

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
        
        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(x_dim, hidden_dim, time_emb_dim))
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + x_dim, hidden_dim, time_emb_dim))
            
        self.final_linear = nn.Linear(hidden_dim, x_dim * 2)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = x
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, t_emb)
            else:
                h = block(torch.cat([h, x], dim=-1), t_emb)
        return self.final_linear(h)

# ==========================================
# 3. Cold Diffusion Process (Static Domain Trajectory)
# ==========================================
class DeterministicColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=300):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.SQRT_2 = math.sqrt(2.0)

    def degrade(self, z_face, z_flower, t):
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        w1 = torch.sqrt(1.0 - 0.5 * gamma)
        w2 = torch.sqrt(0.5 * gamma)
        return w1 * z_face + w2 * z_flower

    def compute_loss(self, z_face, z_flower):
        """Computes direct static domain loss mapping face to Head 1 and flower to Head 2."""
        b = z_face.shape[0]
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=z_face.device).long()
        
        x_t = self.degrade(z_face, z_flower, t)
        pred = self.model(x_t, t)
        
        pred_z_face, pred_z_flower = pred.chunk(2, dim=-1)
        
        # Static cross-domain mapping without permutation checks
        loss_face = F.mse_loss(pred_z_face, z_face)
        loss_flower = F.mse_loss(pred_z_flower, z_flower)
        
        return loss_face + loss_flower

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        """Refines the face embedding directly and extracts the flower complement mathematically."""
        device = c.device
        b = c.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = c.clone()
        
        for t in tqdm(timesteps, desc='TACOs Sampling', leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            pred_raw = self.model(x_t, t_batch)
            
            # Head 1 is statically locked to the dominant face trajectory path
            pred_z_face, _ = pred_raw.chunk(2, dim=-1)
            
            # Extract the flower embedding mathematically from the midpoint equation
            pred_z_flower = self.SQRT_2 * c - pred_z_face
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            deg_t = self.degrade(pred_z_face, pred_z_flower, t_batch)
            deg_t_prev = self.degrade(pred_z_face, pred_z_flower, t_prev_batch)
            
            x_t = x_t - deg_t + deg_t_prev
            
        final_z_face = x_t
        final_z_flower = self.SQRT_2 * c - final_z_face
        return final_z_face, final_z_flower

# ==========================================
# 4. Early Stopping Tracker Engine
# ==========================================
class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0  
        else:
            self.counter += 1
            print(f" EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# ==========================================
# 5. Training Mechanics
# ==========================================
def train_cross_domain_demorph(face_train_path, flower_train_path, face_test_path, flower_test_path, run_name, num_timesteps=300, epochs=150, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Train_Loss", "Val_Cheap_Loss", "Val_Cheap_Cos_Similarity", "Val_TACOs_Reconstruct_MSE"])
            
    # Load and scale datasets independently
    face_train = load_cross_domain_split(face_train_path, "train")
    flower_train = load_cross_domain_split(flower_train_path, "train")
    
    face_test = load_cross_domain_split(face_test_path, "test")
    flower_test = load_cross_domain_split(flower_test_path, "test")
    
    # Combine individual test sets into static evaluation pairs -> Shape: (1000, 2, dim)
    test_pairs = np.stack([face_test[:1000], flower_test[:1000]], axis=1)
    
    embedding_dim = face_train.shape[-1]
    
    train_loader = DataLoader(CrossDomainDemorphTrainDataset(face_train, flower_train, epoch_size=1_000_000), batch_size=32_768, shuffle=True, num_workers=8)
    val_loader = DataLoader(CrossDomainDemorphTestPairsDataset(test_pairs), batch_size=1000, shuffle=False, num_workers=4)

    net = ColdDemorphNet(x_dim=embedding_dim, time_emb_dim=embedding_dim).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    early_stopper = EarlyStopping(patience=patience, min_delta=1e-5)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4, "batch_size": 32_768, "num_layers": 10, "hidden_dim": 2048, "num_timesteps": num_timesteps, "early_stop_patience": patience, "embedding_type": "unCLIP_CrossDomain_Face_Flower", "loss_type": "Static_Domain_MSE"
    })

    best_val_loss = float("inf")
    SQRT_05 = math.sqrt(0.5)
    
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for batch_face, batch_flower in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_face, batch_flower = batch_face.to(device), batch_flower.to(device)
            optimizer.zero_grad()
            loss = diffusion.compute_loss(batch_face, batch_flower)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        net.eval()
        val_cheap_loss = 0.0
        val_cheap_cos = 0.0
        val_tacos_loss = None
        
        with torch.no_grad():
            for batch_face, batch_flower in val_loader:
                batch_face, batch_flower = batch_face.to(device), batch_flower.to(device)
                
                b = batch_face.shape[0]
                t = torch.randint(1, diffusion.num_timesteps + 1, (b,), device=batch_face.device).long()
                x_t = diffusion.degrade(batch_face, batch_flower, t)
                pred = net(x_t, t)
                pred_z_face, pred_z_flower = pred.chunk(2, dim=-1)
                
                # Check metrics along fixed domain assignments
                val_cheap_loss += (F.mse_loss(pred_z_face, batch_face) + F.mse_loss(pred_z_flower, batch_flower)).item()
                
                cos_face = F.cosine_similarity(pred_z_face, batch_face, dim=-1)
                cos_flower = F.cosine_similarity(pred_z_flower, batch_flower, dim=-1)
                val_cheap_cos += 0.5 * (cos_face + cos_flower).mean().item()

        avg_val_cheap_loss = val_cheap_loss / len(val_loader)
        avg_val_cheap_cos = val_cheap_cos / len(val_loader)

        if (epoch + 1) % 25 == 0 or (epoch + 1) == epochs:
            val_tacos_loss_total = 0.0
            with torch.no_grad():
                for batch_face, batch_flower in val_loader:
                    batch_face, batch_flower = batch_face.to(device), batch_flower.to(device)
                    batch_c_vp = SQRT_05 * batch_face + SQRT_05 * batch_flower
                    pred_z_face, pred_z_flower = diffusion.tacos_sample_loop(batch_c_vp)
                    
                    val_tacos_loss_total += (F.mse_loss(pred_z_face, batch_face) + F.mse_loss(pred_z_flower, batch_flower)).item()
            val_tacos_loss = val_tacos_loss_total / len(val_loader)
            
        log_dict = {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_cheap_loss": avg_val_cheap_loss, "val_cheap_cosine_similarity": avg_val_cheap_cos}
        if val_tacos_loss is not None: log_dict["val_tacos_reconstruct_raw_mse"] = val_tacos_loss
        wandb.log(log_dict)
        
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_cheap_loss:.6f}", f"{avg_val_cheap_cos:.6f}", f"{val_tacos_loss:.6f}" if val_tacos_loss is not None else "N/A"])
            
        checkpoint_data = {'epoch': epoch + 1, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint_data, ckpt_dir / "last.pt")
        
        if avg_val_cheap_loss < best_val_loss:
            best_val_loss = avg_val_cheap_loss
            torch.save(checkpoint_data, ckpt_dir / "best.pt")
            
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Cheap Loss: {avg_val_cheap_loss:.4f} | Val Cheap Cosine Sim: {avg_val_cheap_cos:.4f}" + (f" | Val TACOs MSE: {val_tacos_loss:.4f}" if val_tacos_loss is not None else ""))
        
        if early_stopper(avg_val_cheap_loss):
            print(f"\n[EARLY STOPPING TRIGGERED] Validation profile plateaued. Terminating run.")
            break

    wandb.finish()

# ==========================================
# 6. Evaluation Script
# ==========================================
def evaluate_cross_domain_demorph(face_test_path, flower_test_path, run_name, num_timesteps=300, mode='iterative'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / f"eval_{mode}.txt"
    
    face_test = load_cross_domain_split(face_test_path, "test")
    flower_test = load_cross_domain_split(flower_test_path, "test")
    test_pairs = np.stack([face_test[:1000], flower_test[:1000]], axis=1)
    
    embedding_dim = face_test.shape[-1]
    test_loader = DataLoader(CrossDomainDemorphTestPairsDataset(test_pairs), batch_size=1000, shuffle=False, num_workers=4)

    net = ColdDemorphNet(x_dim=embedding_dim, time_emb_dim=embedding_dim).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=num_timesteps).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    net.eval()

    SQRT_05 = math.sqrt(0.5)
    SQRT_2 = math.sqrt(2.0)

    with torch.no_grad():
        for batch_face, batch_flower in test_loader:
            batch_face, batch_flower = batch_face.to(device), batch_flower.to(device)
            
            batch_c_vp = SQRT_05 * batch_face + SQRT_05 * batch_flower 
            batch_c_true = 0.5 * (batch_face + batch_flower)           
            
            if mode == 'iterative':
                pred_z_face, pred_z_flower = diffusion.tacos_sample_loop(batch_c_vp)
            else:
                pred = net(batch_c_vp, torch.full((batch_face.shape[0],), num_timesteps, device=device).long())
                pred_z_face, _ = pred.chunk(2, dim=-1)
                pred_z_flower = SQRT_2 * batch_c_vp - pred_z_face

            # Direct tracking assignment replaces the old PIT layout
            mse_gt_face = F.mse_loss(pred_z_face, batch_face).item()
            mse_gt_flower = F.mse_loss(pred_z_flower, batch_flower).item()
            cos_gt_face = F.cosine_similarity(pred_z_face, batch_face, dim=-1).mean().item()
            cos_gt_flower = F.cosine_similarity(pred_z_flower, batch_flower, dim=-1).mean().item()

            mse_face_to_c = F.mse_loss(pred_z_face, batch_c_true).item()
            mse_flower_to_c = F.mse_loss(pred_z_flower, batch_c_true).item()
            cos_face_to_c = F.cosine_similarity(pred_z_face, batch_c_true, dim=-1).mean().item()
            cos_flower_to_c = F.cosine_similarity(pred_z_flower, batch_c_true, dim=-1).mean().item()
            
            ref_mse_face_to_c = F.mse_loss(batch_face, batch_c_true).item()
            ref_mse_flower_to_c = F.mse_loss(batch_flower, batch_c_true).item()
            ref_cos_face_to_c = F.cosine_similarity(batch_face, batch_c_true, dim=-1).mean().item()
            ref_cos_flower_to_c = F.cosine_similarity(batch_flower, batch_c_true, dim=-1).mean().item()

            mse_inter_pred = F.mse_loss(pred_z_face, pred_z_flower).item()
            cos_inter_pred = F.cosine_similarity(pred_z_face, pred_z_flower, dim=-1).mean().item()
            
            ref_mse_inter_gt = F.mse_loss(batch_face, batch_flower).item()
            ref_cos_inter_gt = F.cosine_similarity(batch_face, batch_flower, dim=-1).mean().item()

            # Success Rate of Reversal (%S) for cross-domain vectors
            cos_pf_gt = F.cosine_similarity(pred_z_face, batch_face, dim=-1)
            cos_fl_gt = F.cosine_similarity(pred_z_flower, batch_flower, dim=-1)
            cos_pf_c = F.cosine_similarity(pred_z_face, batch_c_true, dim=-1)
            cos_fl_c = F.cosine_similarity(pred_z_flower, batch_c_true, dim=-1)

            pct_S_face = (cos_pf_gt > cos_pf_c).float().mean().item() * 100
            pct_S_flower = (cos_fl_gt > cos_fl_c).float().mean().item() * 100
            pct_S_comb = 0.5 * (pct_S_face + pct_S_flower)

    results_text = (
        f"--- Cross-Domain Evaluation Results: {mode.upper()} ---\nRun Name: {run_name}\n----------------------------------------\n"
        f"[1. Prediction to Ground-Truth Alignment]\n"
        f"  - Face Embedding   -> MSE Distance: {mse_gt_face:.6f} | Cosine Similarity: {cos_gt_face:.6f}\n"
        f"  - Flower Embedding -> MSE Distance: {mse_gt_flower:.6f} | Cosine Similarity: {cos_gt_flower:.6f}\n\n"
        f"[2. Proximity to True Average Mixture (c)]\n"
        f"  - Predicted Face to c   -> MSE Distance: {mse_face_to_c:.6f} | Cosine Similarity: {cos_face_to_c:.6f}\n"
        f"  - Baseline GT Face to c -> MSE Distance: {ref_mse_face_to_c:.6f} | Cosine Similarity: {ref_cos_face_to_c:.6f}\n"
        f"  - Predicted Flower to c   -> MSE Distance: {mse_flower_to_c:.6f} | Cosine Similarity: {cos_flower_to_c:.6f}\n"
        f"  - Baseline GT Flower to c -> MSE Distance: {ref_mse_flower_to_c:.6f} | Cosine Similarity: {ref_cos_flower_to_c:.6f}\n\n"
        f"[3. Predicted Outputs Inter-Relationship / Spread]\n"
        f"  - Inter-Predicted (Face vs Flower) -> MSE: {mse_inter_pred:.6f} | Cosine Similarity: {cos_inter_pred:.6f}\n"
        f"  - Inter-Ground-Truth (Face vs Flower) -> MSE: {ref_mse_inter_gt:.6f} | Cosine Similarity: {ref_cos_inter_gt:.6f}\n\n"
        f"[4. Success Rate of Reversal (%S)]\n"
        f"  - Face Domain   -> %S_Face: {pct_S_face:.2f}%\n"
        f"  - Flower Domain -> %S_Flower: {pct_S_flower:.2f}%\n"
        f"  - Combined Total -> %S: {pct_S_comb:.2f}%\n"
    )
    
    print("\n" + results_text)
    with open(out_file_path, "w") as f: 
        f.write(results_text)

if __name__ == "__main__":
    # Define paths pointing to your dual-domain unCLIP structures
    FACE_TRAIN = "/nas-ctm01/homes/dacordeiro/Face-DM/unclip_embeddings_faces/ffhq256_unclip_zsem_train.npy"
    FLOWER_TRAIN = "/nas-ctm01/homes/dacordeiro/Face-DM/unclip_embeddings_flowers/ffhq256_unclip_zsem_train.npy"
    
    FACE_TEST = "/nas-ctm01/homes/dacordeiro/Face-DM/unclip_embeddings_faces/ffhq256_unclip_zsem_test.npy"
    FLOWER_TEST = "/nas-ctm01/homes/dacordeiro/Face-DM/unclip_embeddings_flowers/ffhq256_unclip_zsem_test.npy"
    
    RUN_NAME = "unclip_cross_domain_baseline"
    
    train_cross_domain_demorph(FACE_TRAIN, FLOWER_TRAIN, FACE_TEST, FLOWER_TEST, RUN_NAME, num_timesteps=300, epochs=150, patience=20)
    evaluate_cross_domain_demorph(FACE_TEST, FLOWER_TEST, RUN_NAME, num_timesteps=300, mode='one_shot')
    evaluate_cross_domain_demorph(FACE_TEST, FLOWER_TEST, RUN_NAME, num_timesteps=300, mode='iterative')

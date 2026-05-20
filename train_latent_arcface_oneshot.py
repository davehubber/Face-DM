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
class OneShotArcFaceDemorphDataset(Dataset):
    """
    Dynamically generates pairs of distinct ArcFace embeddings.
    Applies the sqrt(512) scaling directly at generation.
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
# 2. Optimized One-Shot Architecture
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.silu(self.norm1(self.linear1(x)))
        x = self.dropout(x)
        x = self.norm2(self.linear2(x))
        return self.silu(x + residual)

class OneShotDemorphNet(nn.Module):
    """
    Takes a 512D perfect average embedding 'c' and maps it directly 
    to a 1024D output representing BOTH decoupled clean targets [z1, z2].
    """
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=1024, num_blocks=6):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.silu = nn.SiLU()
        
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])
        
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, c):
        h = self.silu(self.norm_in(self.in_proj(c)))
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h)

# ==========================================
# 3. Permutation-Invariant Optimization Wrapper
# ==========================================
class PermutationInvariantDemorpher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compute_loss(self, c, z1, z2):
        """
        Calculates loss dynamically matching predictions to targets 
        to ensure ordering constraints do not bottleneck convergence.
        """
        # Predict the 1024D coupled vector and split into two 512D tensors
        pred_z12 = self.model(c)
        pred_z1, pred_z2 = pred_z12.chunk(2, dim=-1)
        
        # Choice 1: pred_z1 -> z1, pred_z2 -> z2
        loss_choice1 = F.l1_loss(pred_z1, z1, reduction='none').mean(dim=-1) + \
                       F.l1_loss(pred_z2, z2, reduction='none').mean(dim=-1)
                       
        # Choice 2: pred_z1 -> z2, pred_z2 -> z1
        loss_choice2 = F.l1_loss(pred_z1, z2, reduction='none').mean(dim=-1) + \
                       F.l1_loss(pred_z2, z1, reduction='none').mean(dim=-1)
        
        # Take the minimum penalty path per sample across the batch
        loss = torch.minimum(loss_choice1, loss_choice2).mean()
        return loss

    @torch.no_grad()
    def predict(self, c):
        pred_z12 = self.model(c)
        return pred_z12.chunk(2, dim=-1)

# ==========================================
# 4. Evaluation & Training Loop
# ==========================================
def train_oneshot_demorph(arcface_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])
            
    print("Loading ArcFace embeddings...")
    arcface_path = Path(arcface_path_str).resolve()
    
    scale_factor = np.sqrt(512)
    arcface_embs = (np.load(arcface_path).astype(np.float32)) * scale_factor

    split_idx = int(len(arcface_embs) * 0.9)
    train_embs, val_embs = arcface_embs[:split_idx], arcface_embs[split_idx:]
    
    train_dataset = OneShotArcFaceDemorphDataset(train_embs, epoch_size=1_000_000, deterministic=False)
    val_dataset = OneShotArcFaceDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    
    train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=8192, shuffle=False, num_workers=4)

    net = OneShotDemorphNet(in_dim=512, hidden_dim=2048, out_dim=1024, num_blocks=6).to(device)
    wrapper = PermutationInvariantDemorpher(net).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 2e-4,
        "batch_size": 8192,
        "num_blocks": 6,
        "hidden_dim": 2048,
        "latent_space": "ArcFace",
        "type": "One-Shot Permutation-Invariant",
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
            batch_c = (batch_z1 + batch_z2) / 2.0
            
            optimizer.zero_grad()
            loss = wrapper.compute_loss(batch_c, batch_z1, batch_z2)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"PIT_L1_loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        
        # Validation
        net.eval()
        val_loss_total = 0.0
        
        with torch.no_grad():
            for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                batch_c = (batch_z1 + batch_z2) / 2.0
                
                loss_val = wrapper.compute_loss(batch_c, batch_z1, batch_z2)
                val_loss_total += loss_val.item()
                
        avg_val_loss = val_loss_total / len(val_loader)
        
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}"])
            
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }
        
        torch.save(checkpoint_data, ckpt_dir / "last.pt")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint_data, ckpt_dir / "best.pt")
            print(f"--> Saved best checkpoint to {ckpt_dir / 'best.pt'}")
            
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    wandb.finish()

def evaluate_oneshot_demorph(arcface_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    out_file_path = exp_dir / "eval_oneshot.txt"
    
    print("Loading ArcFace embeddings for evaluation...")
    scale_factor = math.sqrt(512)
    arcface_embs = (np.load(Path(arcface_path_str).resolve()).astype(np.float32)) * scale_factor
    split_idx = int(len(arcface_embs) * 0.9)
    val_embs = arcface_embs[split_idx:]
    
    val_dataset = OneShotArcFaceDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)

    net = OneShotDemorphNet(in_dim=512, hidden_dim=2048, out_dim=1024, num_blocks=6).to(device)
    wrapper = PermutationInvariantDemorpher(net).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    total_l1_scaled = 0.0
    total_l1_raw = 0.0
    total_cosine = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in tqdm(val_loader, desc="Evaluating One-Shot"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = (batch_z1 + batch_z2) / 2.0
            
            # Extract predictions
            pred_z1, pred_z2 = wrapper.predict(batch_c)
            
            # Since the model is permutation invariant, we need to find which prediction 
            # maps closest to which target for every item in the batch to calculate accurate metrics.
            for i in range(batch_z1.shape[0]):
                p1, p2 = pred_z1[i], pred_z2[i]
                t1, t2 = batch_z1[i], batch_z2[i]
                
                # Check alignment configurations
                dist_opt1 = F.l1_loss(p1, t1) + F.l1_loss(p2, t2)
                dist_opt2 = F.l1_loss(p1, t2) + F.l1_loss(p2, t1)
                
                # Assign aligned pairs based on optimal distance
                if dist_opt1 < dist_opt2:
                    a_p1, a_p2 = p1, p2
                else:
                    a_p1, a_p2 = p2, p1
                
                # 1. Scaled Metrics
                total_l1_scaled += (F.l1_loss(a_p1, t1).item() + F.l1_loss(a_p2, t2).item()) / 2.0
                
                # 2. Map back to strictly L2-normalized domain for Raw Hypersphere space assessment
                p1_raw = F.normalize(a_p1 / scale_factor, p=2, dim=-1)
                p2_raw = F.normalize(a_p2 / scale_factor, p=2, dim=-1)
                t1_raw = F.normalize(t1 / scale_factor, p=2, dim=-1)
                t2_raw = F.normalize(t2 / scale_factor, p=2, dim=-1)
                
                total_l1_raw += (F.l1_loss(p1_raw, t1_raw).item() + F.l1_loss(p2_raw, t2_raw).item()) / 2.0
                total_cosine += (F.cosine_similarity(p1_raw, t1_raw, dim=0).item() + F.cosine_similarity(p2_raw, t2_raw, dim=0).item()) / 2.0
                
                num_batches += 1

    avg_l1_scaled = total_l1_scaled / num_batches
    avg_l1_raw = total_l1_raw / num_batches
    avg_cosine = total_cosine / num_batches

    results_text = (
        f"--- Evaluation Results: ONE-SHOT PERMUTATION INVARIANT ---\n"
        f"Run Name: {run_name}\n"
        f"Validation Pairs Evaluated: {num_batches}\n"
        f"---------------------------------------------------------\n"
        f"Mean L1 Distance (Scaled space): {avg_l1_scaled:.6f}\n"
        f"Mean L1 Distance (Raw ArcFace):  {avg_l1_raw:.6f}\n"
        f"Mean Cosine Similarity:          {avg_cosine:.6f}\n"
    )
    
    print("\n" + results_text)
    with open(out_file_path, "w") as f:
        f.write(results_text)

if __name__ == "__main__":
    DATA_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy"
    RUN_ID = "oneshot_arcface_pit"

    train_oneshot_demorph(arcface_path_str=DATA_PATH, run_name=RUN_ID)
    evaluate_oneshot_demorph(arcface_path_str=DATA_PATH, run_name=RUN_ID)

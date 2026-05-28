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
class OneShotDiffAEDemorphDataset(Dataset):
    """
    Dynamically generates pairs of distinct Diff-AE semantic embeddings.
    Operates entirely in the pre-computed Z-score normalized space.
    """
    def __init__(self, embeddings: np.ndarray, epoch_size: int = 1000000, deterministic: bool = False):
        self.embeddings = embeddings
        self.epoch_size = epoch_size
        self.deterministic = deterministic
        self.num_samples = len(embeddings)
        
        if self.deterministic:
            np.random.seed(42)
            # Ensure static pairing matrices for evaluation consistency
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
    Maps a 512D clean perfect average condition 'c' directly 
    to a 1024D output tensor containing both estimated targets [z1, z2].
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
# 3. Permutation-Invariant Training Wrapper
# ==========================================
class PermutationInvariantDemorpher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compute_loss(self, c, z1, z2):
        """
        Calculates L1 loss in a permutation-invariant manner (PIT), preventing
        the target matching sequence order from bottlenecking head differentiation.
        """
        # Predict the 1024D coupled vector and split into two 512D chunks
        pred_z12 = self.model(c)
        pred_z1, pred_z2 = pred_z12.chunk(2, dim=-1)
        
        # Scenario A: Head 1 -> z1, Head 2 -> z2
        loss_head_a = F.l1_loss(pred_z1, z1, reduction='none').mean(dim=-1) + \
                      F.l1_loss(pred_z2, z2, reduction='none').mean(dim=-1)
                       
        # Scenario B: Head 1 -> z2, Head 2 -> z1
        loss_head_b = F.l1_loss(pred_z1, z2, reduction='none').mean(dim=-1) + \
                      F.l1_loss(pred_z2, z1, reduction='none').mean(dim=-1)
        
        # Isolate minimum penalty per assignment across the batch axis
        loss = torch.minimum(loss_head_a, loss_head_b).mean()
        return loss

    @torch.no_grad()
    def predict(self, c):
        pred_z12 = self.model(c)
        return pred_z12.chunk(2, dim=-1)

# ==========================================
# 4. Evaluation & Training Loop
# ==========================================
def load_and_zscore_diffae(diffae_path_str: str) -> np.ndarray:
    diffae_path = Path(diffae_path_str).resolve()
    print(f"Loading DiffAE embeddings from: {diffae_path}")
    raw_diffae = np.load(diffae_path).astype(np.float32)
    
    mean_path = diffae_path.parent / f"{diffae_path.stem}_mean.npy"
    std_path = diffae_path.parent / f"{diffae_path.stem}_std.npy"
    
    if mean_path.exists() and std_path.exists():
        zscore_mean = np.load(mean_path).astype(np.float32)
        zscore_std = np.load(std_path).astype(np.float32)
        diffae_embs = (raw_diffae - zscore_mean) / zscore_std
        print("Z-score normalization applied successfully.")
        return diffae_embs
    else:
        raise FileNotFoundError(f"Missing Z-score statistics: {mean_path.name} or {std_path.name}")

def train_oneshot_diffae(diffae_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])
            
    # Process inputs directly in standard Z-score space (No manual sqrt scaling required)
    diffae_embs = load_and_zscore_diffae(diffae_path_str)

    split_idx = int(len(diffae_embs) * 0.9)
    train_embs, val_embs = diffae_embs[:split_idx], diffae_embs[split_idx:]
    
    train_dataset = OneShotDiffAEDemorphDataset(train_embs, epoch_size=1_000_000, deterministic=False)
    val_dataset = OneShotDiffAEDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16384, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)

    net = OneShotDemorphNet(in_dim=512, hidden_dim=2048, out_dim=1024, num_blocks=6).to(device)
    wrapper = PermutationInvariantDemorpher(net).to(device)
    
    # Specified optimization setup at 3e-4
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 3e-4,
        "batch_size": 16384,
        "num_blocks": 6,
        "hidden_dim": 2048,
        "latent_space": "Diff-AE",
        "type": "One-Shot Permutation-Invariant (PIT)"
    })

    epochs = 50 
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_z1, batch_z2 in pbar:
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = (batch_z1 + batch_z2) / math.sqrt(2)
            
            optimizer.zero_grad()
            loss = wrapper.compute_loss(batch_c, batch_z1, batch_z2)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"PIT_L1_loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        
        # Validation Loop
        net.eval()
        val_loss_total = 0.0
        
        with torch.no_grad():
            for batch_z1, batch_z2 in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                batch_c = (batch_z1 + batch_z2) / math.sqrt(2)
                
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

def evaluate_oneshot_diffae(diffae_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    out_file_path = exp_dir / "eval_oneshot.txt"
    
    print("Loading DiffAE embeddings for evaluation...")
    diffae_embs = load_and_zscore_diffae(diffae_path_str)
    split_idx = int(len(diffae_embs) * 0.9)
    val_embs = diffae_embs[split_idx:]
    
    val_dataset = OneShotDiffAEDemorphDataset(val_embs, epoch_size=10_000, deterministic=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)

    net = OneShotDemorphNet(in_dim=512, hidden_dim=2048, out_dim=1024, num_blocks=6).to(device)
    wrapper = PermutationInvariantDemorpher(net).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    total_l1 = 0.0
    total_cosine = 0.0
    num_samples_evaluated = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in tqdm(val_loader, desc="Evaluating One-Shot"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            batch_c = (batch_z1 + batch_z2) / math.sqrt(2)
            
            # Extract predictions [batch, 512]
            pred_z1, pred_z2 = wrapper.predict(batch_c)
            
            # Realign predictions locally on a per-sample basis for downstream metrics
            for i in range(batch_z1.shape[0]):
                p1, p2 = pred_z1[i], pred_z2[i]
                t1, t2 = batch_z1[i], batch_z2[i]
                
                dist_opt1 = F.l1_loss(p1, t1) + F.l1_loss(p2, t2)
                dist_opt2 = F.l1_loss(p1, t2) + F.l1_loss(p2, t1)
                
                if dist_opt1 < dist_opt2:
                    a_p1, a_p2 = p1, p2
                else:
                    a_p1, a_p2 = p2, p1
                
                # Metrics calculations in standard continuous coordinate space
                total_l1 += (F.l1_loss(a_p1, t1).item() + F.l1_loss(a_p2, t2).item()) / 2.0
                total_cosine += (F.cosine_similarity(a_p1, t1, dim=0).item() + F.cosine_similarity(a_p2, t2, dim=0).item()) / 2.0
                
                num_samples_evaluated += 1

    avg_l1 = total_l1 / num_samples_evaluated
    avg_cosine = total_cosine / num_samples_evaluated

    results_text = (
        f"--- Evaluation Results: ONE-SHOT DIFF-AE (PIT) ---\n"
        f"Run Name: {run_name}\n"
        f"Validation Pairs Evaluated: {num_samples_evaluated}\n"
        f"---------------------------------------------------\n"
        f"Mean L1 Distance:  {avg_l1:.6f}\n"
        f"Mean Cosine Similarity: {avg_cosine:.6f}\n"
    )
    
    print("\n" + results_text)
    with open(out_file_path, "w") as f:
        f.write(results_text)

if __name__ == "__main__":
    DATA_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    RUN_ID = "oneshot_diffae_real"

    train_oneshot_diffae(diffae_path_str=DATA_PATH, run_name=RUN_ID)
    evaluate_oneshot_diffae(diffae_path_str=DATA_PATH, run_name=RUN_ID)

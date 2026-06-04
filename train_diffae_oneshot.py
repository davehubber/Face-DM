import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
from tqdm import tqdm

# ==========================================
# 1. Dataset & Data Loading
# ==========================================
class DirectDemorphDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, epoch_size: int = 1000000):
        self.embeddings = embeddings
        self.epoch_size = epoch_size
        self.num_samples = len(embeddings)
        
        print(f"Generating {epoch_size} unique, order-agnostic training pairs...")
        pair_set = set()
        
        # Fast rejection sampling to get strictly unique pairs where i < j
        while len(pair_set) < epoch_size:
            needed = epoch_size - len(pair_set)
            i = np.random.randint(0, self.num_samples, size=needed * 2)
            j = np.random.randint(0, self.num_samples, size=needed * 2)
            
            valid = i != j
            i, j = i[valid], j[valid]
            
            pairs = np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1)
            pair_set.update(map(tuple, pairs.tolist()))
            
        self.pairs = np.array(list(pair_set)[:epoch_size])
        print("Dataset pairs generated successfully.")

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        return (
            torch.tensor(self.embeddings[idx1], dtype=torch.float32), 
            torch.tensor(self.embeddings[idx2], dtype=torch.float32)
        )

def load_split_and_normalize(base_path_str: str, split: str) -> np.ndarray:
    base_path = Path(base_path_str).resolve()
    parent = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_val", "").replace("_test", "")
    
    split_path = parent / f"{stem}_{split}.npy"
    mean_path = parent / f"{stem}_train_mean.npy"
    std_path = parent / f"{stem}_train_std.npy"
    
    data = np.load(split_path).astype(np.float32)
    if mean_path.exists() and std_path.exists():
        mean = np.load(mean_path).astype(np.float32)
        std = np.load(std_path).astype(np.float32)
        data = (data - mean) / std
    else:
        raise FileNotFoundError(f"Normalization statistics missing at {mean_path} or {std_path}")
    return data

# ==========================================
# 2. Baseline Network Architecture
# ==========================================
class BaselineBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.norm(self.linear(x)))

class BaselineDemorphNet(nn.Module):
    def __init__(self, x_dim=512, hidden_dim=2048, num_layers=10):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        # First layer takes 512, outputs hidden_dim
        self.blocks.append(BaselineBlock(x_dim, hidden_dim))
        
        # Subsequent layers concatenate the original 512 input to prevent signal decay
        for _ in range(num_layers - 1):
            self.blocks.append(BaselineBlock(hidden_dim + x_dim, hidden_dim))
            
        self.final_linear = nn.Linear(hidden_dim, x_dim * 2)

    def forward(self, x):
        h = x
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h)
            else:
                h = block(torch.cat([h, x], dim=-1))
                
        out = self.final_linear(h)
        pred_z1, pred_z2 = out.chunk(2, dim=-1)
        return pred_z1, pred_z2

# ==========================================
# 3. Loss & Training Mechanics
# ==========================================
def compute_permutation_invariant_loss(pred_z1, pred_z2, target_z1, target_z2):
    loss_a = F.l1_loss(pred_z1, target_z1, reduction='none').mean(dim=-1) + \
             F.l1_loss(pred_z2, target_z2, reduction='none').mean(dim=-1)
             
    loss_b = F.l1_loss(pred_z1, target_z2, reduction='none').mean(dim=-1) + \
             F.l1_loss(pred_z2, target_z1, reduction='none').mean(dim=-1)
             
    loss = torch.min(loss_a, loss_b)
    return loss.mean()

def train_baseline_demorph(diffae_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    # 1. Load Data
    train_embs = load_split_and_normalize(diffae_path_str, "train")
    val_embs = load_split_and_normalize(diffae_path_str, "val")
    
    train_loader = DataLoader(DirectDemorphDataset(train_embs, epoch_size=1_000_000), batch_size=20_000, shuffle=True, num_workers=8)
    val_loader = DataLoader(DirectDemorphDataset(val_embs, epoch_size=10_000), batch_size=10_000, shuffle=False, num_workers=4)

    # 2. Init Baseline Network
    net = BaselineDemorphNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4, "batch_size": 20_000, "num_layers": 10, 
        "hidden_dim": 2048, "architecture": "direct_predict_baseline"
    })

    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Train_Loss", "Val_Loss"])

    epochs = 50
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        
        for batch_z1, batch_z2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            optimizer.zero_grad()
            
            c = (batch_z1 + batch_z2) / 2.0
            pred_z1, pred_z2 = net(c)
            loss = compute_permutation_invariant_loss(pred_z1, pred_z2, batch_z1, batch_z2)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Pass
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_z1, batch_z2 in val_loader:
                batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                c = (batch_z1 + batch_z2) / 2.0
                
                pred_z1, pred_z2 = net(c)
                val_loss += compute_permutation_invariant_loss(pred_z1, pred_z2, batch_z1, batch_z2).item()
                
        avg_val_loss = val_loss / len(val_loader)
            
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{avg_val_loss:.6f}"])
            
        checkpoint_data = {'epoch': epoch + 1, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint_data, ckpt_dir / "last.pt")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint_data, ckpt_dir / "best.pt")
            
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    wandb.finish()

# ==========================================
# 4. Evaluation Script
# ==========================================
def evaluate_baseline_demorph(diffae_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / "eval_baseline.txt"
    
    test_embs = load_split_and_normalize(diffae_path_str, "test")
    test_loader = DataLoader(DirectDemorphDataset(test_embs, epoch_size=10_000), batch_size=10_000, shuffle=False, num_workers=4)

    net = BaselineDemorphNet().to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    net.eval()

    total_l1_1, total_l1_2, total_cosine_1, total_cosine_2 = 0.0, 0.0, 0.0, 0.0
    total_cos_z1_z2, total_cos_pred1_pred2 = 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in test_loader:
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            c = (batch_z1 + batch_z2) / 2.0
            
            pred_z1, pred_z2 = net(c)

            dist_a = F.l1_loss(pred_z1, batch_z1, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z2, reduction='none').mean(dim=1)
            dist_b = F.l1_loss(pred_z1, batch_z2, reduction='none').mean(dim=1) + F.l1_loss(pred_z2, batch_z1, reduction='none').mean(dim=1)
            mask_a = (dist_a <= dist_b).unsqueeze(-1)
            
            aligned_pred_z1 = torch.where(mask_a, pred_z1, pred_z2)
            aligned_pred_z2 = torch.where(mask_a, pred_z2, pred_z1)

            total_l1_1 += F.l1_loss(aligned_pred_z1, batch_z1).item()
            total_l1_2 += F.l1_loss(aligned_pred_z2, batch_z2).item()
            total_cosine_1 += F.cosine_similarity(aligned_pred_z1, batch_z1, dim=-1).mean().item()
            total_cosine_2 += F.cosine_similarity(aligned_pred_z2, batch_z2, dim=-1).mean().item()
            
            total_cos_z1_z2 += F.cosine_similarity(batch_z1, batch_z2, dim=-1).mean().item()
            total_cos_pred1_pred2 += F.cosine_similarity(aligned_pred_z1, aligned_pred_z2, dim=-1).mean().item()
            
            num_batches += 1

    results_text = (
        f"--- Evaluation Results: BASELINE (NO COND) ---\nRun Name: {run_name}\n----------------------------------------\n"
        f"Base Reference Angular Properties:\n"
        f"  - CosSim(True Baseline z1, True Baseline z2): {total_cos_z1_z2 / num_batches:.6f}\n\n"
        f"[Embedding 1 Performance Alignment]\n"
        f"  - L1 Distance:                       {total_l1_1 / num_batches:.6f}\n"
        f"  - Cosine Similarity to Target:       {total_cosine_1 / num_batches:.6f}\n\n"
        f"[Embedding 2 Performance Alignment]\n"
        f"  - L1 Distance:                       {total_l1_2 / num_batches:.6f}\n"
        f"  - Cosine Similarity to Target:       {total_cosine_2 / num_batches:.6f}\n\n"
        f"Generated Outputs Inter-Relationship:\n"
        f"  - CosSim(Aligned Pred z1, Aligned Pred z2):  {total_cos_pred1_pred2 / num_batches:.6f}\n"
    )
    print("\n" + results_text)
    with open(out_file_path, "w") as f: f.write(results_text)

if __name__ == "__main__":
    BASE_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    RUN_NAME = "diffae_oneshot_baseline"
    
    train_baseline_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME)
    evaluate_baseline_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME)

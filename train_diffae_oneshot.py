import math
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
from tqdm import tqdm
import joblib

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
            # Oversample slightly to account for duplicates/self-pairs
            i = np.random.randint(0, self.num_samples, size=needed * 2)
            j = np.random.randint(0, self.num_samples, size=needed * 2)
            
            # Remove self pairs
            valid = i != j
            i, j = i[valid], j[valid]
            
            # Enforce order (i < j) to make pairs order-agnostic (z1, z2 == z2, z1)
            pairs = np.stack([np.minimum(i, j), np.maximum(i, j)], axis=1)
            
            # Update set (automatically drops duplicates)
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
# 2. Network Architecture
# ==========================================
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

class DirectDemorphNet(nn.Module):
    def __init__(self, cond_input_dim: int, x_dim=512, hidden_dim=2048, num_layers=10, cond_hidden_dim=512):
        super().__init__()
        
        # MLP to process the [L2_Norm, PC_Scores] vector
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, cond_hidden_dim),
            nn.SiLU(),
            nn.Linear(cond_hidden_dim, cond_hidden_dim)
        )
        
        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(x_dim, hidden_dim, cond_hidden_dim))
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + x_dim, hidden_dim, cond_hidden_dim))
            
        # Network predicts BOTH z1 and z2 simultaneously (output dim = 1024)
        self.final_linear = nn.Linear(hidden_dim, x_dim * 2)

    def forward(self, x, cond_vec):
        cond_emb = self.cond_mlp(cond_vec)
        h = x
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, cond_emb)
            else:
                h = block(torch.cat([h, x], dim=-1), cond_emb)
                
        out = self.final_linear(h)
        # Split the 1024-d output into two 512-d predictions
        pred_z1, pred_z2 = out.chunk(2, dim=-1)
        return pred_z1, pred_z2

# ==========================================
# 3. Loss & Training Mechanics
# ==========================================
def compute_permutation_invariant_loss(pred_z1, pred_z2, target_z1, target_z2):
    """
    Computes L1 loss dynamically allowing pred_z1 to match either target_z1 or target_z2.
    """
    # Option A: pred1 matches target1, pred2 matches target2
    loss_a = F.l1_loss(pred_z1, target_z1, reduction='none').mean(dim=-1) + \
             F.l1_loss(pred_z2, target_z2, reduction='none').mean(dim=-1)
             
    # Option B: pred1 matches target2, pred2 matches target1
    loss_b = F.l1_loss(pred_z1, target_z2, reduction='none').mean(dim=-1) + \
             F.l1_loss(pred_z2, target_z1, reduction='none').mean(dim=-1)
             
    # Take the minimum valid pairing per batch item
    loss = torch.min(loss_a, loss_b)
    return loss.mean()

def load_pca_tensors(data_dir: Path, prefix: str, device: torch.device):
    """Finds and loads the PCA model, returning its components as GPU tensors."""
    # Find the joblib file dynamically
    pca_files = list(data_dir.glob(f"{prefix}_pca_model_*comp.joblib"))
    if not pca_files:
        raise FileNotFoundError(f"No PCA joblib model found for prefix {prefix} in {data_dir}")
        
    pca_path = pca_files[0]
    print(f"Loading PCA model for conditioning: {pca_path.name}")
    pca = joblib.load(pca_path)
    
    # Extract matrices to GPU
    pca_components = torch.tensor(pca.components_, dtype=torch.float32, device=device)
    pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
    n_components = pca_components.shape[0]
    
    return pca_components, pca_mean, n_components

def train_direct_demorph(diffae_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    base_path = Path(diffae_path_str).resolve()
    data_dir = base_path.parent
    prefix = base_path.stem.replace("_train", "").replace("_val", "").replace("_test", "")
    
    # 1. Load Data & PCA
    pca_components, pca_mean, n_components = load_pca_tensors(data_dir, prefix, device)
    cond_input_dim = 1 + n_components # [L2 Norm] + [PC Scores]
    
    train_embs = load_split_and_normalize(diffae_path_str, "train")
    val_embs = load_split_and_normalize(diffae_path_str, "val")
    
    train_loader = DataLoader(DirectDemorphDataset(train_embs, epoch_size=1_000_000), batch_size=25_000, shuffle=True, num_workers=8)
    val_loader = DataLoader(DirectDemorphDataset(val_embs, epoch_size=10_000), batch_size=10_000, shuffle=False, num_workers=4)

    # 2. Init Network
    net = DirectDemorphNet(cond_input_dim=cond_input_dim).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4, "batch_size": 25_000, "num_layers": 10, 
        "hidden_dim": 2048, "cond_dim": cond_input_dim, "architecture": "direct_predict"
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
            
            # Compute Average
            c = (batch_z1 + batch_z2) / 2.0
            
            # Compute Conditioning Vector
            c_norm = torch.norm(c, p=2, dim=1, keepdim=True)
            pc_scores = torch.matmul(c - pca_mean, pca_components.T)
            cond_vec = torch.cat([c_norm, pc_scores], dim=1)
            
            # Forward & Loss
            pred_z1, pred_z2 = net(c, cond_vec)
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
                
                c_norm = torch.norm(c, p=2, dim=1, keepdim=True)
                pc_scores = torch.matmul(c - pca_mean, pca_components.T)
                cond_vec = torch.cat([c_norm, pc_scores], dim=1)
                
                pred_z1, pred_z2 = net(c, cond_vec)
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
def evaluate_direct_demorph(diffae_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    out_file_path = exp_dir / "eval_direct.txt"
    
    base_path = Path(diffae_path_str).resolve()
    pca_components, pca_mean, n_components = load_pca_tensors(base_path.parent, base_path.stem.replace("_train", "").replace("_val", "").replace("_test", ""), device)
    
    test_embs = load_split_and_normalize(diffae_path_str, "test")
    test_loader = DataLoader(DirectDemorphDataset(test_embs, epoch_size=10_000), batch_size=10_000, shuffle=False, num_workers=4)

    net = DirectDemorphNet(cond_input_dim=1 + n_components).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    net.eval()

    total_l1_1, total_l1_2, total_cosine_1, total_cosine_2 = 0.0, 0.0, 0.0, 0.0
    total_cos_z1_z2, total_cos_pred1_pred2 = 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_z1, batch_z2 in test_loader:
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            c = (batch_z1 + batch_z2) / 2.0
            
            c_norm = torch.norm(c, p=2, dim=1, keepdim=True)
            pc_scores = torch.matmul(c - pca_mean, pca_components.T)
            cond_vec = torch.cat([c_norm, pc_scores], dim=1)
            
            pred_z1, pred_z2 = net(c, cond_vec)

            # Route predictions to targets to track metrics linearly
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
        f"--- Evaluation Results: DIRECT PREDICTION ---\nRun Name: {run_name}\n----------------------------------------\n"
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
    RUN_NAME = "diffae_oneshot_pca_cond"
    
    train_direct_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME)
    evaluate_direct_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME)
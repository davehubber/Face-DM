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
        self.blocks.append(BaselineBlock(x_dim, hidden_dim))
        
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
# 3. Penalized Loss & Training Mechanics
# ==========================================
def compute_penalized_loss(pred_z1, pred_z2, target_z1, target_z2, c, lambda_rep=0.5, lambda_cyc=1.0, margin=0.1):
    # 1. Base Permutation Invariant L1 Loss
    loss_a = F.l1_loss(pred_z1, target_z1, reduction='none').mean(dim=-1) + \
             F.l1_loss(pred_z2, target_z2, reduction='none').mean(dim=-1)
             
    loss_b = F.l1_loss(pred_z1, target_z2, reduction='none').mean(dim=-1) + \
             F.l1_loss(pred_z2, target_z1, reduction='none').mean(dim=-1)
             
    loss_base = torch.min(loss_a, loss_b).mean()
    
    # 2. Inter-Latent Repulsion (Cosine Similarity Penalty)
    # Penalize if the cosine similarity between predictions is greater than the margin
    cos_sim = F.cosine_similarity(pred_z1, pred_z2, dim=-1)
    loss_repulsion = F.relu(cos_sim - margin).mean()
    
    # 3. Cycle Consistency (Anchor Penalty)
    # Ensure the midpoint of predictions strictly equals the input morph 'c'
    pred_c = (pred_z1 + pred_z2) / 2.0
    loss_cycle = F.l1_loss(pred_c, c)
    
    # Total Loss
    total_loss = loss_base + (lambda_rep * loss_repulsion) + (lambda_cyc * loss_cycle)
    
    return total_loss, loss_base, loss_repulsion, loss_cycle

def train_baseline_demorph(diffae_path_str: str, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.csv"
    
    # Hyperparameters for new losses
    LAMBDA_REPULSION = 0.5
    LAMBDA_CYCLE = 1.0
    REPULSION_MARGIN = 0.1
    
    train_embs = load_split_and_normalize(diffae_path_str, "train")
    val_embs = load_split_and_normalize(diffae_path_str, "val")
    
    train_loader = DataLoader(DirectDemorphDataset(train_embs, epoch_size=1_000_000), batch_size=20_000, shuffle=True, num_workers=8)
    val_loader = DataLoader(DirectDemorphDataset(val_embs, epoch_size=10_000), batch_size=10_000, shuffle=False, num_workers=4)

    net = BaselineDemorphNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.01)
    
    wandb.init(project="Face-DM", name=run_name, dir=str(exp_dir), config={
        "learning_rate": 1e-4, "batch_size": 20_000, "num_layers": 10, 
        "hidden_dim": 2048, "architecture": "direct_predict_penalized",
        "lambda_repulsion": LAMBDA_REPULSION, "lambda_cycle": LAMBDA_CYCLE,
        "repulsion_margin": REPULSION_MARGIN
    })

    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["Epoch", "Total_Loss", "Base_L1", "Repulsion_Loss", "Cycle_Loss", "Val_Loss"])

    epochs = 50
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        net.train()
        epoch_total, epoch_base, epoch_rep, epoch_cyc = 0.0, 0.0, 0.0, 0.0
        
        for batch_z1, batch_z2 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            optimizer.zero_grad()
            
            c = (batch_z1 + batch_z2) / 2.0
            pred_z1, pred_z2 = net(c)
            
            loss, l_base, l_rep, l_cyc = compute_penalized_loss(
                pred_z1, pred_z2, batch_z1, batch_z2, c, 
                lambda_rep=LAMBDA_REPULSION, lambda_cyc=LAMBDA_CYCLE, margin=REPULSION_MARGIN
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_total += loss.item()
            epoch_base += l_base.item()
            epoch_rep += l_rep.item()
            epoch_cyc += l_cyc.item()
            
        num_batches = len(train_loader)
        avg_total = epoch_total / num_batches
        avg_base = epoch_base / num_batches
        avg_rep = epoch_rep / num_batches
        avg_cyc = epoch_cyc / num_batches
        
        # Validation Pass
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_z1, batch_z2 in val_loader:
                batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
                c = (batch_z1 + batch_z2) / 2.0
                
                pred_z1, pred_z2 = net(c)
                v_loss, _, _, _ = compute_penalized_loss(
                    pred_z1, pred_z2, batch_z1, batch_z2, c, 
                    lambda_rep=LAMBDA_REPULSION, lambda_cyc=LAMBDA_CYCLE, margin=REPULSION_MARGIN
                )
                val_loss += v_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
            
        wandb.log({
            "epoch": epoch + 1, "train_total_loss": avg_total, 
            "train_base_L1": avg_base, "train_repulsion": avg_rep, 
            "train_cycle": avg_cyc, "val_total_loss": avg_val_loss
        })
        
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{avg_total:.6f}", f"{avg_base:.6f}", f"{avg_rep:.6f}", f"{avg_cyc:.6f}", f"{avg_val_loss:.6f}"])
            
        checkpoint_data = {'epoch': epoch + 1, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint_data, ckpt_dir / "last.pt")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint_data, ckpt_dir / "best.pt")
            
        print(f"Epoch {epoch+1} | Total: {avg_total:.4f} (Base: {avg_base:.4f} | Rep: {avg_rep:.4f} | Cyc: {avg_cyc:.4f}) | Val: {avg_val_loss:.4f}")
        
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
    RUN_NAME = "diffae_oneshot_penalized"
    
    train_baseline_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME)
    evaluate_baseline_demorph(diffae_path_str=BASE_PATH, run_name=RUN_NAME)

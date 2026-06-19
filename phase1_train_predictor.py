import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
from tqdm import tqdm
from pathlib import Path

# ==========================================
# 1. Dataset (Same as your original)
# ==========================================
class ConditionalMorphDataset(Dataset):
    def __init__(self, morph_embs: np.ndarray, bf_embs: np.ndarray, metadata_path: Path):
        self.morph_embs = morph_embs
        self.bf_embs = bf_embs
        self.records = []
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx_m = int(row['embedding_index'])
                col_a = 'source_idx_A_in_train_bf' if 'source_idx_A_in_train_bf' in row else 'source_idx_A_in_eval_bf'
                col_b = 'source_idx_B_in_train_bf' if 'source_idx_B_in_train_bf' in row else 'source_idx_B_in_eval_bf'
                
                idx_a = int(row[col_a])
                idx_b = int(row[col_b])
                
                if idx_a >= 0 and idx_b >= 0:
                    self.records.append((idx_m, idx_a, idx_b))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        idx_m, idx_a, idx_b = self.records[idx]
        M = self.morph_embs[idx_m]
        A = self.bf_embs[idx_a]
        B = self.bf_embs[idx_b]

        # Symmetric Training
        if torch.rand(1).item() > 0.5:
            condition, target = A, B
        else:
            condition, target = B, A

        return (
            torch.tensor(M, dtype=torch.float32),
            torch.tensor(condition, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )

# ==========================================
# 2. Coarse Predictor Network
# ==========================================
class CoarsePredictorNet(nn.Module):
    """
    A robust MLP to unmix the morph. 
    Concatenates Morph and Condition to predict the Target.
    """
    def __init__(self, dim=512, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GroupNorm(32, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, dim)
        )

    def forward(self, morph, condition):
        x = torch.cat([morph, condition], dim=-1)
        return self.net(x)

# ==========================================
# 3. Early Stopping
# ==========================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# ==========================================
# 4. Training Loop
# ==========================================
def train_phase1(data_dir: str, run_name: str, epochs=150, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    root = Path(data_dir)

    print("Loading data...")
    train_bf = np.load(root / "train_bonafide_zsem.npy").astype(np.float32)
    train_morphs = np.load(root / "train_morph_zsem.npy").astype(np.float32)
    
    global_mean = train_bf.mean(axis=0, keepdims=True)
    global_std = train_bf.std(axis=0, keepdims=True) + 1e-8
    
    train_bf = (train_bf - global_mean) / global_std
    train_morphs = (train_morphs - global_mean) / global_std
    
    eval_bf = np.load(root / "eval_bonafide_zsem.npy").astype(np.float32)
    eval_morphs = np.load(root / "eval_morph_zsem.npy").astype(np.float32)
    eval_bf = (eval_bf - global_mean) / global_std
    eval_morphs = (eval_morphs - global_mean) / global_std

    train_loader = DataLoader(ConditionalMorphDataset(train_morphs, train_bf, root / "train_morph_metadata.csv"), batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(ConditionalMorphDataset(eval_morphs, eval_bf, root / "eval_morph_metadata.csv"), batch_size=1000, shuffle=False, num_workers=4)

    net = CoarsePredictorNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    early_stopper = EarlyStopping(patience=100)

    wandb.init(project="Face-DM-Phase1", name=run_name, config={"batch_size": batch_size})
    best_val_loss = float("inf")

    for epoch in range(epochs):
        net.train()
        train_loss = 0.0
        for M, cond, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            M, cond, tgt = M.to(device), cond.to(device), tgt.to(device)
            optimizer.zero_grad()
            
            pred_tgt = net(M, cond)
            loss = F.mse_loss(pred_tgt, tgt) # MSE works better for latent optimization
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        net.eval()
        val_loss, val_l1 = 0.0, 0.0
        with torch.no_grad():
            for M, cond, tgt in val_loader:
                M, cond, tgt = M.to(device), cond.to(device), tgt.to(device)
                pred_tgt = net(M, cond)
                val_loss += F.mse_loss(pred_tgt, tgt).item()
                val_l1 += F.l1_loss(pred_tgt, tgt).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_l1 = val_l1 / len(val_loader)

        wandb.log({"train_loss_mse": avg_train_loss, "val_loss_mse": avg_val_loss, "val_loss_l1": avg_val_l1})
        print(f"Epoch {epoch+1} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f} | Val L1: {avg_val_l1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), ckpt_dir / "predictor_best.pt")

        if early_stopper(avg_val_loss):
            print("Early stopping triggered.")
            break

    wandb.finish()

if __name__ == "__main__":
    DATA_DIR = "/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings"
    train_phase1(DATA_DIR, "phase1_coarse_predictor")

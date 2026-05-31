import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==========================================
# 1. Dataset Definition
# ==========================================
class EmbeddingDataset(Dataset):
    def __init__(self, diffae_path: Path, arcface_path: Path, mean_path: Path, std_path: Path):
        """
        Loads the paired embeddings and strictly applies Z-score normalization 
        using the pre-computed training statistics.
        """
        # Load raw numpy arrays
        x_raw = np.load(diffae_path).astype(np.float32)
        y_raw = np.load(arcface_path).astype(np.float32)
        
        # Load normalization stats
        train_mean = np.load(mean_path).astype(np.float32)
        train_std = np.load(std_path).astype(np.float32)
        
        # Apply Z-score normalization to inputs
        x_norm = (x_raw - train_mean) / train_std
        
        # Convert to PyTorch tensors
        self.x = torch.from_numpy(x_norm)
        self.y = torch.from_numpy(y_raw)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ==========================================
# 2. Architecture Definition
# ==========================================
class LatentMapperMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Expansion block
        self.block1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        # Processing block
        self.block2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        # Projection block
        self.out = nn.Linear(1024, 512)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.out(x)
        
        # THE ANCHOR: Force output onto the L2 unit hypersphere
        return F.normalize(x, p=2, dim=1)


# ==========================================
# 3. Main Execution
# ==========================================
def main():
    # --- Configurations ---
    BATCH_SIZE = 256
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Paths (Adjust as needed) ---
    BASE_DIR = Path("/nas-ctm01/homes/dacordeiro")
    DIFF_DIR = BASE_DIR / "Face-DM/diffae_embeddings"
    ARC_DIR = BASE_DIR / "arcface_embeddings/Face-DM"
    
    # DiffAE inputs
    d_train = DIFF_DIR / "ffhq256_diffae_zsem_train.npy"
    d_val = DIFF_DIR / "ffhq256_diffae_zsem_val.npy"
    d_test = DIFF_DIR / "ffhq256_diffae_zsem_test.npy"
    d_mean = DIFF_DIR / "ffhq256_diffae_zsem_train_mean.npy"
    d_std = DIFF_DIR / "ffhq256_diffae_zsem_train_std.npy"
    
    # ArcFace targets
    a_train = ARC_DIR / "ffhq256_deepface_arcface_retinaface_l2norm_train.npy"
    a_val = ARC_DIR / "ffhq256_deepface_arcface_retinaface_l2norm_val.npy"
    a_test = ARC_DIR / "ffhq256_deepface_arcface_retinaface_l2norm_test.npy"

    print(f"--- Initializing Data Loaders on {DEVICE} ---")
    train_dataset = EmbeddingDataset(d_train, a_train, d_mean, d_std)
    val_dataset = EmbeddingDataset(d_val, a_val, d_mean, d_std)
    test_dataset = EmbeddingDataset(d_test, a_test, d_mean, d_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Setup Model, Loss, and Optimizer ---
    model = LatentMapperMLP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # CosineEmbeddingLoss expects (input1, input2, target_tensor). 
    # Target tensor contains 1s because we want the vectors to be identical (angle = 0).
    criterion = nn.CosineEmbeddingLoss()

    best_val_loss = float('inf')
    best_model_path = "best_mapper_model.pth"

    # ==========================================
    # 4. Training Loop
    # ==========================================
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        # TQDM progress bar for training
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        
        for x_batch, y_batch in train_bar:
            x_batch, y_batch = x_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            preds = model(x_batch)
            
            # Create a target tensor of 1s mapping to each item in the batch
            target_ones = torch.ones(x_batch.size(0)).to(DEVICE)
            loss = criterion(preds, y_batch, target_ones)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)
                preds = model(x_batch)
                
                target_ones = torch.ones(x_batch.size(0)).to(DEVICE)
                loss = criterion(preds, y_batch, target_ones)
                val_loss += loss.item() * x_batch.size(0)
                
        val_loss /= len(val_dataset)
        
        # Print Epoch Summary
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    total_time = (time.time() - start_time) / 60
    print(f"\n--- Training Complete in {total_time:.2f} minutes ---")

    # ==========================================
    # 5. Final Evaluation & Report Generation
    # ==========================================
    print("\n--- Running Final Evaluation on Isolated Test Set ---")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_cosine_sims = []
    test_mses = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        for x_batch, y_batch in test_bar:
            x_batch, y_batch = x_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)
            preds = model(x_batch)
            
            # Calculate metrics
            cos_sim = F.cosine_similarity(preds, y_batch, dim=1)
            mse = F.mse_loss(preds, y_batch, reduction='none').mean(dim=1)
            
            test_cosine_sims.extend(cos_sim.cpu().numpy())
            test_mses.extend(mse.cpu().numpy())
            
    avg_cosine = np.mean(test_cosine_sims)
    avg_mse = np.mean(test_mses)
    min_cosine = np.min(test_cosine_sims)
    
    # Generate the text report
    report_content = (
        "====================================================\n"
        "           LATENT MAPPER EVALUATION REPORT          \n"
        "====================================================\n\n"
        f"Model Architecture: MLP (512 -> 1024 -> 1024 -> 512)\n"
        f"Normalization:      Input Z-Score | Output L2 Normalization\n"
        f"Loss Function:      Cosine Embedding Loss\n"
        f"Test Set Size:      {len(test_dataset)} samples\n\n"
        "--- METRICS ---\n"
        f"Average Cosine Similarity:  {avg_cosine:.4f} (Closer to 1.0 is better)\n"
        f"Minimum Cosine Similarity:  {min_cosine:.4f} (Worst-case translation)\n"
        f"Average Mean Squared Error: {avg_mse:.6f} (Closer to 0.0 is better)\n\n"
        "====================================================\n"
    )
    
    report_path = "evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(report_content)
    print(f"[SUCCESS] Report saved to: {report_path}")

if __name__ == "__main__":
    main()
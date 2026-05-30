import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# ==========================================
# Re-using your exact Data Structures
# ==========================================
class ColdDiffAEDemorphDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, epoch_size: int = 1000000, deterministic: bool = False):
        self.embeddings = embeddings
        self.epoch_size = epoch_size
        self.deterministic = deterministic
        self.num_samples = len(embeddings)
        
        if self.deterministic:
            np.random.seed(42)
            idx1 = np.random.randint(0, self.num_samples, size=epoch_size)
            offsets = np.random.randint(1, self.num_samples, size=epoch_size)
            idx2 = (idx1 + offsets) % self.num_samples
            self.pairs = np.stack([idx1, idx2], axis=1)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if self.deterministic:
            idx1, idx2 = self.pairs[idx]
        else:
            idx1 = np.random.randint(0, self.num_samples)
            offset = np.random.randint(1, self.num_samples)
            idx2 = (idx1 + offset) % self.num_samples
            
        return torch.tensor(self.embeddings[idx1], dtype=torch.float32), torch.tensor(self.embeddings[idx2], dtype=torch.float32)

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
# Imbalance Analysis Core Pipeline
# ==========================================
def analyze_validation_asymmetry(diffae_path_str: str, output_report_name: str = "validation_cosine_imbalance.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the z-score normalized validation split
    val_embs = load_split_and_normalize(diffae_path_str, "val")
    
    # Recreate the exact 10k deterministic pairs used in your validation loops
    val_loader = DataLoader(
        ColdDiffAEDemorphDataset(val_embs, epoch_size=10_000, deterministic=True), 
        batch_size=10_000, 
        shuffle=False
    )
    
    print(f"Analyzing angular asymmetry across 10,000 validation pairs on {device}...")
    
    with torch.no_grad():
        for batch_z1, batch_z2 in val_loader:
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            
            # Compute true average mixture
            batch_c = (batch_z1 + batch_z2) / 2.0
            
            # Compute cosine similarities from average mixture to both components
            sim_c_z1 = F.cosine_similarity(batch_c, batch_z1, dim=-1)
            sim_c_z2 = F.cosine_similarity(batch_c, batch_z2, dim=-1)
            
            # Calculate absolute imbalance
            imbalances = torch.abs(sim_c_z1 - sim_c_z2)
            
            # Extract L2 magnitudes to verify relationship with vector length
            mag_z1 = torch.norm(batch_z1, p=2, dim=-1)
            mag_z2 = torch.norm(batch_z2, p=2, dim=-1)
            
            # Check if the average embedding is closer to the vector with the LARGER magnitude
            closer_to_z1 = sim_c_z1 > sim_c_z2
            larger_is_z1 = mag_z1 > mag_z2
            magnitude_rule_holds = (closer_to_z1 == larger_is_z1)
            rule_percentage = magnitude_rule_holds.float().mean().item() * 100.0

    # Convert tensors to numpy arrays for final reporting statistics
    imbalances_np = imbalances.cpu().numpy()
    sim_c_z1_np = sim_c_z1.cpu().numpy()
    sim_c_z2_np = sim_c_z2.cpu().numpy()
    
    # Format the summary metrics report text
    report_content = (
        f"=========================================================================\n"
        f"      DIFF-AE VALIDATION MANIFOLD: ANGULAR IMBALANCE EVALUATION REPORT\n"
        f"=========================================================================\n\n"
        f"Total Deterministic Pairs Evaluated: {len(imbalances_np):,}\n"
        f"Vector Space Dimensionality:          512 (z_sem)\n\n"
        f"-------------------------------------------------------------------------\n"
        f"1. COSINE SIMILARITY STATISTICS\n"
        f"-------------------------------------------------------------------------\n"
        f"  - Overall Mean CosSim(c, z1):       {np.mean(sim_c_z1_np):.6f}\n"
        f"  - Overall Mean CosSim(c, z2):       {np.mean(sim_c_z2_np):.6f}\n"
        f"  - Absolute Maximum CosSim Observed:  {max(np.max(sim_c_z1_np), np.max(sim_c_z2_np)):.6f}\n"
        f"  - Absolute Minimum CosSim Observed:  {min(np.min(sim_c_z1_np), np.min(sim_c_z2_np)):.6f}\n\n"
        f"-------------------------------------------------------------------------\n"
        f"2. ANGULAR SKEW / IMBALANCE DISTRIBUTION (|CosSim1 - CosSim2|)\n"
        f"-------------------------------------------------------------------------\n"
        f"  - Mean Angular Imbalance:           {np.mean(imbalances_np):.6f}\n"
        f"  - Standard Deviation of Imbalance:   {np.std(imbalances_np):.6f}\n"
        f"  - Maximum Pairwise Imbalance:       {np.max(imbalances_np):.6f}\n"
        f"  - Minimum Pairwise Imbalance:       {np.min(imbalances_np):.6f}\n\n"
        f"-------------------------------------------------------------------------\n"
        f"3. GEOMETRIC HYPOTHESIS VERIFICATION\n"
        f"-------------------------------------------------------------------------\n"
        f"  - Rate at which the average vector 'c' is angularly closer\n"
        f"    to the embedding with the larger L2 norm magnitude: {rule_percentage:.2f}%\n\n"
        f"Conclusion: If the rate above is 100.00%, it mathematically confirms that\n"
        f"relative embedding length completely rules the angular skew towards the midpoint.\n"
        f"=========================================================================\n"
    )

    # Export out to text file
    output_path = Path(diffae_path_str).parent / output_report_name
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print("\n" + report_content)
    print(f"[SUCCESS] Informative validation symmetry report exported safely to:\n  -> {output_path}")

if __name__ == "__main__":
    # Point this to your master file path 
    BASE_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    
    analyze_validation_asymmetry(diffae_path_str=BASE_PATH)
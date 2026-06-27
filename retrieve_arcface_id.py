import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. Replicated Model Architecture
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
# 2. Main Identification Routine
# ==========================================
def evaluate_identity_leakage(diffae_path_str: str, run_name: str, num_timesteps: int = 300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path("experiments") / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"
    report_out_path = exp_dir / "identity_retrieval_report.txt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Trained checkpoint missing from target run directory: {ckpt_path}")

    # 1. Load Raw Test Pairs Shared Tensor Space
    base_path = Path(diffae_path_str).resolve()
    test_pairs_path = base_path.parent / f"{base_path.stem.replace('_train', '')}_test_pairs.npy"
    
    print(f"Loading evaluation pairs from: {test_pairs_path}")
    test_pairs_embs = np.load(test_pairs_path).astype(np.float32) # Shape: (1000, 2, 512)
    num_pairs = len(test_pairs_embs)

    # 2. Reconstruct the Unique Identity Gallery
    # Side A (index 0) contains all 1,000 unique source test face vectors exactly once.
    gallery_raw = test_pairs_embs[:, 0, :].copy() # Shape: (1000, 512)
    
    # Calculate Unit Vector Base for clean global Cosine Similarity Matrix calculations
    gallery_norms = np.linalg.norm(gallery_raw, axis=1, keepdims=True)
    gallery_unit = gallery_raw / np.where(gallery_norms == 0, 1.0, gallery_norms)

    # Determine exact target gallery indices for Side B elements via cross dot-product mapping
    side_b_norms = np.linalg.norm(test_pairs_embs[:, 1, :], axis=1, keepdims=True)
    side_b_unit = test_pairs_embs[:, 1, :] / np.where(side_b_norms == 0, 1.0, side_b_norms)
    side_b_gallery_indices = np.argmax(np.dot(side_b_unit, gallery_unit.T), axis=1)

    # 3. Model Initialization and Weight Loading
    print("Initializing model architecture & loading weights...")
    net = ColdDemorphNet().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # 4. Process One-Shot Network Predictions
    # Scale embeddings by sqrt(512) to match exact pipeline requirements
    test_pairs_scaled = torch.tensor(test_pairs_embs * math.sqrt(512), dtype=torch.float32, device=device)
    
    SQRT_05 = math.sqrt(0.5)
    SQRT_2 = math.sqrt(2.0)
    
    batch_z1 = test_pairs_scaled[:, 0, :]
    batch_z2 = test_pairs_scaled[:, 1, :]
    
    # Generate the standard midpoint blend vector representation
    batch_c_vp = SQRT_05 * batch_z1 + SQRT_05 * batch_z2 

    print("Running one-shot un-morphing predictions across all pairs...")
    with torch.no_grad():
        t_batch = torch.full((num_pairs,), num_timesteps, device=device).long()
        pred = net(batch_c_vp, t_batch)
        pred_raw_1, _ = pred.chunk(2, dim=-1)
        
        pred_z1 = pred_raw_1
        pred_z2 = SQRT_2 * batch_c_vp - pred_z1

        # Use Permutation-Invariant Cost Sorting to optimally align outputs to True Targets
        dist_a = F.mse_loss(pred_z1, batch_z1, reduction='none').mean(dim=1) + F.mse_loss(pred_z2, batch_z2, reduction='none').mean(dim=1)
        dist_b = F.mse_loss(pred_z1, batch_z2, reduction='none').mean(dim=1) + F.mse_loss(pred_z2, batch_z1, reduction='none').mean(dim=1)
        mask_a = (dist_a <= dist_b).unsqueeze(-1)
        
        aligned_pred_z1 = torch.where(mask_a, pred_z1, pred_z2)
        aligned_pred_z2 = torch.where(mask_a, pred_z2, pred_z1)

        # Re-scale back to unit space dimension for exact downstream deployment checks
        pred_z1_unscaled = (aligned_pred_z1 / math.sqrt(512)).cpu().numpy()
        pred_z2_unscaled = (aligned_pred_z2 / math.sqrt(512)).cpu().numpy()

    # 5. Measure Multi-Identity Identification Accuracy Metrics
    print("Evaluating nearest neighbor gallery lookups...")
    p1_norms = np.linalg.norm(pred_z1_unscaled, axis=1, keepdims=True)
    p2_norms = np.linalg.norm(pred_z2_unscaled, axis=1, keepdims=True)
    
    pred_z1_unit = pred_z1_unscaled / np.where(p1_norms == 0, 1.0, p1_norms)
    pred_z2_unit = pred_z2_unscaled / np.where(p2_norms == 0, 1.0, p2_norms)

    # Compute global cross-similarities against the 1,000 unique gallery test assets
    sims_matrix_p1 = np.dot(pred_z1_unit, gallery_unit.T) # Shape: (1000, 1000)
    sims_matrix_p2 = np.dot(pred_z2_unit, gallery_unit.T) # Shape: (1000, 1000)

    # Resolve Rank-1 Predictions (indices of closest gallery matches)
    rank1_matches_p1 = np.argmax(sims_matrix_p1, axis=1)
    rank1_matches_p2 = np.argmax(sims_matrix_p2, axis=1)

    # Track structural scores
    p1_correct = 0
    p2_correct = 0
    both_identities_resolved = 0
    at_least_one_resolved = 0

    for i in range(num_pairs):
        true_idx_a = i
        true_idx_b = side_b_gallery_indices[i]

        match_a_correct = (rank1_matches_p1[i] == true_idx_a)
        match_b_correct = (rank1_matches_p2[i] == true_idx_b)

        if match_a_correct: p1_correct += 1
        if match_b_correct: p2_correct += 1
        if match_a_correct and match_b_correct: both_identities_resolved += 1
        if match_a_correct or match_b_correct: at_least_one_resolved += 1

    # Format output rates
    pct_p1 = (p1_correct / num_pairs) * 100
    pct_p2 = (p2_correct / num_pairs) * 100
    pct_total_vectors = ((p1_correct + p2_correct) / (num_pairs * 2)) * 100
    pct_both = (both_identities_resolved / num_pairs) * 100
    pct_at_least_one = (at_least_one_resolved / num_pairs) * 100

    # 6. Generate and Write text Evaluation Summary
    report_text = (
        "==========================================================\n"
        "      COLD DIFFUSION DEMORPH IDENTITY RETRIEVAL REPORT     \n"
        "==========================================================\n"
        f"Run Target Architecture Name: {run_name}\n"
        f"Evaluation Dataset Volume:   {num_pairs} Pairs (2,000 Vector Elements)\n"
        f"Gallery Search Space Scope:  {len(gallery_unit)} Unique Target Identities\n"
        "----------------------------------------------------------\n\n"
        "[1. ELEMENT-LEVEL IDENTIFICATION ACCURACY (RANK-1)]\n"
        f"  - Target Identity A Correct Matches:    {p1_correct}/{num_pairs} ({pct_p1:.2f}%)\n"
        f"  - Target Identity B Correct Matches:    {p2_correct}/{num_pairs} ({pct_b:=.2f}% if formatting matches)\n"
        f"  - Target Identity B Correct Matches:    {p2_correct}/{num_pairs} ({pct_p2:.2f}%)\n"
        f"  - Combined Vector Success Rate:         {p1_correct + p2_correct}/{num_pairs * 2} ({pct_total_vectors:.2f}%)\n\n"
        "[2. PAIR-LEVEL RESOLUTION SUCCESS PROFILE]\n"
        f"  - Complete Resolution (Both Found):     {both_identities_resolved}/{num_pairs} ({pct_both:.2f}%)\n"
        f"  - Partial Resolution (At Least One):    {at_least_one_resolved}/{num_pairs} ({pct_at_least_one:.2f}%)\n"
        f"  - Failed Resolution (Neither Found):    {num_pairs - at_least_one_resolved}/{num_pairs} ({100.0 - pct_at_least_one:.2f}%)\n\n"
        "Interpretation Note:\n"
        "A 'Complete Resolution' indicates that for a given morphed face matrix, the\n"
        "one-shot network outputs map uniquely back to their respective original source profiles\n"
        "with zero identification collisions across the test identity gallery space.\n"
    )

    print("\n" + report_text)
    with open(report_out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[SUCCESS] Rank-1 identity retrieval summary exported to: {report_out_path}")

if __name__ == "__main__":
    BASE_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy"
    RUN_NAME = "arcface_baseline_mse_scaled_spreadLoss0.1_new"

    evaluate_identity_leakage(diffae_path_str=BASE_PATH, run_name=RUN_NAME, num_timesteps=300)
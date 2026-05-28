import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. Dataset & Data Loading
# ==========================================
class CombinedEvaluationDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, epoch_size: int = 10000, deterministic: bool = True):
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
# 2. Network Architectures
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
        h = self.norm(h) * (1 + scale) + shift
        return self.silu(h)


class ColdDemorphNet(nn.Module):
    def __init__(self, x_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        cond_dim = time_emb_dim
        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(x_dim, hidden_dim, cond_dim))
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + x_dim, hidden_dim, cond_dim))
        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, t):
        cond = self.time_mlp(t)
        h = x
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, cond)
            else:
                h = block(torch.cat([h, x], dim=-1), cond)
        return self.final_linear(h)


class DeterministicColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=4, terminal_alpha=0.60):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.terminal_alpha = terminal_alpha

    def degrade(self, z1, z2, t):
        alpha = 1.0 - (1.0 - self.terminal_alpha) * (t / self.num_timesteps).view(-1, 1).float()
        z1_u = F.normalize(z1, p=2, dim=-1)
        z2_u = F.normalize(z2, p=2, dim=-1)
        
        dot = torch.sum(z1_u * z2_u, dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)
        omega = torch.acos(dot)
        sin_omega = torch.sin(omega)
        
        eps = 1e-5
        linear_mask = (sin_omega < eps).float()
        
        factor1 = torch.sin(alpha * omega) / (sin_omega + 1e-8)
        factor2 = torch.sin((1.0 - alpha) * omega) / (sin_omega + 1e-8)
        slerp_mix = factor1 * z1_u + factor2 * z2_u
        
        lerp_mix = alpha * z1_u + (1.0 - alpha) * z2_u
        lerp_mix = F.normalize(lerp_mix, p=2, dim=-1)
        
        mixed = (1.0 - linear_mask) * slerp_mix + linear_mask * lerp_mix
        return mixed * math.sqrt(512)


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


class PermutationInvariantDemorpher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def predict(self, c):
        pred_z12 = self.model(c)
        return pred_z12.chunk(2, dim=-1)


# ==========================================
# 3. Hybrid TACOs Geometry Sampling Loop
# ==========================================
def run_hybrid_tacos_sampling(diffusion, model, x_t, batch_c):
    device = batch_c.device
    b = batch_c.shape[0]
    timesteps = torch.arange(diffusion.num_timesteps, 0, -1, device=device).long()
    
    for t in timesteps:
        t_batch = torch.full((b,), t, device=device, dtype=torch.long)
        
        # Predict primary z1 target from the biased blend
        pred_z1 = model(x_t, t_batch)
        pred_z1_norm = F.normalize(pred_z1, p=2, dim=-1) * math.sqrt(512)
        
        # Geometrically extract pred_z2 using the altered average constraint
        dot_c_z1 = torch.sum(batch_c * pred_z1_norm, dim=-1, keepdim=True)
        k = 2.0 * dot_c_z1 / 512.0
        pred_z2_norm = k * batch_c - pred_z1_norm
        pred_z2_norm = F.normalize(pred_z2_norm, p=2, dim=-1) * math.sqrt(512)
        
        t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
        
        # Continuous sampling updates
        deg_t = diffusion.degrade(pred_z1_norm, pred_z2_norm, t_batch)
        deg_t_prev = diffusion.degrade(pred_z1_norm, pred_z2_norm, t_prev_batch)
        
        x_t = x_t - deg_t + deg_t_prev
        
    final_z1 = F.normalize(x_t, p=2, dim=-1) * math.sqrt(512)
    dot_c_z1 = torch.sum(batch_c * final_z1, dim=-1, keepdim=True)
    k = 2.0 * dot_c_z1 / 512.0
    final_z2 = k * batch_c - final_z1
    final_z2 = F.normalize(final_z2, p=2, dim=-1) * math.sqrt(512)
    
    return final_z1, final_z2


# ==========================================
# 4. Batched Alignment Metric Calculator
# ==========================================
def compute_aligned_metrics(p1, p2, t1, t2, scale_factor):
    """
    Computes permutation-invariant individual slot metrics dynamically per sample.
    """
    dist_opt1 = F.l1_loss(p1, t1, reduction='none').mean(dim=-1) + F.l1_loss(p2, t2, reduction='none').mean(dim=-1)
    dist_opt2 = F.l1_loss(p1, t2, reduction='none').mean(dim=-1) + F.l1_loss(p2, t1, reduction='none').mean(dim=-1)
    
    mask = (dist_opt1 < dist_opt2).float().unsqueeze(-1)
    
    a_p1 = mask * p1 + (1.0 - mask) * p2
    a_p2 = mask * p2 + (1.0 - mask) * p1
    
    # L1 Distance (Scaled space)
    l1_scaled_1 = F.l1_loss(a_p1, t1, reduction='none').mean(dim=-1)
    l1_scaled_2 = F.l1_loss(a_p2, t2, reduction='none').mean(dim=-1)
    
    # L1 Distance & Cosine Similarity (Raw ArcFace Space)
    a_p1_raw = F.normalize(a_p1 / scale_factor, p=2, dim=-1)
    a_p2_raw = F.normalize(a_p2 / scale_factor, p=2, dim=-1)
    t1_raw = F.normalize(t1 / scale_factor, p=2, dim=-1)
    t2_raw = F.normalize(t2 / scale_factor, p=2, dim=-1)
    
    l1_raw_1 = F.l1_loss(a_p1_raw, t1_raw, reduction='none').mean(dim=-1)
    l1_raw_2 = F.l1_loss(a_p2_raw, t2_raw, reduction='none').mean(dim=-1)
    
    cos_1 = F.cosine_similarity(a_p1_raw, t1_raw, dim=-1)
    cos_2 = F.cosine_similarity(a_p2_raw, t2_raw, dim=-1)
    
    return {
        'l1_scaled_1': l1_scaled_1.sum().item(),
        'l1_scaled_2': l1_scaled_2.sum().item(),
        'l1_raw_1': l1_raw_1.sum().item(),
        'l1_raw_2': l1_raw_2.sum().item(),
        'cos_1': cos_1.sum().item(),
        'cos_2': cos_2.sum().item()
    }


# ==========================================
# 5. Combined Evaluation Pipeline
# ==========================================
def evaluate_combined_system(arcface_path_str: str, oneshot_run: str, iterative_run: str, num_timesteps: int = 4, terminal_alpha: float = 0.60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale_factor = math.sqrt(512)
    
    # Load and process validation embeddings
    print("Loading data...")
    arcface_embs = (np.load(Path(arcface_path_str).resolve()).astype(np.float32)) * scale_factor
    split_idx = int(len(arcface_embs) * 0.9)
    val_embs = arcface_embs[split_idx:]
    val_dataset = CombinedEvaluationDataset(val_embs, epoch_size=10000, deterministic=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=4)
    
    # Initialize networks
    oneshot_net = OneShotDemorphNet(in_dim=512, hidden_dim=2048, out_dim=1024, num_blocks=6).to(device)
    oneshot_wrapper = PermutationInvariantDemorpher(oneshot_net).to(device)
    
    iterative_net = ColdDemorphNet(x_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(iterative_net, num_timesteps=num_timesteps, terminal_alpha=terminal_alpha).to(device)
    
    # Load weights
    oneshot_ckpt = Path("experiments") / oneshot_run / "checkpoints" / "best.pt"
    iterative_ckpt = Path("experiments") / iterative_run / "checkpoints" / "best.pt"
    
    print(f"Loading One-Shot weights from: {oneshot_ckpt}")
    oneshot_net.load_state_dict(torch.load(oneshot_ckpt, map_location=device)['model_state_dict'])
    print(f"Loading Iterative weights from: {iterative_ckpt}")
    iterative_net.load_state_dict(torch.load(iterative_ckpt, map_location=device)['model_state_dict'])
    
    oneshot_net.eval()
    iterative_net.eval()
    
    # Metric Storage
    methods = ['oneshot_baseline', 'hybrid_path1', 'hybrid_path2', 'hybrid_combined_primary']
    stats = {m: {k: 0.0 for k in ['l1_scaled_1', 'l1_scaled_2', 'l1_raw_1', 'l1_raw_2', 'cos_1', 'cos_2']} for m in methods}
    total_samples = 0
    
    with torch.no_grad():
        for batch_z1, batch_z2 in tqdm(val_loader, desc="Running Combined Evaluation"):
            batch_z1, batch_z2 = batch_z1.to(device), batch_z2.to(device)
            b = batch_z1.shape[0]
            total_samples += b
            
            # Form the altered hyperspherical average embedding input
            batch_c = F.normalize(batch_z1 + batch_z2, p=2, dim=-1) * scale_factor
            
            # Step 1: Extract base estimates using the One-Shot system
            pred_os_z1, pred_os_z2 = oneshot_wrapper.predict(batch_c)
            pred_os_z1 = F.normalize(pred_os_z1, p=2, dim=-1) * scale_factor
            pred_os_z2 = F.normalize(pred_os_z2, p=2, dim=-1) * scale_factor
            
            # Step 2: Compute TACOs Error Correction inputs for the final timestep T
            pred_c_shared = F.normalize(pred_os_z1 + pred_os_z2, p=2, dim=-1) * scale_factor
            t_T = torch.full((b,), num_timesteps, device=device, dtype=torch.long)
            
            deg_T_path1 = diffusion.degrade(pred_os_z1, pred_os_z2, t_T)
            x_T_path1 = batch_c - pred_c_shared + deg_T_path1
            
            deg_T_path2 = diffusion.degrade(pred_os_z2, pred_os_z1, t_T)
            x_T_path2 = batch_c - pred_c_shared + deg_T_path2
            
            # Step 3: Run sampling loops along the independent paths
            p1_z1, p1_z2 = run_hybrid_tacos_sampling(diffusion, iterative_net, x_T_path1, batch_c)
            p2_z1, p2_z2 = run_hybrid_tacos_sampling(diffusion, iterative_net, x_T_path2, batch_c)
            
            # Step 4: Gather aligned evaluation statistics
            m_os = compute_aligned_metrics(pred_os_z1, pred_os_z2, batch_z1, batch_z2, scale_factor)
            m_p1 = compute_aligned_metrics(p1_z1, p1_z2, batch_z1, batch_z2, scale_factor)
            m_p2 = compute_aligned_metrics(p2_z2, p2_z1, batch_z1, batch_z2, scale_factor)
            m_comb = compute_aligned_metrics(p1_z1, p2_z1, batch_z1, batch_z2, scale_factor)
            
            for k in m_os.keys():
                stats['oneshot_baseline'][k] += m_os[k]
                stats['hybrid_path1'][k] += m_p1[k]
                stats['hybrid_path2'][k] += m_p2[k]
                stats['hybrid_combined_primary'][k] += m_comb[k]

    # Print results summary table
    print("\n" + "="*85)
    print(f"{'EVALUATION METRICS SUMMARY (REPORTED PER EMBEDDING SLOT)':^85}")
    print("="*85)
    
    for m in methods:
        print(f"\n▶ Method: {m.upper().replace('_', ' ')}")
        print(f"  {'Metric Domain':<35} | {'Embedding 1':<18} | {'Embedding 2':<18}")
        print(f"  {'-'*35}-+-{'-'*18}-+-{'-'*18}")
        print(f"  L1 Distance (Scaled space)         | {stats[m]['l1_scaled_1']/total_samples:<18.6f} | {stats[m]['l1_scaled_2']/total_samples:<18.6f}")
        print(f"  L1 Distance (Raw ArcFace)          | {stats[m]['l1_raw_1']/total_samples:<18.6f} | {stats[m]['l1_raw_2']/total_samples:<18.6f}")
        print(f"  Cosine Similarity (Raw ArcFace)   | {stats[m]['cos_1']/total_samples:<18.6f} | {stats[m]['cos_2']/total_samples:<18.6f}")
    print("="*85 + "\n")


if __name__ == "__main__":
    DATA_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy"
    ONESHOT_RUN = "oneshot_arcface_real"
    ITERATIVE_RUN = "asymmetric_z1_alpha_55_real"

    evaluate_combined_system(
        arcface_path_str=DATA_PATH,
        oneshot_run=ONESHOT_RUN,
        iterative_run=ITERATIVE_RUN,
        num_timesteps=4,
        terminal_alpha=0.60
    )

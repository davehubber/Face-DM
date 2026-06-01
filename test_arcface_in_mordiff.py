import re
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from deepface import DeepFace

# ==========================================
# 1. Configuration & Paths
# ==========================================
MORDIFF_DIR = Path("MorDIFF_crop")
FRLL_DIR = Path("FRLL")
RUN_NAME = "angular_ortho_loss_run"
REPORT_PATH = Path(f"experiments/{RUN_NAME}/mordiff_realworld_eval.txt")

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
ALIGN = True
NORMALIZATION = "ArcFace"

# ==========================================
# 2. Model Architecture (Must match training)
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
    def __init__(self, model, num_timesteps=10):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, z1, z2, t):
        scale_factor = math.sqrt(512)
        zm = F.normalize(z1 + z2, p=2, dim=-1) * scale_factor
        alpha = 1.0 - (t / self.num_timesteps).view(-1, 1).float()
        return alpha * z1 + (1.0 - alpha) * zm

    @torch.no_grad()
    def tacos_sample_loop(self, c, disable_tqdm=True):
        device = c.device
        b = c.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = c.clone()
        
        for t in tqdm(timesteps, desc='TACOs', leave=False, disable=disable_tqdm):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            pred_z = self.model(x_t, t_batch)
            kappa = 2.0 * torch.sum(c * pred_z, dim=-1, keepdim=True) / 512.0
            comp_z = kappa * c - pred_z
            
            opt1_z1, opt1_z2 = pred_z, comp_z
            opt2_z1, opt2_z2 = comp_z, pred_z
            
            deg_t_opt1 = self.degrade(opt1_z1, opt1_z2, t_batch)
            deg_t_opt2 = self.degrade(opt2_z1, opt2_z2, t_batch)
            
            cos_opt1 = F.cosine_similarity(deg_t_opt1, x_t, dim=-1)
            cos_opt2 = F.cosine_similarity(deg_t_opt2, x_t, dim=-1)
            
            is_opt1 = (cos_opt1 >= cos_opt2).unsqueeze(-1)
            
            z1_est = torch.where(is_opt1, opt1_z1, opt2_z1)
            z2_est = torch.where(is_opt1, opt1_z2, opt2_z2)
            
            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            deg_t = self.degrade(z1_est, z2_est, t_batch)
            deg_t_prev = self.degrade(z1_est, z2_est, t_prev_batch)
            
            x_t = x_t - deg_t + deg_t_prev
            
        final_z1 = x_t
        kappa_final = 2.0 * torch.sum(c * final_z1, dim=-1, keepdim=True) / 512.0
        final_z2 = kappa_final * c - final_z1
        return final_z1, final_z2

# ==========================================
# 3. Helper Functions
# ==========================================
def choose_main_face(face_objs):
    if len(face_objs) == 1:
        return face_objs[0]
    def area(obj):
        fa = obj.get("facial_area", {})
        return float(fa.get("w", 0) * fa.get("h", 0))
    return max(face_objs, key=area)

def get_l2_normalized_embedding(img_path: Path) -> np.ndarray:
    face_objs = DeepFace.represent(
        img_path=str(img_path),
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        align=ALIGN,
        enforce_detection=True,
        normalization=NORMALIZATION,
    )
    obj = choose_main_face(face_objs)
    emb = np.asarray(obj["embedding"], dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Invalid embedding norm")
    return emb / norm

# ==========================================
# 4. Main Evaluation Pipeline
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    ckpt_path = Path("experiments") / RUN_NAME / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")
        
    net = ColdDemorphNet(x_dim=512, hidden_dim=2048, num_layers=10).to(device)
    diffusion = DeterministicColdDemorph(net, num_timesteps=5).to(device)
    
    print(f"Loading weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # Index Files
    print("Indexing FRLL base images...")
    base_images = {img.stem: img for img in FRLL_DIR.rglob("*.jpg")}
    
    print("Locating morph images...")
    morph_paths = list(MORDIFF_DIR.rglob("morphed/*.png"))
    
    base_emb_cache = {}
    filename_pattern = re.compile(r"morph_(.+)_and_(.+)")
    scale_factor = math.sqrt(512)

    metrics = {
        "l1_scaled_z1": 0.0, "l1_scaled_z2": 0.0,
        "l1_raw_z1": 0.0, "l1_raw_z2": 0.0,
        "cos_z1": 0.0, "cos_z2": 0.0,
        "cos_real_z1_z2": 0.0,
        "cos_inter_pred": 0.0,
        "original_morph_cos_midpoint": 0.0 # Track how linear the original morph was
    }
    
    successful_evals = 0
    errors = []

    with torch.no_grad():
        for morph_path in tqdm(morph_paths, desc="Evaluating MorDIFF Dataset"):
            match = filename_pattern.search(morph_path.stem)
            if not match:
                continue
            
            id1, id2 = match.groups()
            base_path1, base_path2 = base_images.get(id1), base_images.get(id2)

            if not base_path1 or not base_path2:
                continue

            try:
                # 1. Get/Compute Base Embeddings
                if id1 not in base_emb_cache:
                    base_emb_cache[id1] = get_l2_normalized_embedding(base_path1)
                emb1 = base_emb_cache[id1]

                if id2 not in base_emb_cache:
                    base_emb_cache[id2] = get_l2_normalized_embedding(base_path2)
                emb2 = base_emb_cache[id2]

                # 2. Get Morph Embedding & Scale it properly for the model!
                emb_morph_raw = get_l2_normalized_embedding(morph_path)
                
                # Convert to Tensors and add Batch Dimension
                z1_true_raw = torch.tensor(emb1, dtype=torch.float32, device=device).unsqueeze(0)
                z2_true_raw = torch.tensor(emb2, dtype=torch.float32, device=device).unsqueeze(0)
                
                # IMPORTANT: Scale the morph embedding to act as 'batch_c'
                m_raw_tensor = torch.tensor(emb_morph_raw, dtype=torch.float32, device=device).unsqueeze(0)
                batch_c = m_raw_tensor * scale_factor
                
                # Calculate original linearity (for reference)
                midpoint = F.normalize(z1_true_raw + z2_true_raw, p=2, dim=-1)
                morph_linearity = F.cosine_similarity(m_raw_tensor, midpoint, dim=-1).item()

                # 3. Model Inference (TACOs Sampling)
                pred_z1_scaled, pred_z2_scaled = diffusion.tacos_sample_loop(batch_c, disable_tqdm=True)

                # 4. Raw Space Conversion
                p1_raw = F.normalize(pred_z1_scaled, p=2, dim=-1)
                p2_raw = F.normalize(pred_z2_scaled, p=2, dim=-1)

                # 5. Dynamic Permutation Alignment
                cos_p1_z1 = F.cosine_similarity(p1_raw, z1_true_raw, dim=-1)
                cos_p1_z2 = F.cosine_similarity(p1_raw, z2_true_raw, dim=-1)
                
                is_p1_to_z1 = (cos_p1_z1 >= cos_p1_z2).unsqueeze(-1)
                
                z1_pred_scaled = torch.where(is_p1_to_z1, pred_z1_scaled, pred_z2_scaled)
                z2_pred_scaled = torch.where(is_p1_to_z1, pred_z2_scaled, pred_z1_scaled)
                
                z1_pred_raw = torch.where(is_p1_to_z1, p1_raw, p2_raw)
                z2_pred_raw = torch.where(is_p1_to_z1, p2_raw, p1_raw)

                # 6. Compute Real-world Metrics
                metrics["l1_scaled_z1"] += F.l1_loss(z1_pred_scaled, z1_true_raw * scale_factor, reduction='none').mean().item()
                metrics["l1_scaled_z2"] += F.l1_loss(z2_pred_scaled, z2_true_raw * scale_factor, reduction='none').mean().item()
                
                metrics["l1_raw_z1"] += F.l1_loss(z1_pred_raw, z1_true_raw, reduction='none').mean().item()
                metrics["l1_raw_z2"] += F.l1_loss(z2_pred_raw, z2_true_raw, reduction='none').mean().item()
                
                metrics["cos_z1"] += F.cosine_similarity(z1_pred_raw, z1_true_raw, dim=-1).mean().item()
                metrics["cos_z2"] += F.cosine_similarity(z2_pred_raw, z2_true_raw, dim=-1).mean().item()
                
                metrics["cos_real_z1_z2"] += F.cosine_similarity(z1_true_raw, z2_true_raw, dim=-1).mean().item()
                metrics["cos_inter_pred"] += F.cosine_similarity(p1_raw, p2_raw, dim=-1).mean().item()
                metrics["original_morph_cos_midpoint"] += morph_linearity

                successful_evals += 1

            except Exception as e:
                errors.append(f"{morph_path.name}: {repr(e)}")

    # ==========================================
    # 5. Generate Report
    # ==========================================
    if successful_evals == 0:
        print("No successful evaluations.")
        return

    for k in metrics:
        metrics[k] /= successful_evals

    results_text = (
        f"==================================================\n"
        f"REAL-WORLD MORDIFF DE-MORPHING EVALUATION REPORT\n"
        f"==================================================\n"
        f"Run Name:         {RUN_NAME}\n"
        f"Total Evaluated:  {successful_evals}\n"
        f"Target Space:     ArcFace 512-D\n"
        f"Model Timesteps:  {diffusion.num_timesteps}\n"
        f"--------------------------------------------------\n\n"
        f"0. BASELINE ALIGNMENT\n"
        f"--------------------------------------------------\n"
        f"  - Original Morph Linearity (Cos to NLERP): {metrics['original_morph_cos_midpoint']:.6f}\n"
        f"    (The model had to correct for this structural offset before separating)\n\n"
        f"1. GROUND TRUTH Z1 RECONSTRUCTION\n"
        f"--------------------------------------------------\n"
        f"  - L1 Distance (Scaled space): {metrics['l1_scaled_z1']:.6f}\n"
        f"  - L1 Distance (Raw ArcFace):  {metrics['l1_raw_z1']:.6f}\n"
        f"  - Cosine Similarity vs Z1:    {metrics['cos_z1']:.6f}\n\n"
        f"2. GROUND TRUTH Z2 RECONSTRUCTION\n"
        f"--------------------------------------------------\n"
        f"  - L1 Distance (Scaled space): {metrics['l1_scaled_z2']:.6f}\n"
        f"  - L1 Distance (Raw ArcFace):  {metrics['l1_raw_z2']:.6f}\n"
        f"  - Cosine Similarity vs Z2:    {metrics['cos_z2']:.6f}\n\n"
        f"3. GEOMETRIC ALIGNMENT & ORTHOGONALITY COMPARISON\n"
        f"--------------------------------------------------\n"
        f"  - Baseline Cosine Sim (True Z1 vs True Z2):  {metrics['cos_real_z1_z2']:.6f}\n"
        f"  - Generated Cosine Sim (Pred_Z1 vs Pred_Z2): {metrics['cos_inter_pred']:.6f}\n\n"
    )

    if errors:
        results_text += f"Errors encountered: {len(errors)}\n"

    print("\n" + results_text)
    
    with open(REPORT_PATH, "w") as f:
        f.write(results_text)
        
    print(f"Saved evaluation results to {REPORT_PATH}")

if __name__ == "__main__":
    main()
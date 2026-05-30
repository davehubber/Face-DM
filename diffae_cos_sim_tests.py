import sys
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
import lpips

# Setup paths to your repositories
PATH_TO_DIFF_MODEL = "../diffae"
sys.path.append(PATH_TO_DIFF_MODEL)

from templates import ffhq256_autoenc
from experiment import LitModel  # Adjust import based on your template layout

def run_perceptual_correlation_analysis(base_path_str: str, out_dir_str: str = "experiments_analysis"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing pipeline on device: {device}")
    
    base_path = Path(base_path_str).resolve()
    parent = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_val", "").replace("_test", "")
    
    master_npy_path = parent / f"{stem}.npy"
    metadata_csv_path = parent / f"{stem}_metadata.csv"
    mean_path = parent / f"{stem}_train_mean.npy"
    std_path = parent / f"{stem}_train_std.npy"
    
    out_dir = Path(out_dir_str).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data and apply z-score parameters
    print("Loading and normalizing master embeddings...")
    embeddings = np.load(master_npy_path).astype(np.float32)
    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    normalized_embs = (embeddings - mean) / std
    
    num_samples = len(normalized_embs)
    torch_embs = torch.from_numpy(normalized_embs).to(device)

    # 2. Parse image paths file mapping
    print("Reading file system path mappings from metadata log...")
    idx_to_path = {}
    with open(metadata_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_path[int(row['embedding_index'])] = row['image_path']

    # 3. Find 10 pairs matching thresholds
    print("Searching for candidate pairs matching semantic geometry limits...")
    np.random.seed(42)
    high_pairs, low_pairs = [], []
    chunk_size = 50_000

    while len(high_pairs) < 10 or len(low_pairs) < 10:
        idx1 = np.random.randint(0, num_samples, size=chunk_size)
        offsets = np.random.randint(1, num_samples, size=chunk_size)
        idx2 = (idx1 + offsets) % num_samples
        
        t_idx1 = torch.from_numpy(idx1).to(device)
        t_idx2 = torch.from_numpy(idx2).to(device)
        
        sims = F.cosine_similarity(torch_embs[t_idx1], torch_embs[t_idx2], dim=-1).cpu().numpy()
        
        for k in range(chunk_size):
            sim = sims[k]
            i1, i2 = idx1[k], idx2[k]
            
            mag1 = np.linalg.norm(normalized_embs[i1])
            mag2 = np.linalg.norm(normalized_embs[i2])
            mag_diff = abs(mag1 - mag2)
            
            pair_data = {'idx1': i1, 'idx2': i2, 'sim': sim, 'mag_diff': mag_diff}
            
            if sim >= 0.8 and len(high_pairs) < 10:
                high_pairs.append(pair_data)
            elif sim <= -0.35 and len(low_pairs) < 10:
                low_pairs.append(pair_data)
                
            if len(high_pairs) == 10 and len(low_pairs) == 10:
                break

    # 4. Initialize Diff-AE and LPIPS networks
    print("Loading pretrained Diffusion Autoencoder checkpoints...")
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load(f'{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)
    
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)

    # 5. Image Processing Pipeline
    transform = transforms.Compose([
        transforms.Resize(conf.img_size),
        transforms.CenterCrop(conf.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def load_and_transform(path):
        return transform(Image.open(path).convert('RGB')).to(device)

    def evaluate_perceptual_manifold(pairs_list, title):
        print(f"Running reconstructions and LPIPS calculations for {title}...")
        scores = []
        visual_tensors = []
        
        for p in pairs_list:
            img1 = load_and_transform(idx_to_path[p['idx1']])
            img2 = load_and_transform(idx_to_path[p['idx2']])
            batch = torch.stack([img1, img2])
            
            with torch.no_grad():
                cond = model.encode(batch)
                xT = model.encode_stochastic(batch, cond, T=250)
                pred = model.render(xT, cond, T=20)

                pred_for_save = pred.clamp(0, 1)

                pred_for_lpips = pred_for_save * 2 - 1

                lpips_val = loss_fn_lpips(pred_for_lpips[0:1], pred_for_lpips[1:2]).item()

            p['lpips'] = lpips_val
            scores.append(lpips_val)

            visual_tensors.extend([pred_for_save[0].cpu(), pred_for_save[1].cpu()])
            
        return scores, visual_tensors

    # Compute targets and save visual collections
    high_lpips, high_imgs = evaluate_perceptual_manifold(high_pairs, "High CosSim")
    low_lpips, low_imgs = evaluate_perceptual_manifold(low_pairs, "Low CosSim")
    
    print("Saving side-by-side reconstruction layout grids...")
    save_image(make_grid(high_imgs, nrow=2), out_dir / "high_cosine_pairs_grid.png")

    save_image(make_grid(low_imgs, nrow=2), out_dir / "low_cosine_pairs_grid.png")

    # 6. Analyze internal magnitude dynamics within High-CosSim cluster
    closest_mag_pair = min(high_pairs, key=lambda x: x['mag_diff'])
    distant_mag_pair = max(high_pairs, key=lambda x: x['mag_diff'])

    # 7. Write Summary Log Report
    report_path = out_dir / "perceptual_correlation_report.txt"
    print(f"Writing statistical summary text logs -> {report_path}")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=========================================================================\n")
        f.write("        DIFF-AE SEMANTIC GEOMETRY VS VISUAL PERCEPTUAL DISTANCE\n")
        f.write("=========================================================================\n\n")
        
        f.write("--- CONDITIONAL ANALYSIS: HIGH COSINE SIMILARITY PAIRS (>= 0.8) ---\n")
        for idx, p in enumerate(high_pairs):
            f.write(f"Pair {idx+1:02d} | Indices: ({p['idx1']:.0f}, {p['idx2']:.0f}) | CosSim: {p['sim']:.4f} | MagDiff: {p['mag_diff']:.4f} | LPIPS: {p['lpips']:.6f}\n")
        f.write(f"==> Cumulative Group Mean LPIPS: {np.mean(high_lpips):.6f}\n\n")
        
        f.write("--- CONDITIONAL ANALYSIS: LOW COSINE SIMILARITY PAIRS (<= -0.35) ---\n")
        for idx, p in enumerate(low_pairs):
            f.write(f"Pair {idx+1:02d} | Indices: ({p['idx1']:.0f}, {p['idx2']:.0f}) | CosSim: {p['sim']:.4f} | MagDiff: {p['mag_diff']:.4f} | LPIPS: {p['lpips']:.6f}\n")
        f.write(f"==> Cumulative Group Mean LPIPS: {np.mean(low_lpips):.6f}\n\n")
        
        f.write("=========================================================================\n")
        f.write("        VECTOR LENGTH VARIATION SUB-TEST (WITHIN HIGH-COSSIM CLUSTER)\n")
        f.write("=========================================================================\n")
        f.write(f"Closest Norm Proximity Pair:   Indices ({closest_mag_pair['idx1']}, {closest_mag_pair['idx2']}) | MagDiff: {closest_mag_pair['mag_diff']:.4f} | LPIPS: {closest_mag_pair['lpips']:.6f}\n")
        f.write(f"Most Distant Norm Proximity Pair: Indices ({distant_mag_pair['idx1']}, {distant_mag_pair['idx2']}) | MagDiff: {distant_mag_pair['mag_diff']:.4f} | LPIPS: {distant_mag_pair['lpips']:.6f}\n")

    print("\nProcessing complete. Review artifacts generated in the output directory.")

if __name__ == "__main__":
    DATA_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    run_perceptual_correlation_analysis(base_path_str=DATA_PATH)

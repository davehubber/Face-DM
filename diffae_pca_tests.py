import sys
import csv
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from sklearn.decomposition import PCA

# Setup paths to your repositories
PATH_TO_DIFF_MODEL = "../diffae"
sys.path.append(PATH_TO_DIFF_MODEL)

from templates import ffhq256_autoenc
from experiment import LitModel

def run_pca_attribute_analysis(base_path_str: str, out_dir_str: str = "pca_analysis_results"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing PCA analysis pipeline on: {device}")
    
    base_path = Path(base_path_str).resolve()
    parent = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_val", "").replace("_test", "")
    
    master_npy_path = parent / f"{stem}.npy"
    metadata_csv_path = parent / f"{stem}_metadata.csv"
    
    out_dir = Path(out_dir_str).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Isolate the Raw Training Split (No Z-score normalization)
    print("Loading raw master embeddings...")
    raw_master_embeddings = np.load(master_npy_path).astype(np.float32)
    total_samples = len(raw_master_embeddings)
    
    np.random.seed(42)
    split_indices = np.arange(total_samples)
    np.random.shuffle(split_indices)
    train_end = int(total_samples * 0.8)
    train_indices = split_indices[:train_end]
    
    train_embeddings = raw_master_embeddings[train_indices]
    print(f"Isolated training split size: {len(train_embeddings)} embeddings")

    # 2. Perform PCA using Scikit-Learn
    print("Fitting PCA model to raw training manifold...")
    pca = PCA(n_components=3)
    pca.fit(train_embeddings)
    
    # Transform to get the projection scores
    train_scores = pca.transform(train_embeddings)
    
    # 3. Adequately store top components and mean for downstream scoring
    components_path = out_dir / "pca_top3_components.npy"
    mean_path = out_dir / "pca_mean.npy"
    np.save(components_path, pca.components_)
    np.save(mean_path, pca.mean_)
    print(f"[SUCCESS] Saved projection references to:\n  -> {components_path}\n  -> {mean_path}")

    # 4. Generate Text Summary Report
    report_path = out_dir / "pca_statistical_report.txt"
    var_ratios = pca.explained_variance_ratio_
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=========================================================================\n")
        f.write("             EMBEDDING MANIFOLD PRINCIPAL COMPONENT REPORT\n")
        f.write("=========================================================================\n\n")
        f.write(f"Total Source Vectors Analyzed: {len(train_embeddings):,}\n")
        f.write(f"Dimensionality Profile:        {train_embeddings.shape[1]}D -> 3D\n\n")
        f.write("--- EXPLAINED VARIANCE METRICS ---\n")
        for k in range(3):
            f.write(f"  - Principal Component {k+1}: {var_ratios[k]*100:.4f}% of global variance\n")
        f.write(f"Total Cumulative Variance Captured: {np.sum(var_ratios)*100:.4f}%\n")
    print(f"[SUCCESS] Exported statistical text summary log -> {report_path}")

    # 5. Parse Path Mappings from Metadata CSV Log
    idx_to_path = {}
    with open(metadata_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_path[int(row['embedding_index'])] = row['image_path']

    # 6. Load Pretrained Diff-AE Network
    print("Loading pretrained Diffusion Autoencoder model checkpoints...")
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load(f'{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(conf.img_size),
        transforms.CenterCrop(conf.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def load_image_tensor(path):
        return transform(Image.open(path).convert('RGB')).unsqueeze(0).to(device)

    # 7. Perform Vector Traversal and Generate Visual Grids
    print("Beginning latent traversal along Principal Axes...")
    for k in range(3):
        pc_vector = pca.components_[k]
        scores_k = train_scores[:, k]
        sigma_k = np.std(scores_k)
        
        # Identify extreme outlier index points within the training split
        local_min_idx = np.argsort(scores_k)[int(0.10 * len(scores_k))]
        local_max_idx = np.argsort(scores_k)[int(0.90 * len(scores_k))]
        
        # Map back to absolute indices of the master dataset
        master_min_idx = train_indices[local_min_idx]
        master_max_idx = train_indices[local_max_idx]
        
        # Pull original base raw vectors
        z_low_base = train_embeddings[local_min_idx]
        z_high_base = train_embeddings[local_max_idx]
        
        # Define a safe jump distance step size (2.5 standard deviations)
        jump_step = 2.5 * sigma_k
        
        grid_tensors = []
        
        with torch.no_grad():
            # --- TOP ROW: LOW PRESENCE EMBEDDING -> INCREASING PC INFLUENCE ---
            img_low = load_image_tensor(idx_to_path[master_min_idx])
            # Pass original raw vector directly as a tensor
            z_low_tensor = torch.tensor(z_low_base, dtype=torch.float32, device=device).unsqueeze(0)
            xT_low = model.encode_stochastic(img_low, z_low_tensor, T=250)
            
            # Generate points: [Base Low, Jump 1, Jump 2]
            for step in [0, 1, 2]:
                z_manip = z_low_base + (step * jump_step) * pc_vector
                z_manip_tensor = torch.tensor(z_manip, dtype=torch.float32, device=device).unsqueeze(0)
                pred_img = model.render(xT_low, z_manip_tensor, T=20)
                grid_tensors.append(pred_img.squeeze(0).cpu())
                
            # --- BOTTOM ROW: HIGH PRESENCE EMBEDDING -> DECREASING PC INFLUENCE ---
            img_high = load_image_tensor(idx_to_path[master_max_idx])
            z_high_tensor = torch.tensor(z_high_base, dtype=torch.float32, device=device).unsqueeze(0)
            xT_high = model.encode_stochastic(img_high, z_high_tensor, T=250)
            
            # Generate points: [Base High, Jump 1, Jump 2]
            for step in [0, 1, 2]:
                z_manip = z_high_base - (step * jump_step) * pc_vector
                z_manip_tensor = torch.tensor(z_manip, dtype=torch.float32, device=device).unsqueeze(0)
                pred_img = model.render(xT_high, z_manip_tensor, T=20)
                grid_tensors.append(pred_img.squeeze(0).cpu())
                
        # Export compiled visual grid file layouts (2 rows by 3 columns)
        grid_path = out_dir / f"principal_component_{k+1}_attribute_grid.png"
        grid_tensors = [torch.clamp(img, 0.0, 1.0) for img in grid_tensors]
        grid_mesh = make_grid(grid_tensors, nrow=3, normalize=False)
        save_image(grid_mesh, grid_path)
        print(f" -> Exported Visual Grid for PC {k+1} to: {grid_path}")

    print("\nProcessing complete. Review your exported text logs and structural transformation grids.")

if __name__ == "__main__":
    TARGET_DATASET = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    run_pca_attribute_analysis(base_path_str=TARGET_DATASET)

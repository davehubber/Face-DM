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


def compute_pairwise_pc_metrics(
    train_scores: np.ndarray,
    n_pairs: int = 1_000_000,
    seed: int = 42,
):
    """
    Generate random training pairs exactly in the same spirit as
    ColdDiffAEDemorphDataset: idx1 is random, idx2 is idx1 plus a non-zero
    random offset modulo num_samples.

    Important math note:
    For c = (z1 + z2) / 2 and linear PCA projection, score(c) is exactly
    (score(z1) + score(z2)) / 2. Therefore, in a single PC, the average is
    always equally distant from z1 and z2. The average-asymmetry metric below
    is kept as a sanity check and should be ~0.

    The useful metric is therefore the PC with the largest pair separation:
        argmax_k |score_k(z1) - score_k(z2)|
    which tells you which of the first PCs most strongly separates the two
    embeddings in that pair.
    """
    num_samples, n_components = train_scores.shape

    rng = np.random.default_rng(seed)
    idx1 = rng.integers(0, num_samples, size=n_pairs)
    offsets = rng.integers(1, num_samples, size=n_pairs)
    idx2 = (idx1 + offsets) % num_samples

    scores_1 = train_scores[idx1]
    scores_2 = train_scores[idx2]
    scores_avg = (scores_1 + scores_2) / 2.0

    # This is the metric literally implied by comparing the average to each true
    # embedding along a PC. It should be numerically zero because PCA is linear.
    dist_avg_to_1 = np.abs(scores_avg - scores_1)
    dist_avg_to_2 = np.abs(scores_avg - scores_2)
    avg_asymmetry = np.abs(dist_avg_to_1 - dist_avg_to_2)

    # This is the meaningful version for a sorting/separation rule: which PC
    # gives the largest absolute score difference between z1 and z2?
    pair_separation = np.abs(scores_1 - scores_2)
    best_pc_by_separation = np.argmax(pair_separation, axis=1)
    best_pc_counts = np.bincount(best_pc_by_separation, minlength=n_components)
    best_pc_percentages = best_pc_counts / n_pairs * 100.0

    mean_abs_gap_per_pc = pair_separation.mean(axis=0)
    median_abs_gap_per_pc = np.median(pair_separation, axis=0)
    p95_abs_gap_per_pc = np.percentile(pair_separation, 95, axis=0)

    winning_gap = pair_separation[np.arange(n_pairs), best_pc_by_separation]

    return {
        "n_pairs": n_pairs,
        "seed": seed,
        "avg_asymmetry_mean_per_pc": avg_asymmetry.mean(axis=0),
        "avg_asymmetry_max_per_pc": avg_asymmetry.max(axis=0),
        "best_pc_counts": best_pc_counts,
        "best_pc_percentages": best_pc_percentages,
        "mean_abs_gap_per_pc": mean_abs_gap_per_pc,
        "median_abs_gap_per_pc": median_abs_gap_per_pc,
        "p95_abs_gap_per_pc": p95_abs_gap_per_pc,
        "winning_gap_mean": float(winning_gap.mean()),
        "winning_gap_median": float(np.median(winning_gap)),
        "winning_gap_p95": float(np.percentile(winning_gap, 95)),
    }


def run_pca_attribute_analysis(
    base_path_str: str,
    out_dir_str: str = "pca_analysis_results",
    n_components: int = 10,
    n_pair_metric_samples: int = 1_000_000,
    pair_metric_seed: int = 42,
):
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
    print(f"Fitting PCA model with top {n_components} components to raw training manifold...")
    pca = PCA(n_components=n_components)
    pca.fit(train_embeddings)

    # Transform to get the projection scores
    train_scores = pca.transform(train_embeddings).astype(np.float32)

    # 3. Store top components and mean for downstream scoring
    components_path = out_dir / f"pca_top{n_components}_components.npy"
    mean_path = out_dir / "pca_mean.npy"
    np.save(components_path, pca.components_)
    np.save(mean_path, pca.mean_)
    print(f"[SUCCESS] Saved projection references to:\n  -> {components_path}\n  -> {mean_path}")

    # 4. Generate pairwise PC separation metrics using random training pairs
    print(f"Computing pairwise PC metrics on {n_pair_metric_samples:,} random training pairs...")
    pair_metrics = compute_pairwise_pc_metrics(
        train_scores=train_scores,
        n_pairs=n_pair_metric_samples,
        seed=pair_metric_seed,
    )

    # 5. Generate Text Summary Report
    report_path = out_dir / "pca_statistical_report.txt"
    var_ratios = pca.explained_variance_ratio_
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=========================================================================" "\n")
        f.write("             EMBEDDING MANIFOLD PRINCIPAL COMPONENT REPORT" "\n")
        f.write("=========================================================================" "\n\n")
        f.write(f"Total Source Vectors Analyzed: {len(train_embeddings):,}\n")
        f.write(f"Dimensionality Profile:        {train_embeddings.shape[1]}D -> {n_components}D\n")
        f.write("PCA Input Space:               Raw DiffAE z_sem embeddings, no z-score normalization\n\n")

        f.write("--- EXPLAINED VARIANCE METRICS ---\n")
        for k in range(n_components):
            f.write(f"  - Principal Component {k+1:02d}: {var_ratios[k]*100:.4f}% of global variance\n")
        f.write(f"Total Cumulative Variance Captured by top {n_components}: {np.sum(var_ratios)*100:.4f}%\n\n")

        f.write("--- RANDOM TRAINING PAIR PC-SEPARATION METRICS ---\n")
        f.write(f"Random Pair Samples: {pair_metrics['n_pairs']:,}\n")
        f.write(f"Random Pair Seed:    {pair_metrics['seed']}\n")
        f.write("Pair Sampling Rule:  idx1 ~ Uniform[0, N-1]; offset ~ Uniform[1, N-1]; idx2 = (idx1 + offset) % N\n")
        f.write("Pair Embeddings:     Raw training embeddings, no z-score normalization\n\n")

        f.write("Sanity check: average-score asymmetry\n")
        f.write("  For c=(z1+z2)/2, PCA_score(c) is exactly the midpoint of PCA_score(z1) and PCA_score(z2).\n")
        f.write("  Therefore |score(c)-score(z1)| and |score(c)-score(z2)| should be equal for every PC.\n")
        for k in range(n_components):
            f.write(
                f"  - PC{k+1:02d}: mean asymmetry = {pair_metrics['avg_asymmetry_mean_per_pc'][k]:.8e}; "
                f"max asymmetry = {pair_metrics['avg_asymmetry_max_per_pc'][k]:.8e}\n"
            )
        f.write("\n")

        f.write("Useful separation metric: best PC per pair by max |score_k(z1) - score_k(z2)|\n")
        f.write("  This is the PC among the first 10 where the pair differs most strongly in PCA score.\n")
        for k in range(n_components):
            f.write(
                f"  - PC{k+1:02d}: wins {int(pair_metrics['best_pc_counts'][k]):,} pairs "
                f"({pair_metrics['best_pc_percentages'][k]:.2f}%); "
                f"mean |gap| = {pair_metrics['mean_abs_gap_per_pc'][k]:.6f}; "
                f"median |gap| = {pair_metrics['median_abs_gap_per_pc'][k]:.6f}; "
                f"p95 |gap| = {pair_metrics['p95_abs_gap_per_pc'][k]:.6f}\n"
            )
        f.write("\n")
        f.write("Winning-PC gap distribution across all pairs:\n")
        f.write(f"  - mean winning |gap|:   {pair_metrics['winning_gap_mean']:.6f}\n")
        f.write(f"  - median winning |gap|: {pair_metrics['winning_gap_median']:.6f}\n")
        f.write(f"  - p95 winning |gap|:    {pair_metrics['winning_gap_p95']:.6f}\n")

    print(f"[SUCCESS] Exported statistical text summary log -> {report_path}")

    # 6. Parse Path Mappings from Metadata CSV Log
    idx_to_path = {}
    with open(metadata_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_path[int(row['embedding_index'])] = row['image_path']

    # 7. Load Pretrained Diff-AE Network
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

    # 8. Perform Vector Traversal and Generate Visual Grids
    print(f"Beginning latent traversal along first {n_components} Principal Axes...")
    for k in range(n_components):
        pc_vector = pca.components_[k]
        scores_k = train_scores[:, k]
        sigma_k = np.std(scores_k)

        # Identify extreme outlier index points within the training split
        local_min_idx = np.argmin(scores_k)
        local_max_idx = np.argmax(scores_k)

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


import argparse
import csv
import math
import os
import random
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

from utils_semantic import load_zscore_stats


def load_diffae_autoencoder(checkpoint_path: str, device: torch.device):
    from templates import ffhq256_autoenc
    from config import PretrainConfig
    from experiment import LitModel

    conf = ffhq256_autoenc()
    conf.pretrain = PretrainConfig(name="ffhq256_autoenc", path=checkpoint_path)
    conf.latent_infer_path = None

    model = LitModel(conf).to(device)
    model.eval()
    model.ema_model.eval()
    return model


def unnormalize_semantic(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return z * std + mean


def load_train_pack(dataset_root: str) -> Dict:
    train_split_path = os.path.join(dataset_root, "semantic", "train_zsem.pt")
    if not os.path.exists(train_split_path):
        raise FileNotFoundError(f"Could not find semantic train split file: {train_split_path}")
    return torch.load(train_split_path, map_location="cpu")


def compute_pca_on_standardized_embeddings(z_std: torch.Tensor, num_components: int = 3) -> Dict[str, torch.Tensor]:
    if z_std.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {tuple(z_std.shape)}")

    pca_center = z_std.mean(dim=0, keepdim=True)
    centered = z_std - pca_center

    denom = max(centered.shape[0] - 1, 1)
    covariance = centered.T @ centered / denom

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total_variance = eigenvalues.sum().clamp_min(1e-12)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_explained_variance_ratio = torch.cumsum(explained_variance_ratio, dim=0)

    components = eigenvectors[:, :num_components].T.contiguous()  # [K, D]
    top_eigenvalues = eigenvalues[:num_components].contiguous()
    top_explained = explained_variance_ratio[:num_components].contiguous()
    top_cumulative = cumulative_explained_variance_ratio[:num_components].contiguous()

    scores = centered @ components.T  # [N, K]

    recon_mse_by_k: List[float] = []
    total_centered_mse = centered.pow(2).mean().item()
    for k in range(1, num_components + 1):
        recon = scores[:, :k] @ components[:k]
        mse = (centered - recon).pow(2).mean().item()
        recon_mse_by_k.append(mse)

    return {
        "pca_center": pca_center.squeeze(0),
        "components": components,
        "eigenvalues": top_eigenvalues,
        "explained_variance_ratio": top_explained,
        "cumulative_explained_variance_ratio": top_cumulative,
        "scores": scores,
        "total_centered_mse": torch.tensor(total_centered_mse, dtype=torch.float32),
        "recon_mse_by_k": torch.tensor(recon_mse_by_k, dtype=torch.float32),
    }


def build_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load_image_tensor(path: str, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = build_transform()(image).unsqueeze(0).to(device)
    return tensor


@torch.no_grad()
def decode_pc_influence_grids(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pack = load_train_pack(args.dataset_root)
    raw_embeddings = pack["z_sem"].to(torch.float32)
    sample_ids = list(pack["sample_ids"])
    source_paths = list(pack["source_paths"])

    mean, std = load_zscore_stats(args.dataset_root)
    z_std = (raw_embeddings - mean) / std

    pca = compute_pca_on_standardized_embeddings(z_std, num_components=3)
    pca_center = pca["pca_center"]
    components = pca["components"]
    eigenvalues = pca["eigenvalues"]
    explained_variance_ratio = pca["explained_variance_ratio"]
    cumulative_explained_variance_ratio = pca["cumulative_explained_variance_ratio"]
    scores = pca["scores"]
    recon_mse_by_k = pca["recon_mse_by_k"]
    total_centered_mse = float(pca["total_centered_mse"].item())

    semantic_dir = os.path.join(args.dataset_root, "semantic")
    analysis_dir = os.path.join(semantic_dir, "pca_analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    pca_stats_path = os.path.join(analysis_dir, "semantic_pca_stats.pt")
    torch.save(
        {
            "raw_feature_mean": mean,
            "raw_feature_std": std,
            "pca_center_standardized": pca_center,
            "components_standardized": components,
            "eigenvalues": eigenvalues,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_explained_variance_ratio": cumulative_explained_variance_ratio,
            "num_samples": raw_embeddings.shape[0],
            "embedding_dim": raw_embeddings.shape[1],
        },
        pca_stats_path,
    )
    print(f"Saved reusable PCA stats to: {pca_stats_path}")

    rng = random.Random(args.seed)
    if args.sample_index is None:
        sample_index = rng.randrange(len(sample_ids))
    else:
        sample_index = int(args.sample_index)
        if not (0 <= sample_index < len(sample_ids)):
            raise ValueError(f"sample_index must be in [0, {len(sample_ids) - 1}], got {sample_index}")

    sample_id = sample_ids[sample_index]
    source_path = source_paths[sample_index]
    sample_std = z_std[sample_index]
    sample_raw = raw_embeddings[sample_index]
    sample_scores = scores[sample_index]

    print(f"Selected sample index: {sample_index}")
    print(f"Selected sample id:    {sample_id}")
    print(f"Selected source path:  {source_path}")

    diffae = load_diffae_autoencoder(args.diffae_checkpoint, device)

    image_tensor = load_image_tensor(source_path, device)
    z_gt = diffae.encode(image_tensor)
    xT = diffae.encode_stochastic(image_tensor, z_gt, T=args.decode_t_invert)

    mean_device = mean.to(device).unsqueeze(0)
    std_device = std.to(device).unsqueeze(0)

    sigma_multipliers = [float(x) for x in args.sigma_multipliers]

    for pc_idx in range(3):
        component = components[pc_idx]
        pc_std = math.sqrt(max(float(eigenvalues[pc_idx].item()), 1e-12))
        base_score = float(sample_scores[pc_idx].item())

        decoded_images = []
        csv_rows = []

        for sigma_mult in sigma_multipliers:
            delta = args.direction * sigma_mult * pc_std
            modified_std = sample_std + delta * component
            modified_raw = unnormalize_semantic(
                modified_std.unsqueeze(0).to(device),
                mean_device,
                std_device,
            )
            decoded = diffae.render(xT, cond=modified_raw, T=args.decode_t_decode).detach().cpu().clamp(0, 1)
            decoded_images.append(decoded)

            csv_rows.append(
                {
                    "pc": pc_idx + 1,
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "source_path": source_path,
                    "sigma_multiplier": sigma_mult,
                    "direction": args.direction,
                    "pc_score_before": base_score,
                    "pc_score_after": base_score + delta,
                    "pc_score_delta": delta,
                    "pc_score_std": pc_std,
                }
            )

        grid = make_grid(
            torch.cat(decoded_images, dim=0),
            nrow=len(decoded_images),
            padding=8,
            pad_value=1.0,
        )

        grid_path = os.path.join(analysis_dir, f"pc{pc_idx + 1}_influence_grid.png")
        transforms.ToPILImage()(grid).save(grid_path)

        csv_path = os.path.join(analysis_dir, f"pc{pc_idx + 1}_influence_grid.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "pc",
                    "sample_index",
                    "sample_id",
                    "source_path",
                    "sigma_multiplier",
                    "direction",
                    "pc_score_before",
                    "pc_score_after",
                    "pc_score_delta",
                    "pc_score_std",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"Saved PC{pc_idx + 1} influence grid to: {grid_path}")
        print(f"Saved PC{pc_idx + 1} metadata to:      {csv_path}")

    metrics_path = os.path.join(analysis_dir, "pca_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("Semantic Embedding PCA Analysis\n")
        f.write("===============================\n\n")
        f.write(f"Dataset root: {args.dataset_root}\n")
        f.write(f"Number of samples: {raw_embeddings.shape[0]}\n")
        f.write(f"Embedding dimension: {raw_embeddings.shape[1]}\n")
        f.write(f"Selected sample index: {sample_index}\n")
        f.write(f"Selected sample id: {sample_id}\n")
        f.write(f"Selected source path: {source_path}\n")
        f.write(f"Sigma multipliers used for grids: {sigma_multipliers}\n")
        f.write(f"Direction used for grids: {args.direction}\n\n")

        f.write("Global reconstruction metrics in standardized PCA space\n")
        f.write("------------------------------------------------------\n")
        f.write(f"Total centered embedding MSE: {total_centered_mse:.8f}\n")
        for k in range(1, 4):
            mse_k = float(recon_mse_by_k[k - 1].item())
            captured_fraction = 1.0 - (mse_k / max(total_centered_mse, 1e-12))
            f.write(
                f"Top-{k} reconstruction residual MSE: {mse_k:.8f} | "
                f"captured variance fraction: {captured_fraction:.8f}\n"
            )
        f.write("\n")

        f.write("Per-component metrics\n")
        f.write("---------------------\n")
        for pc_idx in range(3):
            pc_scores = scores[:, pc_idx]
            eigval = float(eigenvalues[pc_idx].item())
            explained = float(explained_variance_ratio[pc_idx].item())
            cumulative = float(cumulative_explained_variance_ratio[pc_idx].item())
            score_std = math.sqrt(max(eigval, 1e-12))
            mean_abs_score = float(pc_scores.abs().mean().item())
            p95_abs_score = float(torch.quantile(pc_scores.abs(), 0.95).item())
            sample_score = float(sample_scores[pc_idx].item())

            f.write(f"PC{pc_idx + 1}\n")
            f.write(f"  eigenvalue: {eigval:.8f}\n")
            f.write(f"  explained variance ratio: {explained:.8f}\n")
            f.write(f"  cumulative explained variance ratio: {cumulative:.8f}\n")
            f.write(f"  projection score std: {score_std:.8f}\n")
            f.write(f"  mean absolute projection score: {mean_abs_score:.8f}\n")
            f.write(f"  95th percentile absolute projection score: {p95_abs_score:.8f}\n")
            f.write(f"  selected sample projection score: {sample_score:.8f}\n\n")

    print(f"Saved PCA metrics report to: {metrics_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PCA on z-scored semantic DiffAE embeddings, save reusable PCA statistics, "
            "and decode a random sample modified along PC1, PC2, and PC3."
        )
    )
    parser.add_argument(
        "--dataset_root",
        default="/nas-ctm01/homes/dacordeiro/Face-DM/encoded_ffhq256_semantic_split",
        help="Dataset root containing semantic/train_zsem.pt",
    )
    parser.add_argument(
        "--diffae_checkpoint",
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt",
        help="FFHQ256 DiffAE autoencoder checkpoint",
    )
    parser.add_argument(
        "--decode_t_invert",
        default=100,
        type=int,
        help="DDIM inversion steps for stochastic encoding",
    )
    parser.add_argument(
        "--decode_t_decode",
        default=100,
        type=int,
        help="Render steps for decoding",
    )
    parser.add_argument(
        "--sample_index",
        default=None,
        type=int,
        help="Optional fixed train-split sample index to visualize. If omitted, a random one is chosen.",
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="Random seed used only when sample_index is not provided.",
    )
    parser.add_argument(
        "--sigma_multipliers",
        nargs=3,
        default=[1.0, 2.0, 3.0],
        type=float,
        help="Three increasing multiples of the PC score std used for the three decoded images in each grid.",
    )
    parser.add_argument(
        "--direction",
        default=1.0,
        type=float,
        help="Direction along the PC. Use +1.0 or -1.0. The sign of a principal component is arbitrary.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    decode_pc_influence_grids(args)

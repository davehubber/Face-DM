import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# =========================================================
# 1. Latent de-morphing model definitions
#    Same architecture as train_latent_diffae.py
# =========================================================

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
            nn.Linear(time_emb_dim * 2, time_emb_dim),
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


class DeterministicColdDemorph(nn.Module):
    def __init__(self, model, num_timesteps=300):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.sqrt_2 = math.sqrt(2.0)

    def degrade(self, z1, z2, t):
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        w1 = torch.sqrt(1.0 - 0.5 * gamma)
        w2 = torch.sqrt(0.5 * gamma)
        return w1 * z1 + w2 * z2

    @torch.no_grad()
    def tacos_sample_loop(self, c):
        device = c.device
        b = c.shape[0]

        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = c.clone()

        prev_pred_z1 = None

        for t in tqdm(timesteps, desc="TACOs Sampling", leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            pred_raw = self.model(x_t, t_batch)
            pred_raw_1, pred_raw_2 = pred_raw.chunk(2, dim=-1)

            if prev_pred_z1 is None:
                pred_z1 = pred_raw_1
            else:
                dist_1 = F.l1_loss(
                    pred_raw_1,
                    prev_pred_z1,
                    reduction="none",
                ).mean(dim=-1)

                dist_2 = F.l1_loss(
                    pred_raw_2,
                    prev_pred_z1,
                    reduction="none",
                ).mean(dim=-1)

                swap_mask = dist_2 < dist_1
                pred_z1 = torch.where(
                    swap_mask.unsqueeze(-1),
                    pred_raw_2,
                    pred_raw_1,
                )

            pred_z2 = self.sqrt_2 * c - pred_z1
            prev_pred_z1 = pred_z1

            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)

            deg_t = self.degrade(pred_z1, pred_z2, t_batch)
            deg_t_prev = self.degrade(pred_z1, pred_z2, t_prev_batch)

            x_t = x_t - deg_t + deg_t_prev

        final_z1 = x_t
        final_z2 = self.sqrt_2 * c - final_z1

        return final_z1, final_z2


# =========================================================
# 2. DiffAE model loading
# =========================================================

def load_diffae_ffhq256_autoencoder(
    diffae_root: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    diffae_root = Path(diffae_root).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    if not diffae_root.exists():
        raise FileNotFoundError(f"DiffAE repo not found: {diffae_root}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"DiffAE checkpoint not found: {checkpoint_path}")

    sys.path.insert(0, str(diffae_root))

    old_cwd = os.getcwd()
    os.chdir(diffae_root)

    try:
        from templates import ffhq256_autoenc
        from config import PretrainConfig
        from experiment import LitModel

        conf = ffhq256_autoenc()
        conf.pretrain = PretrainConfig(
            name="ffhq256_autoenc",
            path=str(checkpoint_path),
        )
        conf.latent_infer_path = None

        model = LitModel(conf)
        model = model.to(device)
        model.eval()

        if hasattr(model, "ema_model"):
            model.ema_model.eval()

        for p in model.parameters():
            p.requires_grad_(False)

    finally:
        os.chdir(old_cwd)

    return model


# =========================================================
# 3. Dataset and image helpers
# =========================================================

class TestPairImageSpaceDataset(Dataset):
    def __init__(
        self,
        raw_pairs: np.ndarray,
        xt_pairs: np.ndarray,
        metadata_csv: Path,
        image_size: int = 256,
    ):
        self.raw_pairs = raw_pairs.astype(np.float32)
        self.xt_pairs = xt_pairs.astype(np.float32)

        with open(metadata_csv, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        if len(self.rows) != len(self.raw_pairs):
            raise ValueError("Mismatch between semantic test pairs and metadata rows.")

        if len(self.rows) != len(self.xt_pairs):
            raise ValueError("Mismatch between stochastic test pairs and metadata rows.")

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        gt1 = self._load_image(row["image_path_A"])
        gt2 = self._load_image(row["image_path_B"])

        return {
            "z1_raw": torch.from_numpy(self.raw_pairs[idx, 0]),
            "z2_raw": torch.from_numpy(self.raw_pairs[idx, 1]),
            "xt1": torch.from_numpy(self.xt_pairs[idx, 0]),
            "xt2": torch.from_numpy(self.xt_pairs[idx, 1]),
            "gt1": gt1,
            "gt2": gt2,
            "pair_index": int(row["pair_index"]),
            "idx_a": int(row["embedding_index_A"]),
            "idx_b": int(row["embedding_index_B"]),
            "path_a": row["image_path_A"],
            "path_b": row["image_path_B"],
            "file_a": row["filename_A"],
            "file_b": row["filename_B"],
        }


# Important range convention used here:
# - Dataset images are normalized to [-1, 1].
# - DiffAE rendered images are assumed to already be in [0, 1].
#   Therefore, do NOT apply (x + 1) / 2 to decoded images.

def gt_norm_to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert ground-truth tensors from [-1, 1] to [0, 1]."""
    return x.clamp(-1, 1).add(1.0).div(2.0)


def decoded_to_01(x: torch.Tensor) -> torch.Tensor:
    """Keep decoded DiffAE tensors in their expected [0, 1] range."""
    return x.clamp(0, 1)


def gt_norm_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    x01 = gt_norm_to_01(x.detach().cpu())
    arr = x01.permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    return arr


def decoded_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    x01 = decoded_to_01(x.detach().cpu())
    arr = x01.permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    return arr


def decoded_to_lpips_input(x: torch.Tensor) -> torch.Tensor:
    """LPIPS expects tensors in [-1, 1]."""
    return decoded_to_01(x).mul(2.0).sub(1.0)


def slerp_tensors(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Spherical interpolation between two stochastic DiffAE codes.

    v0, v1: [B, 3, 256, 256]
    output: [B, 3, 256, 256]
    """
    original_shape = v0.shape

    v0_flat = v0.reshape(v0.shape[0], -1)
    v1_flat = v1.reshape(v1.shape[0], -1)

    v0_norm = F.normalize(v0_flat, p=2, dim=1)
    v1_norm = F.normalize(v1_flat, p=2, dim=1)

    dot = (v0_norm * v1_norm).sum(dim=1, keepdim=True)
    dot = dot.clamp(-1.0, 1.0)

    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    use_lerp = sin_omega.abs() < 1e-6

    safe_sin_omega = torch.where(
        use_lerp,
        torch.ones_like(sin_omega),
        sin_omega,
    )

    part0 = torch.sin((1.0 - t) * omega) / safe_sin_omega
    part1 = torch.sin(t * omega) / safe_sin_omega

    out_flat = part0 * v0_flat + part1 * v1_flat
    lerp_flat = (1.0 - t) * v0_flat + t * v1_flat

    out_flat = torch.where(use_lerp, lerp_flat, out_flat)

    return out_flat.reshape(original_shape)


# =========================================================
# 4. DeepFace ArcFace helpers
# =========================================================

def load_deepface():
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise ImportError(
            "DeepFace is required for ArcFace evaluation. "
            "Install it with: pip install deepface"
        ) from exc

    return DeepFace


def deepface_arcface_embedding(
    img_rgb_uint8: np.ndarray,
    DeepFace,
    detector_backend: str = "skip",
) -> np.ndarray:
    """
    Computes a L2-normalized DeepFace ArcFace embedding.

    The default detector_backend is 'skip' because both FFHQ and the decoded
    DiffAE images are already centered face images. This also avoids generated
    images being discarded because of detector failures.
    """
    reps = DeepFace.represent(
        img_path=img_rgb_uint8,
        model_name="ArcFace",
        detector_backend=detector_backend,
        enforce_detection=False,
        align=False,
        normalization="ArcFace",
    )

    if isinstance(reps, list):
        emb = reps[0]["embedding"]
    else:
        emb = reps["embedding"]

    emb = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb)

    if norm == 0:
        return emb

    return emb / norm


# =========================================================
# 5. Metric helpers
# =========================================================

def tmr_at_fmr(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    fmr: float,
) -> Tuple[float, float]:
    """
    Returns (threshold, TMR%) at a target false match rate.

    The threshold is selected from the impostor-score distribution so that
    approximately fmr of impostor scores are accepted.
    """
    threshold = float(np.quantile(impostor_scores, 1.0 - fmr))
    tmr = float((genuine_scores >= threshold).mean() * 100.0)
    return threshold, tmr


# =========================================================
# 6. Evaluation
# =========================================================

def evaluate_image_space(
    diffae_base_path: str,
    run_name: str,
    diffae_root: str,
    diffae_checkpoint: str,
    experiments_root: str,
    num_timesteps: int,
    mode: str,
    decode_steps: int,
    batch_size: int,
    num_workers: int,
    grid_rows: int,
    ra_threshold: float,
    deepface_detector_backend: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = Path(diffae_base_path).resolve()
    embeddings_dir = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_test_pairs", "")

    raw_pairs_path = embeddings_dir / f"{stem}_test_pairs.npy"
    xt_pairs_path = embeddings_dir / "ffhq256_diffae_xt_test_pairs.npy"
    metadata_path = embeddings_dir / f"{stem}_test_pairs_metadata.csv"
    mean_path = embeddings_dir / f"{stem}_train_mean.npy"
    std_path = embeddings_dir / f"{stem}_train_std.npy"

    exp_dir = Path(experiments_root).resolve() / run_name
    ckpt_path = exp_dir / "checkpoints" / "best.pt"

    report_path = exp_dir / f"eval_image_space_{mode}.txt"
    grid_path = exp_dir / f"eval_image_space_{mode}_grid.png"

    for path in [
        raw_pairs_path,
        xt_pairs_path,
        metadata_path,
        mean_path,
        std_path,
        ckpt_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    exp_dir.mkdir(parents=True, exist_ok=True)

    raw_pairs = np.load(raw_pairs_path).astype(np.float32)
    xt_pairs = np.load(xt_pairs_path).astype(np.float32)

    train_mean = np.load(mean_path).astype(np.float32)
    train_std = np.load(std_path).astype(np.float32)
    train_std = np.where(train_std == 0, 1.0, train_std)

    dataset = TestPairImageSpaceDataset(
        raw_pairs=raw_pairs,
        xt_pairs=xt_pairs,
        metadata_csv=metadata_path,
        image_size=256,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Loading latent de-morphing model...")

    net = ColdDemorphNet().to(device)
    diffusion = DeterministicColdDemorph(
        model=net,
        num_timesteps=num_timesteps,
    ).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    print("Loading DiffAE FFHQ256 autoencoder...")

    diffae_model = load_diffae_ffhq256_autoencoder(
        diffae_root=Path(diffae_root),
        checkpoint_path=Path(diffae_checkpoint),
        device=device,
    )

    print("Loading LPIPS model...")

    try:
        import lpips
    except ImportError as exc:
        raise ImportError(
            "LPIPS is required for this evaluation. "
            "Install it with: pip install lpips"
        ) from exc

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    print("Loading DeepFace ArcFace...")

    DeepFace = load_deepface()

    # -----------------------------------------------------
    # Build ArcFace gallery from the 1000 unique test images.
    # In your split, side B is a derangement of side A, so all
    # test identities/images are present in side A.
    # -----------------------------------------------------
    print("Building ArcFace gallery from the 1000 unique test images...")

    gallery_embeddings: List[np.ndarray] = []
    gallery_index_to_pos: Dict[int, int] = {}

    for pos, row in enumerate(tqdm(dataset.rows, desc="ArcFace gallery")):
        emb_idx = int(row["embedding_index_A"])
        img_path = row["image_path_A"]

        img = Image.open(img_path).convert("RGB")
        img_t = dataset.transform(img)
        img_uint8 = gt_norm_to_uint8_rgb(img_t)

        emb = deepface_arcface_embedding(
            img_rgb_uint8=img_uint8,
            DeepFace=DeepFace,
            detector_backend=deepface_detector_backend,
        )

        gallery_embeddings.append(emb)
        gallery_index_to_pos[emb_idx] = pos

    gallery_embeddings = np.stack(gallery_embeddings, axis=0).astype(np.float32)

    psnr_scores: List[float] = []
    ssim_scores: List[float] = []
    lpips_scores: List[float] = []

    genuine_scores: List[float] = []
    closest_impostor_scores: List[float] = []

    grid_images: List[torch.Tensor] = []

    total_pairs = 0
    sqrt_half = math.sqrt(0.5)

    mean_t = torch.from_numpy(train_mean).to(device).float().unsqueeze(0)
    std_t = torch.from_numpy(train_std).to(device).float().unsqueeze(0)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Image-space evaluation"):
            z1_raw = batch["z1_raw"].to(device).float()
            z2_raw = batch["z2_raw"].to(device).float()

            xt1 = batch["xt1"].to(device).float()
            xt2 = batch["xt2"].to(device).float()

            gt1 = batch["gt1"].to(device).float()
            gt2 = batch["gt2"].to(device).float()

            z1_norm = (z1_raw - mean_t) / std_t
            z2_norm = (z2_raw - mean_t) / std_t

            c_vp = sqrt_half * z1_norm + sqrt_half * z2_norm

            if mode == "iterative":
                pred1_norm, pred2_norm = diffusion.tacos_sample_loop(c_vp)

            elif mode == "one_shot":
                t_batch = torch.full(
                    (z1_norm.shape[0],),
                    num_timesteps,
                    device=device,
                    dtype=torch.long,
                )

                pred = net(c_vp, t_batch)
                pred1_norm, _ = pred.chunk(2, dim=-1)
                pred2_norm = math.sqrt(2.0) * c_vp - pred1_norm

            else:
                raise ValueError("mode must be either 'iterative' or 'one_shot'.")

            # Align prediction order to GT order.
            dist_a = (
                F.l1_loss(pred1_norm, z1_norm, reduction="none").mean(dim=1)
                + F.l1_loss(pred2_norm, z2_norm, reduction="none").mean(dim=1)
            )

            dist_b = (
                F.l1_loss(pred1_norm, z2_norm, reduction="none").mean(dim=1)
                + F.l1_loss(pred2_norm, z1_norm, reduction="none").mean(dim=1)
            )

            keep_order = (dist_a <= dist_b).unsqueeze(-1)

            pred1_aligned_norm = torch.where(keep_order, pred1_norm, pred2_norm)
            pred2_aligned_norm = torch.where(keep_order, pred2_norm, pred1_norm)

            pred1_raw = pred1_aligned_norm * std_t + mean_t
            pred2_raw = pred2_aligned_norm * std_t + mean_t

            # Realistic stochastic conditioning:
            # both outputs use the morph stochastic midpoint.
            morph_xt = slerp_tensors(xt1, xt2, t=0.5)

            pred1_img = diffae_model.render(morph_xt, pred1_raw, T=decode_steps)
            pred2_img = diffae_model.render(morph_xt, pred2_raw, T=decode_steps)

            # LPIPS expects [-1, 1].
            lpips_1 = lpips_model(decoded_to_lpips_input(pred1_img), gt1).view(-1).detach().cpu().tolist()
            lpips_2 = lpips_model(decoded_to_lpips_input(pred2_img), gt2).view(-1).detach().cpu().tolist()

            lpips_scores.extend(lpips_1)
            lpips_scores.extend(lpips_2)

            # PSNR and SSIM in [0, 1].
            pred1_01 = decoded_to_01(pred1_img).detach().cpu()
            pred2_01 = decoded_to_01(pred2_img).detach().cpu()
            gt1_01 = gt_norm_to_01(gt1).detach().cpu()
            gt2_01 = gt_norm_to_01(gt2).detach().cpu()

            current_batch_size = pred1_01.shape[0]

            for i in range(current_batch_size):
                pairs = [
                    (
                        pred1_01[i].permute(1, 2, 0).numpy(),
                        gt1_01[i].permute(1, 2, 0).numpy(),
                    ),
                    (
                        pred2_01[i].permute(1, 2, 0).numpy(),
                        gt2_01[i].permute(1, 2, 0).numpy(),
                    ),
                ]

                for pred_np, gt_np in pairs:
                    psnr_scores.append(
                        peak_signal_noise_ratio(
                            gt_np,
                            pred_np,
                            data_range=1.0,
                        )
                    )

                    ssim_scores.append(
                        structural_similarity(
                            gt_np,
                            pred_np,
                            data_range=1.0,
                            channel_axis=2,
                        )
                    )

            # ArcFace biometric metrics.
            for i in range(current_batch_size):
                idx_a = int(batch["idx_a"][i])
                idx_b = int(batch["idx_b"][i])

                exclude_positions = {
                    gallery_index_to_pos[idx_a],
                    gallery_index_to_pos[idx_b],
                }

                output_items = [
                    (pred1_img[i], idx_a),
                    (pred2_img[i], idx_b),
                ]

                for pred_img_tensor, gt_idx in output_items:
                    pred_uint8 = decoded_to_uint8_rgb(pred_img_tensor)

                    pred_emb = deepface_arcface_embedding(
                        img_rgb_uint8=pred_uint8,
                        DeepFace=DeepFace,
                        detector_backend=deepface_detector_backend,
                    )

                    gt_pos = gallery_index_to_pos[gt_idx]

                    genuine = float(np.dot(pred_emb, gallery_embeddings[gt_pos]))
                    genuine_scores.append(genuine)

                    all_scores = gallery_embeddings @ pred_emb

                    impostor_mask = np.ones(len(all_scores), dtype=bool)
                    for pos in exclude_positions:
                        impostor_mask[pos] = False

                    closest_impostor = float(all_scores[impostor_mask].max())
                    closest_impostor_scores.append(closest_impostor)

            # Qualitative grid.
            for i in range(current_batch_size):
                if len(grid_images) >= grid_rows * 5:
                    break

                morph_z = 0.5 * (z1_raw[i : i + 1] + z2_raw[i : i + 1])
                morph_xt_i = morph_xt[i : i + 1]

                morph_img = diffae_model.render(
                    morph_xt_i,
                    morph_z,
                    T=decode_steps,
                )[0].detach().cpu()

                row_images = [
                    decoded_to_01(morph_img),
                    gt_norm_to_01(gt1[i].detach().cpu()),
                    gt_norm_to_01(gt2[i].detach().cpu()),
                    decoded_to_01(pred1_img[i].detach().cpu()),
                    decoded_to_01(pred2_img[i].detach().cpu()),
                ]

                grid_images.extend(row_images)

            total_pairs += current_batch_size

    psnr_scores = np.asarray(psnr_scores, dtype=np.float32)
    ssim_scores = np.asarray(ssim_scores, dtype=np.float32)
    lpips_scores = np.asarray(lpips_scores, dtype=np.float32)

    genuine_scores = np.asarray(genuine_scores, dtype=np.float32)
    closest_impostor_scores = np.asarray(closest_impostor_scores, dtype=np.float32)

    psnr_mean = float(psnr_scores.mean())
    psnr_std = float(psnr_scores.std())

    ssim_mean = float(ssim_scores.mean())
    ssim_std = float(ssim_scores.std())

    lpips_mean = float(lpips_scores.mean())
    lpips_std = float(lpips_scores.std())

    genuine_mean = float(genuine_scores.mean())
    genuine_std = float(genuine_scores.std())

    impostor_mean = float(closest_impostor_scores.mean())
    impostor_std = float(closest_impostor_scores.std())

    threshold_10, tmr_10 = tmr_at_fmr(
        genuine_scores=genuine_scores,
        impostor_scores=closest_impostor_scores,
        fmr=0.10,
    )

    threshold_1, tmr_1 = tmr_at_fmr(
        genuine_scores=genuine_scores,
        impostor_scores=closest_impostor_scores,
        fmr=0.01,
    )

    threshold_01, tmr_01 = tmr_at_fmr(
        genuine_scores=genuine_scores,
        impostor_scores=closest_impostor_scores,
        fmr=0.001,
    )

    ra = float((genuine_scores >= ra_threshold).mean() * 100.0)

    if len(grid_images) != grid_rows * 5:
        raise RuntimeError(
            f"Expected {grid_rows * 5} images for the grid, "
            f"but got {len(grid_images)}."
        )

    grid_tensor = torch.stack(grid_images, dim=0)

    torchvision.utils.save_image(
        grid_tensor,
        grid_path,
        nrow=5,
        normalize=False,
    )

    report = (
        "Image-space evaluation for latent DiffAE de-morphing\n"
        "===================================================\n"
        f"Run name                       : {run_name}\n"
        f"Mode                           : {mode}\n"
        f"Latent checkpoint              : {ckpt_path}\n"
        f"DiffAE checkpoint              : {diffae_checkpoint}\n"
        f"Semantic test pairs            : {raw_pairs_path}\n"
        f"Stochastic test pairs          : {xt_pairs_path}\n"
        f"Pair metadata                  : {metadata_path}\n"
        f"Decode steps                   : {decode_steps}\n"
        f"Number of test pairs           : {total_pairs}\n"
        f"Number of evaluated outputs    : {len(genuine_scores)}\n"
        f"DeepFace detector backend      : {deepface_detector_backend}\n"
        f"Prediction stochastic code     : morph stochastic midpoint, slerp(xT1, xT2, 0.5)\n"
        f"Decoded image range assumption : DiffAE render output already in [0, 1]\n"
        f"RA threshold                   : {ra_threshold:.3f}\n"
        "\n"
        "[Image quality metrics]\n"
        f"PSNR  : mean = {psnr_mean:.6f} | std = {psnr_std:.6f}\n"
        f"SSIM  : mean = {ssim_mean:.6f} | std = {ssim_std:.6f}\n"
        f"LPIPS : mean = {lpips_mean:.6f} | std = {lpips_std:.6f}\n"
        "\n"
        "[Biometric metrics - DeepFace ArcFace cosine similarity]\n"
        f"Mean genuine similarity        : {genuine_mean:.6f} | std = {genuine_std:.6f}\n"
        f"Mean closest impostor sim.     : {impostor_mean:.6f} | std = {impostor_std:.6f}\n"
        f"Restoration Accuracy (RA)      : {ra:.4f}%\n"
        f"TMR @ 10% FMR                  : {tmr_10:.4f}% | threshold = {threshold_10:.6f}\n"
        f"TMR @ 1% FMR                   : {tmr_1:.4f}% | threshold = {threshold_1:.6f}\n"
        f"TMR @ 0.1% FMR                 : {tmr_01:.4f}% | threshold = {threshold_01:.6f}\n"
        "\n"
        "[Qualitative grid]\n"
        "Each row contains:\n"
        "morph | source 1 | source 2 | decoded prediction 1 | decoded prediction 2\n"
        f"Grid path                      : {grid_path}\n"
        "\n"
        "[Output files]\n"
        f"Text report                    : {report_path}\n"
        f"Image grid                     : {grid_path}\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)


# =========================================================
# 7. CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate latent DiffAE de-morphing in image-space."
    )

    parser.add_argument(
        "--diffae-base-path",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy",
        help="Path to the full DiffAE semantic embedding file.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="diffae_baseline",
        help="Name of the experiment folder containing checkpoints/best.pt.",
    )

    parser.add_argument(
        "--experiments-root",
        type=str,
        default="experiments",
        help="Root folder containing the experiment folders.",
    )

    parser.add_argument(
        "--diffae-root",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/diffae/",
        help="Path to the cloned DiffAE repository.",
    )

    parser.add_argument(
        "--diffae-checkpoint",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt",
        help="Path to the DiffAE FFHQ256 autoencoder checkpoint.",
    )

    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=300,
        help="Number of cold diffusion timesteps used by the trained latent model.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="iterative",
        choices=["iterative", "one_shot"],
        help="Use the iterative TACOs sampler or one-shot prediction.",
    )

    parser.add_argument(
        "--decode-steps",
        type=int,
        default=20,
        help="Number of DiffAE rendering steps.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for image-space decoding/evaluation.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--grid-rows",
        type=int,
        default=5,
        help="Number of qualitative rows to save in the image grid.",
    )

    parser.add_argument(
        "--ra-threshold",
        type=float,
        default=0.4,
        help="Fixed ArcFace similarity threshold for Restoration Accuracy.",
    )

    parser.add_argument(
        "--deepface-detector-backend",
        type=str,
        default="skip",
        help=(
            "DeepFace detector backend. Default is 'skip' because FFHQ/DiffAE "
            "images are already centered face images."
        ),
    )

    args = parser.parse_args()

    evaluate_image_space(
        diffae_base_path=args.diffae_base_path,
        run_name=args.run_name,
        diffae_root=args.diffae_root,
        diffae_checkpoint=args.diffae_checkpoint,
        experiments_root=args.experiments_root,
        num_timesteps=args.num_timesteps,
        mode=args.mode,
        decode_steps=args.decode_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grid_rows=args.grid_rows,
        ra_threshold=args.ra_threshold,
        deepface_detector_backend=args.deepface_detector_backend,
    )

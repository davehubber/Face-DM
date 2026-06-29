import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# =========================================================
# 1. Shared model blocks
# =========================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class AdaLNBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.silu = nn.SiLU()
        self.cond_proj = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        return self.silu(self.norm(h) * (1 + scale) + shift)


# =========================================================
# 2. Regular conditional model from train_latent_diffae_cond.py
# =========================================================

class ConditionalColdDemorphNet(nn.Module):
    def __init__(self, x_dim=512, hidden_dim=2048, num_layers=10, time_emb_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        joint_cond_dim = time_emb_dim + x_dim

        self.blocks = nn.ModuleList()
        self.blocks.append(AdaLNBlock(x_dim, hidden_dim, joint_cond_dim))
        for _ in range(num_layers - 1):
            self.blocks.append(AdaLNBlock(hidden_dim + x_dim, hidden_dim, joint_cond_dim))

        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, v_cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        joint_cond = torch.cat([t_emb, v_cond], dim=-1)

        h = x_t
        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, joint_cond)
            else:
                h = block(torch.cat([h, x_t], dim=-1), joint_cond)

        return self.final_linear(h)


class ConditionalColdDemorph(nn.Module):
    def __init__(self, model: nn.Module, num_timesteps=300):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, M: torch.Tensor, target: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        return gamma * M + (1 - gamma) * target

    def timestep_condition(self, M: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        use_external_condition = (t == self.num_timesteps).view(-1, 1)
        return torch.where(use_external_condition, condition, M)

    @torch.no_grad()
    def sample_loop(self, M: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        device = M.device
        b = M.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = M.clone()

        for t in tqdm(timesteps, desc="Conditional sampling", leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            step_condition = self.timestep_condition(M, condition, t_batch)
            pred_target = self.model(x_t, t_batch, step_condition)

            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            deg_t = self.degrade(M, pred_target, t_batch)
            deg_t_prev = self.degrade(M, pred_target, t_prev_batch)
            x_t = x_t - deg_t + deg_t_prev

        return x_t


# =========================================================
# 3. Conditional refinement model from train_latent_diffae_cond_refine.py
# =========================================================

class RefineColdDemorphNet(nn.Module):
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

        self.final_linear = nn.Linear(hidden_dim, x_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = x_t

        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, t_emb)
            else:
                h = block(torch.cat([h, x_t], dim=-1), t_emb)

        return self.final_linear(h)


class MathBaselineColdDemorph(nn.Module):
    def __init__(self, model: nn.Module, num_timesteps=10):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    @torch.no_grad()
    def sample_loop(self, M: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        device = M.device
        b = M.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()

        z_coarse = (2.0 * M) - condition
        x_t = z_coarse.clone()

        for t in tqdm(timesteps, desc="Refinement sampling", leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            pred_target = self.model(x_t, t_batch)

            gamma_t = (t_batch / self.num_timesteps).view(-1, 1).float()
            gamma_prev = ((t_batch - 1) / self.num_timesteps).view(-1, 1).float()

            deg_t = gamma_t * z_coarse + (1 - gamma_t) * pred_target
            deg_t_prev = gamma_prev * z_coarse + (1 - gamma_prev) * pred_target
            x_t = x_t - deg_t + deg_t_prev

        return x_t


# =========================================================
# 4. Reference-free baseline model from train_latent_diffae.py
# =========================================================

class BaselineColdDemorphNet(nn.Module):
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

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = x

        for i, block in enumerate(self.blocks):
            if i == 0:
                h = block(h, t_emb)
            else:
                h = block(torch.cat([h, x], dim=-1), t_emb)

        return self.final_linear(h)


class BaselineColdDemorph(nn.Module):
    def __init__(self, model: nn.Module, num_timesteps=300):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

    def degrade(self, z1: torch.Tensor, z2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        gamma = (t / self.num_timesteps).view(-1, 1).float()
        w1 = torch.sqrt(1.0 - 0.5 * gamma)
        w2 = torch.sqrt(0.5 * gamma)
        return w1 * z1 + w2 * z2

    @torch.no_grad()
    def informed_sample_target(self, c: torch.Tensor, z_known: torch.Tensor) -> torch.Tensor:
        """
        Informed evaluation used for the reference-free baseline.
        z_known is only used during sampling to re-degrade the target prediction.
        The returned vector is the predicted unknown/target embedding.
        """
        device = c.device
        b = c.shape[0]
        timesteps = torch.arange(self.num_timesteps, 0, -1, device=device).long()
        x_t = c.clone()

        for t in tqdm(timesteps, desc="Informed baseline sampling", leave=False):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            pred_raw = self.model(x_t, t_batch)
            pred_raw_1, pred_raw_2 = pred_raw.chunk(2, dim=-1)

            dist_1 = F.l1_loss(pred_raw_1, z_known, reduction="none").mean(dim=-1)
            dist_2 = F.l1_loss(pred_raw_2, z_known, reduction="none").mean(dim=-1)

            target_mask = (dist_1 > dist_2).unsqueeze(-1)
            pred_target = torch.where(target_mask, pred_raw_1, pred_raw_2)

            t_prev_batch = torch.full((b,), t - 1, device=device, dtype=torch.long)
            deg_t = self.degrade(pred_target, z_known, t_batch)
            deg_t_prev = self.degrade(pred_target, z_known, t_prev_batch)
            x_t = x_t - deg_t + deg_t_prev

        return x_t


# =========================================================
# 5. DiffAE loading and image helpers
# =========================================================

class PathImageDataset(Dataset):
    def __init__(self, paths: List[str], image_size: int = 256):
        self.paths = [str(p) for p in paths]
        self.transform = image_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return {
            "img": self.transform(img),
            "path": path,
            "index": idx,
        }


def image_transform(image_size: int = 256):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def load_diffae_ffhq256_autoencoder(diffae_root: Path, checkpoint_path: Path, device: torch.device):
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
        conf.pretrain = PretrainConfig(name="ffhq256_autoenc", path=str(checkpoint_path))
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


# Range convention:
# - images loaded from disk are normalized to [-1, 1]
# - DiffAE render() is assumed to already return [0, 1]
#   so decoded outputs are NOT converted using (x + 1) / 2.

def gt_norm_to_01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-1, 1).add(1.0).div(2.0)


def decoded_to_01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0, 1)


def gt_path_to_tensor(path: str, device: torch.device, transform=None) -> torch.Tensor:
    if transform is None:
        transform = image_transform(256)
    img = Image.open(path).convert("RGB")
    return transform(img).to(device)


def gt_tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    x01 = gt_norm_to_01(x.detach().cpu())
    arr = x01.permute(1, 2, 0).numpy()
    return (arr * 255.0).round().clip(0, 255).astype(np.uint8)


def decoded_tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    x01 = decoded_to_01(x.detach().cpu())
    arr = x01.permute(1, 2, 0).numpy()
    return (arr * 255.0).round().clip(0, 255).astype(np.uint8)


def slerp_tensors(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    original_shape = v0.shape
    v0_flat = v0.reshape(v0.shape[0], -1)
    v1_flat = v1.reshape(v1.shape[0], -1)

    v0_norm = F.normalize(v0_flat, p=2, dim=1)
    v1_norm = F.normalize(v1_flat, p=2, dim=1)

    dot = (v0_norm * v1_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    use_lerp = sin_omega.abs() < 1e-6
    safe_sin_omega = torch.where(use_lerp, torch.ones_like(sin_omega), sin_omega)

    out_flat = (
        torch.sin((1.0 - t) * omega) / safe_sin_omega * v0_flat
        + torch.sin(t * omega) / safe_sin_omega * v1_flat
    )
    lerp_flat = (1.0 - t) * v0_flat + t * v1_flat
    out_flat = torch.where(use_lerp, lerp_flat, out_flat)

    return out_flat.reshape(original_shape)


# =========================================================
# 6. File and metadata helpers
# =========================================================

def resolve_existing_path(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.exists():
            return path
    raise FileNotFoundError("None of these files were found: " + ", ".join(str(root / name) for name in names))


def optional_existing_path(root: Path, *names: str) -> Optional[Path]:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def read_csv_rows(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metadata_by_embedding_index(path: Path) -> Dict[int, dict]:
    rows = read_csv_rows(path)
    return {int(row["embedding_index"]): row for row in rows}


def load_checkpoint_state_dict(path: Path, model_state_key: str = "model_state_dict"):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and model_state_key in state:
        return state[model_state_key]
    return state


@dataclass
class MorphEvalData:
    root: Path
    train_bf_raw: np.ndarray
    eval_bf_raw: np.ndarray
    eval_morph_raw: np.ndarray
    eval_morph_xt: np.ndarray
    records: List[dict]
    mean: np.ndarray
    std: np.ndarray


def ensure_eval_morph_xt(
    root: Path,
    diffae_model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_size: int = 256,
) -> np.ndarray:
    xt_path = optional_existing_path(root, "eval_morph_xt.npy", "eval_morph_xt_1000.npy")
    if xt_path is not None:
        return np.load(xt_path).astype(np.float32)

    metadata_path = resolve_existing_path(root, "eval_morph_metadata.csv", "eval_morph_metadata_1000.csv")
    rows = read_csv_rows(metadata_path)

    print(f"No eval morph x_T file found in {root}. Computing and caching eval_morph_xt.npy...")

    rows_sorted = sorted(rows, key=lambda r: int(r["embedding_index"]))
    image_paths = [row["image_path"] for row in rows_sorted]
    max_idx = max(int(row["embedding_index"]) for row in rows_sorted)

    dataset = PathImageDataset(image_paths, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    xt_array = None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding eval morph x_T"):
            imgs = batch["img"].to(device, non_blocking=True)
            z_sem = diffae_model.encode(imgs)
            x_t = diffae_model.encode_stochastic(imgs, z_sem, T=250)
            x_t_np = x_t.detach().float().cpu().numpy()

            if xt_array is None:
                xt_array = np.zeros((max_idx + 1,) + x_t_np.shape[1:], dtype=np.float32)

            for local_i, dataset_pos in enumerate(batch["index"].tolist()):
                emb_idx = int(rows_sorted[dataset_pos]["embedding_index"])
                xt_array[emb_idx] = x_t_np[local_i]

    out_path = root / "eval_morph_xt.npy"
    np.save(out_path, xt_array)
    print(f"Saved computed morph stochastic codes to: {out_path}")
    return xt_array.astype(np.float32)


def load_morph_eval_data(
    data_dir: str,
    diffae_model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> MorphEvalData:
    root = Path(data_dir).resolve()

    train_bf_raw = np.load(root / "train_bonafide_zsem.npy").astype(np.float32)
    eval_bf_raw = np.load(root / "eval_bonafide_zsem.npy").astype(np.float32)
    eval_morph_raw = np.load(resolve_existing_path(root, "eval_morph_zsem.npy", "eval_morph_zsem_1000.npy")).astype(np.float32)
    eval_morph_xt = ensure_eval_morph_xt(root, diffae_model, device, batch_size, num_workers)

    mean = train_bf_raw.mean(axis=0, keepdims=True).astype(np.float32)
    std = (train_bf_raw.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

    morph_metadata_path = resolve_existing_path(root, "eval_morph_metadata.csv", "eval_morph_metadata_1000.csv")
    morph_rows = read_csv_rows(morph_metadata_path)

    records = []
    for row in morph_rows:
        idx_m = int(row["embedding_index"])
        col_a = "source_idx_A_in_train_bf" if "source_idx_A_in_train_bf" in row else "source_idx_A_in_eval_bf"
        col_b = "source_idx_B_in_train_bf" if "source_idx_B_in_train_bf" in row else "source_idx_B_in_eval_bf"
        idx_a = int(row[col_a])
        idx_b = int(row[col_b])

        if idx_a < 0 or idx_b < 0:
            continue

        bf_meta_path = root / "train_bonafide_metadata.csv" if col_a.endswith("train_bf") else root / "eval_bonafide_metadata.csv"
        if not bf_meta_path.exists():
            bf_meta_path = root / "train_bonafide_metadata.csv"

        # The metadata source can differ by record only in theory; in practice it is constant.
        bf_meta_by_idx = metadata_by_embedding_index(bf_meta_path)
        rec = {
            "pair_id": idx_m,
            "morph_idx": idx_m,
            "source_a_idx": idx_a,
            "source_b_idx": idx_b,
            "morph_path": row["image_path"],
            "morph_filename": row.get("filename", ""),
            "source_a_path": bf_meta_by_idx[idx_a]["image_path"],
            "source_b_path": bf_meta_by_idx[idx_b]["image_path"],
            "source_a_filename": bf_meta_by_idx[idx_a].get("filename", ""),
            "source_b_filename": bf_meta_by_idx[idx_b].get("filename", ""),
        }
        records.append(rec)

    if not records:
        raise ValueError(f"No valid evaluation records found in {root}")

    return MorphEvalData(
        root=root,
        train_bf_raw=train_bf_raw,
        eval_bf_raw=eval_bf_raw,
        eval_morph_raw=eval_morph_raw,
        eval_morph_xt=eval_morph_xt,
        records=records,
        mean=mean,
        std=std,
    )


@dataclass
class BaselineEvalData:
    root: Path
    raw_pairs: np.ndarray
    xt_pairs: np.ndarray
    records: List[dict]
    mean: np.ndarray
    std: np.ndarray


def load_baseline_eval_data(diffae_base_path: str) -> BaselineEvalData:
    base_path = Path(diffae_base_path).resolve()
    root = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_test_pairs", "")

    raw_pairs = np.load(root / f"{stem}_test_pairs.npy").astype(np.float32)
    xt_pairs = np.load(root / "ffhq256_diffae_xt_test_pairs.npy").astype(np.float32)
    mean = np.load(root / f"{stem}_train_mean.npy").astype(np.float32)[None, :]
    std = np.load(root / f"{stem}_train_std.npy").astype(np.float32)[None, :]
    std = np.where(std == 0, 1.0, std).astype(np.float32)

    metadata_path = root / f"{stem}_test_pairs_metadata.csv"
    rows = read_csv_rows(metadata_path)

    records = []
    for row in rows:
        records.append({
            "pair_id": int(row["pair_index"]),
            "source_a_idx": int(row["embedding_index_A"]),
            "source_b_idx": int(row["embedding_index_B"]),
            "source_a_path": row["image_path_A"],
            "source_b_path": row["image_path_B"],
            "source_a_filename": row.get("filename_A", ""),
            "source_b_filename": row.get("filename_B", ""),
        })

    return BaselineEvalData(
        root=root,
        raw_pairs=raw_pairs,
        xt_pairs=xt_pairs,
        records=records,
        mean=mean,
        std=std,
    )


# =========================================================
# 7. DeepFace ArcFace and metric helpers
# =========================================================

def load_deepface():
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise ImportError("DeepFace is required. Install it with: pip install deepface") from exc
    return DeepFace


def deepface_arcface_embedding(img_rgb_uint8: np.ndarray, DeepFace, detector_backend: str) -> np.ndarray:
    reps = DeepFace.represent(
        img_path=img_rgb_uint8,
        model_name="ArcFace",
        detector_backend=detector_backend,
        enforce_detection=False,
        align=False,
        normalization="ArcFace",
    )

    emb = reps[0]["embedding"] if isinstance(reps, list) else reps["embedding"]
    emb = np.asarray(emb, dtype=np.float32)
    norm = np.linalg.norm(emb)
    return emb if norm == 0 else emb / norm


class ArcFaceCache:
    def __init__(self, DeepFace, device: torch.device, detector_backend: str):
        self.DeepFace = DeepFace
        self.device = device
        self.detector_backend = detector_backend
        self.transform = image_transform(256)
        self.cache: Dict[str, np.ndarray] = {}

    def source_embedding(self, path: str) -> np.ndarray:
        path = str(path)
        if path not in self.cache:
            img_t = gt_path_to_tensor(path, self.device, self.transform)
            img_uint8 = gt_tensor_to_uint8_rgb(img_t)
            self.cache[path] = deepface_arcface_embedding(img_uint8, self.DeepFace, self.detector_backend)
        return self.cache[path]

    def decoded_embedding(self, decoded_img: torch.Tensor) -> np.ndarray:
        img_uint8 = decoded_tensor_to_uint8_rgb(decoded_img)
        return deepface_arcface_embedding(img_uint8, self.DeepFace, self.detector_backend)


def calibrate_arcface_threshold(
    source_paths: List[str],
    arcface_cache: ArcFaceCache,
    fmr: float = 0.001,
    max_pairs: int = 200_000,
    seed: int = 42,
) -> float:
    unique_paths = sorted(set(str(p) for p in source_paths))
    if len(unique_paths) < 2:
        raise ValueError("Need at least two source images to calibrate an impostor threshold.")

    embeddings = np.stack([arcface_cache.source_embedding(p) for p in tqdm(unique_paths, desc="ArcFace source embeddings")], axis=0)
    n = len(embeddings)
    total_pairs = n * (n - 1) // 2

    if total_pairs <= max_pairs:
        scores = []
        for i in range(n):
            sims = embeddings[i + 1:] @ embeddings[i]
            scores.extend(sims.tolist())
        scores = np.asarray(scores, dtype=np.float32)
    else:
        rng = random.Random(seed)
        scores = np.empty(max_pairs, dtype=np.float32)
        for k in range(max_pairs):
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
            scores[k] = float(np.dot(embeddings[i], embeddings[j]))

    return float(np.quantile(scores, 1.0 - fmr))


@dataclass
class MetricsBucket:
    scores_ref: List[float]
    scores_acc: List[float]
    successes: List[int]

    def add(self, score_ref: float, score_acc: float, success: bool):
        self.scores_ref.append(float(score_ref))
        self.scores_acc.append(float(score_acc))
        self.successes.append(int(success))


def empty_bucket() -> MetricsBucket:
    return MetricsBucket(scores_ref=[], scores_acc=[], successes=[])


def summarize_bucket(bucket: MetricsBucket, threshold: float) -> dict:
    if not bucket.scores_ref:
        return {
            "num_samples": 0,
            "accuracy": float("nan"),
            "ASC": float("nan"),
            "ASA": float("nan"),
            "DCI": float("nan"),
            "DAI": float("nan"),
        }

    scores_ref = np.asarray(bucket.scores_ref, dtype=np.float32)
    scores_acc = np.asarray(bucket.scores_acc, dtype=np.float32)
    successes = np.asarray(bucket.successes, dtype=np.float32)

    asc = float(scores_ref.mean())
    asa = float(scores_acc.mean())

    return {
        "num_samples": int(len(scores_ref)),
        "accuracy": float(successes.mean() * 100.0),
        "ASC": asc,
        "ASA": asa,
        "DCI": asc - threshold,
        "DAI": asa - threshold,
    }


# =========================================================
# 8. Shared image-space evaluation engine
# =========================================================

class ImageSpaceEvaluator:
    def __init__(
        self,
        diffae_model,
        arcface_cache: ArcFaceCache,
        device: torch.device,
        output_dir: Path,
        decode_steps: int,
        grid_rows: int,
        threshold: float,
    ):
        self.diffae_model = diffae_model
        self.arcface_cache = arcface_cache
        self.device = device
        self.output_dir = output_dir
        self.decode_steps = decode_steps
        self.grid_rows = grid_rows
        self.threshold = threshold
        self.transform = image_transform(256)

    def evaluate_decoded_batch(
        self,
        test_name: str,
        direction: str,
        pred_z_raw: torch.Tensor,
        morph_xt: torch.Tensor,
        pair_ids: List[int],
        morph_paths: List[Optional[str]],
        reference_paths: List[str],
        target_paths: List[str],
        rows_out: List[dict],
        buckets: Dict[str, MetricsBucket],
        grid_images: List[torch.Tensor],
        morph_grid_images: Optional[torch.Tensor] = None,
    ):
        decoded = self.diffae_model.render(morph_xt, pred_z_raw, T=self.decode_steps)
        decoded = decoded_to_01(decoded)

        b = decoded.shape[0]

        for i in range(b):
            decoded_i = decoded[i]

            ref_emb = self.arcface_cache.source_embedding(reference_paths[i])
            acc_emb = self.arcface_cache.source_embedding(target_paths[i])
            pred_emb = self.arcface_cache.decoded_embedding(decoded_i)

            score_ref = float(np.dot(pred_emb, ref_emb))
            score_acc = float(np.dot(pred_emb, acc_emb))
            success = (score_acc >= self.threshold) and (score_ref < self.threshold)

            rows_out.append({
                "test_name": test_name,
                "direction": direction,
                "pair_id": pair_ids[i],
                "morph_path": morph_paths[i] if morph_paths[i] is not None else "",
                "reference_path": reference_paths[i],
                "target_path": target_paths[i],
                "score_reference": f"{score_ref:.8f}",
                "score_accomplice": f"{score_acc:.8f}",
                "success": int(success),
            })

            buckets[direction].add(score_ref, score_acc, success)
            buckets["combined"].add(score_ref, score_acc, success)

            if len(grid_images) < self.grid_rows * 4:
                if morph_grid_images is not None:
                    morph_img_01 = decoded_to_01(morph_grid_images[i].detach().cpu())
                elif morph_paths[i] is not None:
                    morph_img = gt_path_to_tensor(morph_paths[i], self.device, self.transform)
                    morph_img_01 = gt_norm_to_01(morph_img.detach().cpu())
                else:
                    morph_img_01 = torch.zeros_like(decoded_i.detach().cpu())

                ref_img = gt_path_to_tensor(reference_paths[i], self.device, self.transform)
                tgt_img = gt_path_to_tensor(target_paths[i], self.device, self.transform)

                grid_images.extend([
                    morph_img_01,
                    gt_norm_to_01(ref_img.detach().cpu()),
                    gt_norm_to_01(tgt_img.detach().cpu()),
                    decoded_i.detach().cpu(),
                ])


# =========================================================
# 9. Test evaluators
# =========================================================

def evaluate_conditional_test(
    test_cfg: dict,
    shared,
) -> Tuple[List[dict], List[dict]]:
    device = shared["device"]
    output_root = shared["output_root"]
    experiments_root = Path(shared["experiments_root"]).resolve()
    batch_size = shared["batch_size"]
    num_workers = shared["num_workers"]
    mode = test_cfg.get("mode", shared["mode"])
    num_timesteps = int(test_cfg.get("num_timesteps", 300))
    test_name = test_cfg["name"]

    data = load_morph_eval_data(test_cfg["data_dir"], shared["diffae_model"], device, batch_size, num_workers)

    source_paths = []
    for rec in data.records:
        source_paths.extend([rec["source_a_path"], rec["source_b_path"]])

    threshold = shared["threshold_resolver"](test_name, source_paths)

    exp_dir = experiments_root / test_cfg["run_name"]
    ckpt_path = exp_dir / "checkpoints" / test_cfg.get("checkpoint", "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {test_name}: {ckpt_path}")

    net = ConditionalColdDemorphNet().to(device)
    net.load_state_dict(load_checkpoint_state_dict(ckpt_path))
    net.eval()
    diffusion = ConditionalColdDemorph(net, num_timesteps=num_timesteps).to(device)

    test_dir = output_root / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ImageSpaceEvaluator(
        diffae_model=shared["diffae_model"],
        arcface_cache=shared["arcface_cache"],
        device=device,
        output_dir=test_dir,
        decode_steps=shared["decode_steps"],
        grid_rows=shared["grid_rows"],
        threshold=threshold,
    )

    mean_t = torch.from_numpy(data.mean).to(device).float()
    std_t = torch.from_numpy(data.std).to(device).float()

    eval_bf_norm = (data.eval_bf_raw - data.mean) / data.std
    eval_morph_norm = (data.eval_morph_raw - data.mean) / data.std

    rows_out: List[dict] = []
    buckets = {"A_to_B": empty_bucket(), "B_to_A": empty_bucket(), "combined": empty_bucket()}
    grid_images: List[torch.Tensor] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(data.records), batch_size), desc=f"{test_name} predictions"):
            batch_records = data.records[start:start + batch_size]
            morph_indices = [r["morph_idx"] for r in batch_records]
            a_indices = [r["source_a_idx"] for r in batch_records]
            b_indices = [r["source_b_idx"] for r in batch_records]

            b_M = torch.tensor(eval_morph_norm[morph_indices], device=device).float()
            b_A = torch.tensor(eval_bf_norm[a_indices], device=device).float()
            b_B = torch.tensor(eval_bf_norm[b_indices], device=device).float()
            b_xt = torch.tensor(data.eval_morph_xt[morph_indices], device=device).float()

            if mode == "iterative":
                pred_B_norm = diffusion.sample_loop(b_M, b_A)
                pred_A_norm = diffusion.sample_loop(b_M, b_B)
            elif mode == "one_shot":
                t_max = torch.full((b_M.shape[0],), num_timesteps, device=device, dtype=torch.long)
                pred_B_norm = net(b_M, t_max, b_A)
                pred_A_norm = net(b_M, t_max, b_B)
            else:
                raise ValueError("mode must be 'iterative' or 'one_shot'.")

            pred_B_raw = pred_B_norm * std_t + mean_t
            pred_A_raw = pred_A_norm * std_t + mean_t

            pair_ids = [r["pair_id"] for r in batch_records]
            morph_paths = [r["morph_path"] for r in batch_records]

            evaluator.evaluate_decoded_batch(
                test_name=test_name,
                direction="A_to_B",
                pred_z_raw=pred_B_raw,
                morph_xt=b_xt,
                pair_ids=pair_ids,
                morph_paths=morph_paths,
                reference_paths=[r["source_a_path"] for r in batch_records],
                target_paths=[r["source_b_path"] for r in batch_records],
                rows_out=rows_out,
                buckets=buckets,
                grid_images=grid_images,
            )

            evaluator.evaluate_decoded_batch(
                test_name=test_name,
                direction="B_to_A",
                pred_z_raw=pred_A_raw,
                morph_xt=b_xt,
                pair_ids=pair_ids,
                morph_paths=morph_paths,
                reference_paths=[r["source_b_path"] for r in batch_records],
                target_paths=[r["source_a_path"] for r in batch_records],
                rows_out=rows_out,
                buckets=buckets,
                grid_images=grid_images,
            )

    summary_rows = write_outputs(test_name, test_dir, rows_out, buckets, threshold, grid_images)
    return rows_out, summary_rows


def evaluate_refine_test(test_cfg: dict, shared) -> Tuple[List[dict], List[dict]]:
    device = shared["device"]
    output_root = shared["output_root"]
    experiments_root = Path(shared["experiments_root"]).resolve()
    batch_size = shared["batch_size"]
    num_workers = shared["num_workers"]
    mode = test_cfg.get("mode", shared["mode"])
    num_timesteps = int(test_cfg.get("num_timesteps", 10))
    test_name = test_cfg["name"]

    data = load_morph_eval_data(test_cfg["data_dir"], shared["diffae_model"], device, batch_size, num_workers)

    source_paths = []
    for rec in data.records:
        source_paths.extend([rec["source_a_path"], rec["source_b_path"]])

    threshold = shared["threshold_resolver"](test_name, source_paths)

    exp_dir = experiments_root / test_cfg["run_name"]
    ckpt_path = exp_dir / "checkpoints" / test_cfg.get("checkpoint", "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {test_name}: {ckpt_path}")

    net = RefineColdDemorphNet(num_layers=10).to(device)
    net.load_state_dict(load_checkpoint_state_dict(ckpt_path))
    net.eval()
    diffusion = MathBaselineColdDemorph(net, num_timesteps=num_timesteps).to(device)

    test_dir = output_root / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ImageSpaceEvaluator(
        diffae_model=shared["diffae_model"],
        arcface_cache=shared["arcface_cache"],
        device=device,
        output_dir=test_dir,
        decode_steps=shared["decode_steps"],
        grid_rows=shared["grid_rows"],
        threshold=threshold,
    )

    mean_t = torch.from_numpy(data.mean).to(device).float()
    std_t = torch.from_numpy(data.std).to(device).float()

    eval_bf_norm = (data.eval_bf_raw - data.mean) / data.std
    eval_morph_norm = (data.eval_morph_raw - data.mean) / data.std

    rows_out: List[dict] = []
    buckets = {"A_to_B": empty_bucket(), "B_to_A": empty_bucket(), "combined": empty_bucket()}
    grid_images: List[torch.Tensor] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(data.records), batch_size), desc=f"{test_name} predictions"):
            batch_records = data.records[start:start + batch_size]
            morph_indices = [r["morph_idx"] for r in batch_records]
            a_indices = [r["source_a_idx"] for r in batch_records]
            b_indices = [r["source_b_idx"] for r in batch_records]

            b_M = torch.tensor(eval_morph_norm[morph_indices], device=device).float()
            b_A = torch.tensor(eval_bf_norm[a_indices], device=device).float()
            b_B = torch.tensor(eval_bf_norm[b_indices], device=device).float()
            b_xt = torch.tensor(data.eval_morph_xt[morph_indices], device=device).float()

            if mode == "iterative":
                pred_B_norm = diffusion.sample_loop(b_M, b_A)
                pred_A_norm = diffusion.sample_loop(b_M, b_B)
            elif mode == "one_shot":
                t_max = torch.full((b_M.shape[0],), num_timesteps, device=device, dtype=torch.long)
                z_coarse_B = (2.0 * b_M) - b_A
                z_coarse_A = (2.0 * b_M) - b_B
                pred_B_norm = net(z_coarse_B, t_max)
                pred_A_norm = net(z_coarse_A, t_max)
            else:
                raise ValueError("mode must be 'iterative' or 'one_shot'.")

            pred_B_raw = pred_B_norm * std_t + mean_t
            pred_A_raw = pred_A_norm * std_t + mean_t

            pair_ids = [r["pair_id"] for r in batch_records]
            morph_paths = [r["morph_path"] for r in batch_records]

            evaluator.evaluate_decoded_batch(
                test_name=test_name,
                direction="A_to_B",
                pred_z_raw=pred_B_raw,
                morph_xt=b_xt,
                pair_ids=pair_ids,
                morph_paths=morph_paths,
                reference_paths=[r["source_a_path"] for r in batch_records],
                target_paths=[r["source_b_path"] for r in batch_records],
                rows_out=rows_out,
                buckets=buckets,
                grid_images=grid_images,
            )

            evaluator.evaluate_decoded_batch(
                test_name=test_name,
                direction="B_to_A",
                pred_z_raw=pred_A_raw,
                morph_xt=b_xt,
                pair_ids=pair_ids,
                morph_paths=morph_paths,
                reference_paths=[r["source_b_path"] for r in batch_records],
                target_paths=[r["source_a_path"] for r in batch_records],
                rows_out=rows_out,
                buckets=buckets,
                grid_images=grid_images,
            )

    summary_rows = write_outputs(test_name, test_dir, rows_out, buckets, threshold, grid_images)
    return rows_out, summary_rows


def evaluate_baseline_informed_test(test_cfg: dict, shared) -> Tuple[List[dict], List[dict]]:
    device = shared["device"]
    output_root = shared["output_root"]
    experiments_root = Path(shared["experiments_root"]).resolve()
    batch_size = shared["batch_size"]
    num_timesteps = int(test_cfg.get("num_timesteps", 300))
    test_name = test_cfg["name"]

    data = load_baseline_eval_data(test_cfg["diffae_base_path"])

    source_paths = []
    for rec in data.records:
        source_paths.extend([rec["source_a_path"], rec["source_b_path"]])

    threshold = shared["threshold_resolver"](test_name, source_paths)

    exp_dir = experiments_root / test_cfg["run_name"]
    ckpt_path = exp_dir / "checkpoints" / test_cfg.get("checkpoint", "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {test_name}: {ckpt_path}")

    net = BaselineColdDemorphNet().to(device)
    net.load_state_dict(load_checkpoint_state_dict(ckpt_path))
    net.eval()
    diffusion = BaselineColdDemorph(net, num_timesteps=num_timesteps).to(device)

    test_dir = output_root / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ImageSpaceEvaluator(
        diffae_model=shared["diffae_model"],
        arcface_cache=shared["arcface_cache"],
        device=device,
        output_dir=test_dir,
        decode_steps=shared["decode_steps"],
        grid_rows=shared["grid_rows"],
        threshold=threshold,
    )

    mean_t = torch.from_numpy(data.mean).to(device).float()
    std_t = torch.from_numpy(data.std).to(device).float()

    raw_pairs_norm = (data.raw_pairs - data.mean[:, None, :]) / data.std[:, None, :]

    rows_out: List[dict] = []
    buckets = {"A_to_B": empty_bucket(), "B_to_A": empty_bucket(), "combined": empty_bucket()}
    grid_images: List[torch.Tensor] = []

    sqrt_half = math.sqrt(0.5)

    with torch.no_grad():
        for start in tqdm(range(0, len(data.records), batch_size), desc=f"{test_name} predictions"):
            end = min(start + batch_size, len(data.records))
            batch_records = data.records[start:end]
            idx = np.arange(start, end)

            zA_norm = torch.tensor(raw_pairs_norm[idx, 0], device=device).float()
            zB_norm = torch.tensor(raw_pairs_norm[idx, 1], device=device).float()
            zA_raw = torch.tensor(data.raw_pairs[idx, 0], device=device).float()
            zB_raw = torch.tensor(data.raw_pairs[idx, 1], device=device).float()

            xtA = torch.tensor(data.xt_pairs[idx, 0], device=device).float()
            xtB = torch.tensor(data.xt_pairs[idx, 1], device=device).float()
            morph_xt = slerp_tensors(xtA, xtB, t=0.5)

            c_vp = sqrt_half * zA_norm + sqrt_half * zB_norm

            pred_B_norm = diffusion.informed_sample_target(c_vp, zA_norm)
            pred_A_norm = diffusion.informed_sample_target(c_vp, zB_norm)

            pred_B_raw = pred_B_norm * std_t + mean_t
            pred_A_raw = pred_A_norm * std_t + mean_t

            # Baseline has synthetic morphs. Generate their display images from midpoint semantic + midpoint stochastic.
            morph_z = 0.5 * (zA_raw + zB_raw)
            morph_imgs = shared["diffae_model"].render(morph_xt, morph_z, T=shared["decode_steps"])
            morph_imgs = decoded_to_01(morph_imgs.detach().cpu())

            pair_ids = [r["pair_id"] for r in batch_records]
            morph_paths = [None for _ in batch_records]

            evaluator.evaluate_decoded_batch(
                test_name=test_name,
                direction="A_to_B",
                pred_z_raw=pred_B_raw,
                morph_xt=morph_xt,
                pair_ids=pair_ids,
                morph_paths=morph_paths,
                reference_paths=[r["source_a_path"] for r in batch_records],
                target_paths=[r["source_b_path"] for r in batch_records],
                rows_out=rows_out,
                buckets=buckets,
                grid_images=grid_images,
                morph_grid_images=morph_imgs,
            )

            evaluator.evaluate_decoded_batch(
                test_name=test_name,
                direction="B_to_A",
                pred_z_raw=pred_A_raw,
                morph_xt=morph_xt,
                pair_ids=pair_ids,
                morph_paths=morph_paths,
                reference_paths=[r["source_b_path"] for r in batch_records],
                target_paths=[r["source_a_path"] for r in batch_records],
                rows_out=rows_out,
                buckets=buckets,
                grid_images=grid_images,
                morph_grid_images=morph_imgs,
            )

    summary_rows = write_outputs(test_name, test_dir, rows_out, buckets, threshold, grid_images)
    return rows_out, summary_rows


# =========================================================
# 10. Output helpers
# =========================================================

def write_outputs(
    test_name: str,
    test_dir: Path,
    rows_out: List[dict],
    buckets: Dict[str, MetricsBucket],
    threshold: float,
    grid_images: List[torch.Tensor],
) -> List[dict]:
    per_sample_csv = test_dir / f"{test_name}_per_sample.csv"
    summary_csv = test_dir / f"{test_name}_summary.csv"
    report_txt = test_dir / f"{test_name}_report.txt"
    grid_path = test_dir / f"{test_name}_grid.png"

    fieldnames = [
        "test_name",
        "direction",
        "pair_id",
        "morph_path",
        "reference_path",
        "target_path",
        "score_reference",
        "score_accomplice",
        "success",
    ]

    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    summary_rows = []
    for direction in ["A_to_B", "B_to_A", "combined"]:
        s = summarize_bucket(buckets[direction], threshold)
        row = {
            "test_name": test_name,
            "direction": direction,
            "arcface_threshold": f"{threshold:.8f}",
            "num_samples": s["num_samples"],
            "accuracy": f"{s['accuracy']:.6f}",
            "ASC": f"{s['ASC']:.8f}",
            "ASA": f"{s['ASA']:.8f}",
            "DCI": f"{s['DCI']:.8f}",
            "DAI": f"{s['DAI']:.8f}",
        }
        summary_rows.append(row)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if grid_images:
        grid_tensor = torch.stack(grid_images, dim=0)
        torchvision.utils.save_image(grid_tensor, grid_path, nrow=4, normalize=False)

    report_lines = [
        f"Reference-based image-space evaluation: {test_name}",
        "====================================================",
        f"ArcFace threshold S: {threshold:.8f}",
        "Metrics follow the Accuracy / DCI / DAI protocol:",
        "  success = score_accomplice >= S and score_reference < S",
        "  ASC = mean(score_reference)",
        "  ASA = mean(score_accomplice)",
        "  DCI = ASC - S",
        "  DAI = ASA - S",
        "Decoder range convention: DiffAE render output is treated as already in [0, 1].",
        "",
    ]

    for row in summary_rows:
        report_lines.extend([
            f"[{row['direction']}]",
            f"  samples  : {row['num_samples']}",
            f"  Accuracy : {row['accuracy']}%",
            f"  ASC      : {row['ASC']}",
            f"  ASA      : {row['ASA']}",
            f"  DCI      : {row['DCI']}",
            f"  DAI      : {row['DAI']}",
            "",
        ])

    report_lines.extend([
        "Output files:",
        f"  Per-sample CSV : {per_sample_csv}",
        f"  Summary CSV    : {summary_csv}",
        f"  Grid           : {grid_path}",
    ])

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n" + "\n".join(report_lines))
    return summary_rows


def write_global_outputs(output_root: Path, all_rows: List[dict], all_summary_rows: List[dict]):
    output_root.mkdir(parents=True, exist_ok=True)

    all_samples_path = output_root / "all_reference_based_per_sample.csv"
    all_summary_path = output_root / "all_reference_based_summary.csv"
    all_report_path = output_root / "all_reference_based_summary.txt"

    if all_rows:
        with open(all_samples_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    if all_summary_rows:
        with open(all_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary_rows)

        lines = [
            "Global reference-based image-space evaluation summary",
            "====================================================",
            "",
        ]

        for row in all_summary_rows:
            if row["direction"] != "combined":
                continue
            lines.extend([
                f"[{row['test_name']}]",
                f"  samples   : {row['num_samples']}",
                f"  threshold : {row['arcface_threshold']}",
                f"  Accuracy  : {row['accuracy']}%",
                f"  ASC       : {row['ASC']}",
                f"  ASA       : {row['ASA']}",
                f"  DCI       : {row['DCI']}",
                f"  DAI       : {row['DAI']}",
                "",
            ])

        lines.extend([
            "Output files:",
            f"  Per-sample CSV : {all_samples_path}",
            f"  Summary CSV    : {all_summary_path}",
        ])

        with open(all_report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print("\n" + "\n".join(lines))


# =========================================================
# 11. Config and CLI
# =========================================================

DEFAULT_TESTS = [
    {
        "name": "cond_non_disjoint",
        "kind": "cond",
        "data_dir": "/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings_v1",
        "run_name": "diffae_conditional_non_disjoint",
        "num_timesteps": 300,
        "mode": "iterative",
    },
    {
        "name": "cond_disjoint",
        "kind": "cond",
        "data_dir": "/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings_v2",
        "run_name": "diffae_conditional_disjointID",
        "num_timesteps": 300,
        "mode": "iterative",
    },
    {
        "name": "cond_refine_disjoint",
        "kind": "cond_refine",
        "data_dir": "/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings_v2",
        "run_name": "diffae_conditional_refiner",
        "num_timesteps": 10,
        "mode": "iterative",
    },
    {
        "name": "baseline_informed",
        "kind": "baseline_informed",
        "diffae_base_path": "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy",
        "run_name": "diffae_baseline",
        "num_timesteps": 300,
    },
]


def load_tests_config(path: Optional[str]) -> List[dict]:
    if path is None:
        return DEFAULT_TESTS
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Image-space evaluation for all reference-based DiffAE de-morphing tests."
    )

    parser.add_argument("--tests-config", type=str, default=None, help="Optional JSON file with the list of tests to evaluate.")
    parser.add_argument("--experiments-root", type=str, default="experiments")
    parser.add_argument("--output-dir", type=str, default="reference_based_image_eval")
    parser.add_argument("--diffae-root", type=str, default="/nas-ctm01/homes/dacordeiro/diffae/")
    parser.add_argument("--diffae-checkpoint", type=str, default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--decode-steps", type=int, default=20)
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--mode", type=str, default="iterative", choices=["iterative", "one_shot"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deepface-detector-backend", type=str, default="skip")
    parser.add_argument("--arcface-threshold", type=str, default="auto", help="Use 'auto' or a fixed numeric ArcFace cosine threshold.")
    parser.add_argument("--threshold-fmr", type=float, default=0.001, help="FMR used when --arcface-threshold auto. Default is 0.001 = 0.1% FMR.")
    parser.add_argument("--threshold-max-pairs", type=int, default=200000)
    parser.add_argument("--threshold-seed", type=int, default=42)
    parser.add_argument("--skip-missing", action="store_true", help="Skip a test if a required checkpoint or data file is missing.")

    args = parser.parse_args()

    device = torch.device(args.device)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading DiffAE FFHQ256 autoencoder...")
    diffae_model = load_diffae_ffhq256_autoencoder(Path(args.diffae_root), Path(args.diffae_checkpoint), device)

    print("Loading DeepFace ArcFace...")
    DeepFace = load_deepface()
    arcface_cache = ArcFaceCache(DeepFace, device, detector_backend=args.deepface_detector_backend)

    fixed_threshold = None
    threshold_cache: Dict[str, float] = {}

    if args.arcface_threshold.lower() != "auto":
        fixed_threshold = float(args.arcface_threshold)

    def threshold_resolver(test_name: str, source_paths: List[str]) -> float:
        if fixed_threshold is not None:
            return fixed_threshold
        if test_name not in threshold_cache:
            print(f"Calibrating ArcFace threshold for {test_name} at FMR={args.threshold_fmr}...")
            threshold_cache[test_name] = calibrate_arcface_threshold(
                source_paths=source_paths,
                arcface_cache=arcface_cache,
                fmr=args.threshold_fmr,
                max_pairs=args.threshold_max_pairs,
                seed=args.threshold_seed,
            )
            print(f"Threshold for {test_name}: {threshold_cache[test_name]:.8f}")
        return threshold_cache[test_name]

    shared = {
        "device": device,
        "output_root": output_root,
        "experiments_root": args.experiments_root,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "decode_steps": args.decode_steps,
        "grid_rows": args.grid_rows,
        "mode": args.mode,
        "diffae_model": diffae_model,
        "arcface_cache": arcface_cache,
        "threshold_resolver": threshold_resolver,
    }

    tests = load_tests_config(args.tests_config)

    all_rows: List[dict] = []
    all_summary_rows: List[dict] = []

    for test_cfg in tests:
        test_name = test_cfg["name"]
        kind = test_cfg["kind"]
        print("\n" + "=" * 80)
        print(f"Evaluating test: {test_name} [{kind}]")
        print("=" * 80)

        try:
            if kind == "cond":
                rows, summary = evaluate_conditional_test(test_cfg, shared)
            elif kind == "cond_refine":
                rows, summary = evaluate_refine_test(test_cfg, shared)
            elif kind == "baseline_informed":
                rows, summary = evaluate_baseline_informed_test(test_cfg, shared)
            else:
                raise ValueError(f"Unknown test kind: {kind}")
        except Exception as exc:
            if args.skip_missing:
                print(f"[SKIPPED] {test_name}: {exc}")
                continue
            raise

        all_rows.extend(rows)
        all_summary_rows.extend(summary)

    write_global_outputs(output_root, all_rows, all_summary_rows)


if __name__ == "__main__":
    main()

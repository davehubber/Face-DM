import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
# 1. Model blocks
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


# =========================================================
# 2. DiffAE loading and image helpers
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
# - ground-truth images loaded from disk are normalized to [-1, 1]
# - DiffAE render() already returns images in [0, 1]
#   so decoded outputs MUST NOT be remapped via (x + 1) / 2

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


# =========================================================
# 3. File and metadata helpers
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


def ensure_eval_morph_xt_v1_only(root: Path) -> np.ndarray:
    """
    Non-disjoint v1 already has encoded morph stochastic codes.
    We only load them here.
    """
    xt_path = optional_existing_path(root, "eval_morph_xt.npy", "eval_morph_xt_1000.npy")
    if xt_path is None:
        raise FileNotFoundError(
            f"No stored eval morph stochastic codes found in {root}. "
            f"For the non-disjoint v1 setup, eval_morph_xt_1000.npy should already exist."
        )
    return np.load(xt_path).astype(np.float32)


def load_morph_eval_data_v1(data_dir: str) -> MorphEvalData:
    root = Path(data_dir).resolve()

    train_bf_raw = np.load(root / "train_bonafide_zsem.npy").astype(np.float32)
    eval_bf_raw = np.load(root / "eval_bonafide_zsem.npy").astype(np.float32)
    eval_morph_raw = np.load(resolve_existing_path(root, "eval_morph_zsem.npy", "eval_morph_zsem_1000.npy")).astype(np.float32)
    eval_morph_xt = ensure_eval_morph_xt_v1_only(root)

    mean = train_bf_raw.mean(axis=0, keepdims=True).astype(np.float32)
    std = (train_bf_raw.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)

    morph_metadata_path = resolve_existing_path(root, "eval_morph_metadata.csv", "eval_morph_metadata_1000.csv")
    morph_rows = read_csv_rows(morph_metadata_path)

    # In v1, eval_bonafide is a symlink to train_bonafide.
    bf_meta_path = root / "train_bonafide_metadata.csv"
    if not bf_meta_path.exists():
        bf_meta_path = root / "eval_bonafide_metadata.csv"

    bf_meta_by_idx = metadata_by_embedding_index(bf_meta_path)

    records = []
    for row in morph_rows:
        idx_m = int(row["embedding_index"])
        col_a = "source_idx_A_in_train_bf" if "source_idx_A_in_train_bf" in row else "source_idx_A_in_eval_bf"
        col_b = "source_idx_B_in_train_bf" if "source_idx_B_in_train_bf" in row else "source_idx_B_in_eval_bf"
        idx_a = int(row[col_a])
        idx_b = int(row[col_b])

        if idx_a < 0 or idx_b < 0:
            continue

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


# =========================================================
# 4. DeepFace ArcFace and metric helpers
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

    embeddings = np.stack(
        [arcface_cache.source_embedding(p) for p in tqdm(unique_paths, desc="ArcFace source embeddings")],
        axis=0,
    )
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
# 5. Shared image-space evaluation engine
# =========================================================

class ImageSpaceEvaluator:
    def __init__(
        self,
        diffae_model,
        arcface_cache: ArcFaceCache,
        device: torch.device,
        decode_steps: int,
        grid_rows: int,
        threshold: float,
    ):
        self.diffae_model = diffae_model
        self.arcface_cache = arcface_cache
        self.device = device
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
        morph_paths: List[str],
        reference_paths: List[str],
        target_paths: List[str],
        rows_out: List[dict],
        buckets: Dict[str, MetricsBucket],
        grid_images: List[torch.Tensor],
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
                "morph_path": morph_paths[i],
                "reference_path": reference_paths[i],
                "target_path": target_paths[i],
                "score_reference": f"{score_ref:.8f}",
                "score_accomplice": f"{score_acc:.8f}",
                "success": int(success),
            })

            buckets[direction].add(score_ref, score_acc, success)
            buckets["combined"].add(score_ref, score_acc, success)

            if len(grid_images) < self.grid_rows * 4:
                morph_img = gt_path_to_tensor(morph_paths[i], self.device, self.transform)
                ref_img = gt_path_to_tensor(reference_paths[i], self.device, self.transform)
                tgt_img = gt_path_to_tensor(target_paths[i], self.device, self.transform)

                grid_images.extend([
                    gt_norm_to_01(morph_img.detach().cpu()),
                    gt_norm_to_01(ref_img.detach().cpu()),
                    gt_norm_to_01(tgt_img.detach().cpu()),
                    decoded_i.detach().cpu(),
                ])


# =========================================================
# 6. Output helpers
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
        "Mode: one_shot",
        f"ArcFace threshold S: {threshold:.8f}",
        "Metrics follow the Accuracy / DCI / DAI protocol:",
        "  success = score_accomplice >= S and score_reference < S",
        "  ASC = mean(score_reference)",
        "  ASA = mean(score_accomplice)",
        "  DCI = ASC - S",
        "  DAI = ASA - S",
        "Decoder range convention: DiffAE render output is treated as already in [0, 1].",
        "Prediction mode: one-shot target prediction at t = T.",
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
# 7. Main evaluation for cond_non_disjoint only
# =========================================================

def evaluate_cond_non_disjoint_one_shot(args):
    device = torch.device(args.device)

    test_name = "cond_non_disjoint"
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    test_dir = output_root / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DiffAE FFHQ256 autoencoder...")
    diffae_model = load_diffae_ffhq256_autoencoder(
        Path(args.diffae_root),
        Path(args.diffae_checkpoint),
        device,
    )

    print("Loading DeepFace ArcFace...")
    DeepFace = load_deepface()
    arcface_cache = ArcFaceCache(DeepFace, device, detector_backend=args.deepface_detector_backend)

    print("Loading non-disjoint morph evaluation data...")
    data = load_morph_eval_data_v1(args.data_dir)

    source_paths = []
    for rec in data.records:
        source_paths.extend([rec["source_a_path"], rec["source_b_path"]])

    if args.arcface_threshold.lower() == "auto":
        print(f"Calibrating ArcFace threshold for {test_name} at FMR={args.threshold_fmr}...")
        threshold = calibrate_arcface_threshold(
            source_paths=source_paths,
            arcface_cache=arcface_cache,
            fmr=args.threshold_fmr,
            max_pairs=args.threshold_max_pairs,
            seed=args.threshold_seed,
        )
        print(f"Threshold for {test_name}: {threshold:.8f}")
    else:
        threshold = float(args.arcface_threshold)
        print(f"Using fixed ArcFace threshold for {test_name}: {threshold:.8f}")

    experiments_root = Path(args.experiments_root).resolve()
    ckpt_path = experiments_root / args.run_name / "checkpoints" / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("Loading conditional model checkpoint...")
    net = ConditionalColdDemorphNet().to(device)
    net.load_state_dict(load_checkpoint_state_dict(ckpt_path))
    net.eval()

    evaluator = ImageSpaceEvaluator(
        diffae_model=diffae_model,
        arcface_cache=arcface_cache,
        device=device,
        decode_steps=args.decode_steps,
        grid_rows=args.grid_rows,
        threshold=threshold,
    )

    mean_t = torch.from_numpy(data.mean).to(device).float()
    std_t = torch.from_numpy(data.std).to(device).float()

    eval_bf_norm = (data.eval_bf_raw - data.mean) / data.std
    eval_morph_norm = (data.eval_morph_raw - data.mean) / data.std

    rows_out: List[dict] = []
    buckets = {"A_to_B": empty_bucket(), "B_to_A": empty_bucket(), "combined": empty_bucket()}
    grid_images: List[torch.Tensor] = []

    num_timesteps = args.num_timesteps

    with torch.no_grad():
        for start in tqdm(range(0, len(data.records), args.batch_size), desc=f"{test_name} predictions"):
            batch_records = data.records[start:start + args.batch_size]
            morph_indices = [r["morph_idx"] for r in batch_records]
            a_indices = [r["source_a_idx"] for r in batch_records]
            b_indices = [r["source_b_idx"] for r in batch_records]

            b_M = torch.tensor(eval_morph_norm[morph_indices], device=device).float()
            b_A = torch.tensor(eval_bf_norm[a_indices], device=device).float()
            b_B = torch.tensor(eval_bf_norm[b_indices], device=device).float()
            b_xt = torch.tensor(data.eval_morph_xt[morph_indices], device=device).float()

            # ONE-SHOT prediction at t = T
            t_max = torch.full((b_M.shape[0],), num_timesteps, device=device, dtype=torch.long)
            pred_B_norm = net(b_M, t_max, b_A)  # A as reference, predict B
            pred_A_norm = net(b_M, t_max, b_B)  # B as reference, predict A

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

    summary_rows = write_outputs(
        test_name=test_name,
        test_dir=test_dir,
        rows_out=rows_out,
        buckets=buckets,
        threshold=threshold,
        grid_images=grid_images,
    )

    # Also write the same global-style summary files at the output root,
    # now containing just this single test.
    write_global_outputs(output_root, rows_out, summary_rows)


# =========================================================
# 8. CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="One-shot image-space evaluation for the non-disjoint conditional DiffAE de-morphing model."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/morph_embeddings_v1",
        help="Path to the non-disjoint morph embedding dataset (v1).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="diffae_conditional_jointID",
        help="Experiment folder name under experiments-root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint filename inside experiments/<run-name>/checkpoints/.",
    )
    parser.add_argument(
        "--experiments-root",
        type=str,
        default="experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reference_based_image_eval",
        help="Same root output folder used by the all-tests script.",
    )
    parser.add_argument(
        "--diffae-root",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/diffae/",
    )
    parser.add_argument(
        "--diffae-checkpoint",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=300,
        help="Training/evaluation timestep count of the conditional model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--deepface-detector-backend",
        type=str,
        default="skip",
    )
    parser.add_argument(
        "--arcface-threshold",
        type=str,
        default="auto",
        help="Use 'auto' or a fixed numeric ArcFace cosine threshold.",
    )
    parser.add_argument(
        "--threshold-fmr",
        type=float,
        default=0.001,
        help="FMR used when --arcface-threshold auto. Default 0.001 = 0.1%% FMR.",
    )
    parser.add_argument(
        "--threshold-max-pairs",
        type=int,
        default=200000,
    )
    parser.add_argument(
        "--threshold-seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    evaluate_cond_non_disjoint_one_shot(args)


if __name__ == "__main__":
    main()

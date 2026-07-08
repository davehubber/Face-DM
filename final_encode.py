#!/usr/bin/env python3
"""
Build DiffAE semantic embedding datasets for MAD22 and SMDD.

Key behavior
------------
- Semantic embeddings (z_sem) are cached once and reused for all split exports.
- Stochastic codes (x_T) are computed/exported only for TEST MORPHS.
- No stochastic codes are stored for train morphs, train sources, test accomplices,
  or test criminals.
- Default stochastic timestep is T=20.
- Normalization statistics are computed only from the TRAIN SOURCE embeddings:
      train_acc_zsem.npy + train_cri_zsem.npy
  not from the morph embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

MAD_MORPH_RE = re.compile(r"^(.+?)-vs-(.+)$", re.IGNORECASE)
SMDD_MORPH_RE = re.compile(r"^morphed_(img\d+)_(img\d+)$", re.IGNORECASE)
SMDD_FALLBACK_RE = re.compile(r"^morphed_(.+?)_(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class MorphRecord:
    morph_path: Path
    morph_filename: str
    morph_stem: str
    acc_stem: str
    cri_stem: str
    acc_id: str
    cri_id: str
    acc_path: Path
    cri_path: Path


@dataclass
class SemanticCache:
    cache_dir: Path
    zsem_path: Path
    metadata_path: Path
    stem_to_index_path: Path
    metadata: List[Dict[str, str]]
    stem_to_index: Dict[str, int]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except Exception:
        return

    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json_dump(obj, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_csv_write(rows: List[Dict], path: Path, fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_np_save(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)
    fsync_dir(path.parent)


def open_atomic_memmap(
    path: Path,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> Tuple[np.memmap, Path]:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")

    if tmp.exists():
        tmp.unlink()

    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype, shape=shape)
    return mm, tmp


def finalize_memmap(mm: np.memmap, tmp_path: Path, final_path: Path) -> None:
    mm.flush()
    del mm
    os.replace(tmp_path, final_path)
    fsync_dir(final_path.parent)


def remove_stale_train_xt_files(out_dir: Path) -> None:
    for name in ["train_morph_xt.npy", "train_acc_xt.npy", "train_cri_xt.npy"]:
        p = out_dir / name
        if p.exists():
            print(f"[cleanup] Removing stale train stochastic file: {p}")
            p.unlink()


def remove_stale_source_test_xt_files(out_dir: Path) -> None:
    for name in ["test_acc_xt.npy", "test_cri_xt.npy"]:
        p = out_dir / name
        if p.exists():
            print(f"[cleanup] Removing stale test source stochastic file: {p}")
            p.unlink()


def remove_stale_global_xt_cache(cache_dir: Path) -> None:
    p = cache_dir / "all_xt.npy"
    if p.exists():
        print(f"[cleanup] Removing stale global stochastic cache: {p}")
        p.unlink()


def remove_stale_extra_norm_files(out_dir: Path) -> None:
    """
    Older versions created several normalization files. The current version only
    creates:
        train_zsem_mean.npy
        train_zsem_std.npy
    computed from train_acc_zsem + train_cri_zsem.
    """
    stale_names = [
        "train_zsem_morph_only_mean.npy",
        "train_zsem_morph_only_std.npy",
        "train_zsem_sources_only_mean.npy",
        "train_zsem_sources_only_std.npy",
    ]

    for name in stale_names:
        p = out_dir / name
        if p.exists():
            print(f"[cleanup] Removing stale normalization file: {p}")
            p.unlink()


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], image_size: int = 256):
        self.paths = list(image_paths)
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return {
            "img": img,
            "index": index,
            "path": str(path),
            "filename": path.name,
            "stem": path.stem,
        }


def load_diffae_ffhq256_autoencoder(
    diffae_root: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    diffae_root = diffae_root.resolve()
    checkpoint_path = checkpoint_path.resolve()

    if not diffae_root.exists():
        raise FileNotFoundError(f"DiffAE repo not found: {diffae_root}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if str(diffae_root) not in sys.path:
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


def discover_images(root_dir: Path) -> List[Path]:
    return sorted(
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_cache_metadata(metadata_path: Path) -> List[Dict[str, str]]:
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def maybe_load_semantic_cache(cache_dir: Path) -> Optional[SemanticCache]:
    zsem_path = cache_dir / "all_zsem.npy"
    metadata_path = cache_dir / "all_metadata.csv"
    stem_to_index_path = cache_dir / "stem_to_index.json"

    if not (
        zsem_path.exists()
        and metadata_path.exists()
        and stem_to_index_path.exists()
    ):
        return None

    metadata = load_cache_metadata(metadata_path)

    with stem_to_index_path.open("r", encoding="utf-8") as f:
        stem_to_index = {k: int(v) for k, v in json.load(f).items()}

    z = np.load(zsem_path, mmap_mode="r")

    if len(metadata) != z.shape[0]:
        print(f"[warning] Semantic cache metadata mismatch in {cache_dir}; recomputing.")
        return None

    return SemanticCache(
        cache_dir=cache_dir,
        zsem_path=zsem_path,
        metadata_path=metadata_path,
        stem_to_index_path=stem_to_index_path,
        metadata=metadata,
        stem_to_index=stem_to_index,
    )


def encode_images_to_semantic_cache(
    image_paths: Sequence[Path],
    cache_dir: Path,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_size: int,
    desc: str,
    force_recompute: bool = False,
) -> SemanticCache:
    ensure_dir(cache_dir)
    remove_stale_global_xt_cache(cache_dir)

    if not force_recompute:
        cache = maybe_load_semantic_cache(cache_dir)
        if cache is not None:
            print(f"[cache hit] {desc}: {cache.zsem_path}")
            return cache

    image_paths = list(image_paths)

    if not image_paths:
        raise ValueError(f"No image paths were provided for cache: {cache_dir}")

    print(f"[encoding] {desc}: {len(image_paths)} images")

    dataset = ImagePathDataset(image_paths, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    zsem_path = cache_dir / "all_zsem.npy"
    metadata_path = cache_dir / "all_metadata.csv"
    stem_to_index_path = cache_dir / "stem_to_index.json"

    metadata_rows: List[Dict[str, str]] = []
    z_mm: Optional[np.memmap] = None
    z_tmp: Optional[Path] = None
    write_pos = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs = batch["img"].to(device, non_blocking=True)

            z_sem = model.encode(imgs)
            z_np = z_sem.detach().float().cpu().numpy().astype(np.float32)

            if z_mm is None:
                z_shape = (len(image_paths),) + tuple(z_np.shape[1:])
                z_mm, z_tmp = open_atomic_memmap(zsem_path, z_shape, z_np.dtype)

            n = z_np.shape[0]
            z_mm[write_pos: write_pos + n] = z_np

            batch_indices = batch["index"].tolist()
            batch_paths = batch["path"]
            batch_filenames = batch["filename"]
            batch_stems = batch["stem"]

            for local_i, idx in enumerate(batch_indices):
                metadata_rows.append({
                    "cache_index": int(idx),
                    "image_path": batch_paths[local_i],
                    "filename": batch_filenames[local_i],
                    "stem": batch_stems[local_i],
                })

            write_pos += n

    assert z_mm is not None and z_tmp is not None
    finalize_memmap(z_mm, z_tmp, zsem_path)

    metadata_rows = sorted(metadata_rows, key=lambda r: int(r["cache_index"]))
    stem_to_index = {
        row["stem"]: int(row["cache_index"])
        for row in metadata_rows
    }

    atomic_csv_write(
        metadata_rows,
        metadata_path,
        fieldnames=["cache_index", "image_path", "filename", "stem"],
    )
    atomic_json_dump(stem_to_index, stem_to_index_path)

    return SemanticCache(
        cache_dir=cache_dir,
        zsem_path=zsem_path,
        metadata_path=metadata_path,
        stem_to_index_path=stem_to_index_path,
        metadata=metadata_rows,
        stem_to_index=stem_to_index,
    )


def encode_stochastic_to_file(
    image_paths: Sequence[Path],
    out_path: Path,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_size: int,
    stochastic_T: int,
    desc: str,
) -> None:
    image_paths = list(image_paths)
    ensure_dir(out_path.parent)

    if not image_paths:
        print(f"[warning] No images provided for stochastic encoding: {out_path}")
        atomic_np_save(out_path, np.empty((0,), dtype=np.float32))
        return

    dataset = ImagePathDataset(image_paths, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    xt_mm: Optional[np.memmap] = None
    xt_tmp: Optional[Path] = None
    write_pos = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs = batch["img"].to(device, non_blocking=True)

            z_sem = model.encode(imgs)
            x_t = model.encode_stochastic(imgs, z_sem, T=stochastic_T)
            x_np = x_t.detach().float().cpu().numpy().astype(np.float32)

            if xt_mm is None:
                xt_shape = (len(image_paths),) + tuple(x_np.shape[1:])
                xt_mm, xt_tmp = open_atomic_memmap(out_path, xt_shape, x_np.dtype)

            n = x_np.shape[0]
            xt_mm[write_pos: write_pos + n] = x_np
            write_pos += n

    assert xt_mm is not None and xt_tmp is not None
    finalize_memmap(xt_mm, xt_tmp, out_path)


def path_lookup_from_images(image_paths: Sequence[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}

    for p in image_paths:
        if p.stem in lookup:
            print(
                f"[warning] duplicate image stem {p.stem}. "
                f"Keeping {lookup[p.stem]}, ignoring {p}"
            )
            continue

        lookup[p.stem] = p

    return lookup


def mad_identity_from_stem(stem: str) -> str:
    return stem.split("_")[0]


def parse_mad_morph(path: Path) -> Optional[Tuple[str, str]]:
    match = MAD_MORPH_RE.match(path.stem)

    if not match:
        return None

    return match.group(1), match.group(2)


def parse_smdd_morph(path: Path) -> Optional[Tuple[str, str]]:
    match = SMDD_MORPH_RE.match(path.stem)

    if match:
        return match.group(1), match.group(2)

    match = SMDD_FALLBACK_RE.match(path.stem)

    if match:
        return match.group(1), match.group(2)

    return None


def build_mad_morph_records(
    method_dir: Path,
    bonafide_lookup: Dict[str, Path],
) -> Tuple[List[MorphRecord], Dict[str, int]]:
    records: List[MorphRecord] = []
    stats = {
        "total_images": 0,
        "valid_morphs": 0,
        "invalid_name": 0,
        "missing_bonafide": 0,
    }

    for p in tqdm(discover_images(method_dir), desc=f"Discover MAD22 {method_dir.name}"):
        stats["total_images"] += 1

        parsed = parse_mad_morph(p)

        if parsed is None:
            stats["invalid_name"] += 1
            continue

        acc_stem, cri_stem = parsed

        if acc_stem not in bonafide_lookup or cri_stem not in bonafide_lookup:
            stats["missing_bonafide"] += 1
            continue

        records.append(MorphRecord(
            morph_path=p,
            morph_filename=p.name,
            morph_stem=p.stem,
            acc_stem=acc_stem,
            cri_stem=cri_stem,
            acc_id=mad_identity_from_stem(acc_stem),
            cri_id=mad_identity_from_stem(cri_stem),
            acc_path=bonafide_lookup[acc_stem],
            cri_path=bonafide_lookup[cri_stem],
        ))
        stats["valid_morphs"] += 1

    records.sort(key=lambda r: str(r.morph_path))
    return records, stats


def build_smdd_morph_records(
    morph_dir: Path,
    bonafide_lookup: Dict[str, Path],
) -> Tuple[List[MorphRecord], Dict[str, int]]:
    records: List[MorphRecord] = []
    stats = {
        "total_images": 0,
        "valid_morphs": 0,
        "invalid_name": 0,
        "missing_bonafide": 0,
    }

    for p in tqdm(discover_images(morph_dir), desc="Discover SMDD morphs"):
        stats["total_images"] += 1

        parsed = parse_smdd_morph(p)

        if parsed is None:
            stats["invalid_name"] += 1
            continue

        acc_stem, cri_stem = parsed

        if acc_stem not in bonafide_lookup or cri_stem not in bonafide_lookup:
            stats["missing_bonafide"] += 1
            continue

        records.append(MorphRecord(
            morph_path=p,
            morph_filename=p.name,
            morph_stem=p.stem,
            acc_stem=acc_stem,
            cri_stem=cri_stem,
            acc_id=acc_stem,
            cri_id=cri_stem,
            acc_path=bonafide_lookup[acc_stem],
            cri_path=bonafide_lookup[cri_stem],
        ))
        stats["valid_morphs"] += 1

    records.sort(key=lambda r: str(r.morph_path))
    return records, stats


def random_morph_split(
    n: int,
    train_pct: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_train = round(n * train_pct / 100.0)

    train = sorted(indices[:n_train].tolist())
    test = sorted(indices[n_train:].tolist())
    excluded: List[int] = []

    return train, test, excluded


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def identity_disjoint_mad_split(
    records: Sequence[MorphRecord],
    split_json: Path,
) -> Tuple[List[int], List[int], List[int], Dict]:
    data = load_json(split_json)

    train_ids = set(str(x) for x in data.get("train_ids", []))
    test_ids = set(str(x) for x in data.get("test_ids", []))

    if not train_ids or not test_ids:
        raise ValueError(
            f"MAD22 split JSON must contain non-empty train_ids and test_ids: {split_json}"
        )

    train: List[int] = []
    test: List[int] = []
    excluded: List[int] = []

    for i, r in enumerate(records):
        if r.acc_id in train_ids and r.cri_id in train_ids:
            train.append(i)
        elif r.acc_id in test_ids and r.cri_id in test_ids:
            test.append(i)
        else:
            excluded.append(i)

    return train, test, excluded, data


def fixed_test_smdd_split(
    records: Sequence[MorphRecord],
    split_json: Path,
) -> Tuple[List[int], List[int], List[int], Dict]:
    data = load_json(split_json)

    test_ids = set(str(x) for x in data.get("test_bonafide_ids", []))
    train_ids = set(str(x) for x in data.get("train_bonafide_ids", []))

    if not test_ids:
        raise ValueError(
            f"SMDD split JSON must contain test_bonafide_ids: {split_json}"
        )

    train: List[int] = []
    test: List[int] = []
    excluded: List[int] = []

    for i, r in enumerate(records):
        acc_test = r.acc_stem in test_ids
        cri_test = r.cri_stem in test_ids

        if acc_test and cri_test:
            test.append(i)

        elif not acc_test and not cri_test:
            if train_ids and not (
                r.acc_stem in train_ids and r.cri_stem in train_ids
            ):
                excluded.append(i)
            else:
                train.append(i)

        else:
            excluded.append(i)

    return train, test, excluded, data


def cache_indices_for_records(
    records: Sequence[MorphRecord],
    record_indices: Sequence[int],
    morph_cache: SemanticCache,
    bf_cache: SemanticCache,
) -> Tuple[List[int], List[int], List[int]]:
    morph_indices: List[int] = []
    acc_indices: List[int] = []
    cri_indices: List[int] = []

    for idx in record_indices:
        r = records[idx]

        morph_indices.append(morph_cache.stem_to_index[r.morph_stem])
        acc_indices.append(bf_cache.stem_to_index[r.acc_stem])
        cri_indices.append(bf_cache.stem_to_index[r.cri_stem])

    return morph_indices, acc_indices, cri_indices


def save_indexed_array(
    src_npy: Path,
    indices: Sequence[int],
    dst_npy: Path,
    chunk_size: int,
    desc: str,
) -> None:
    src = np.load(src_npy, mmap_mode="r")
    indices = list(indices)

    out_shape = (len(indices),) + tuple(src.shape[1:])
    out_mm, tmp = open_atomic_memmap(dst_npy, out_shape, src.dtype)

    for start in tqdm(range(0, len(indices), chunk_size), desc=desc):
        end = min(start + chunk_size, len(indices))

        if start == end:
            continue

        out_mm[start:end] = src[indices[start:end]]

    finalize_memmap(out_mm, tmp, dst_npy)


def split_metadata_rows(
    dataset_name: str,
    method_name: str,
    scenario_name: str,
    split_name: str,
    records: Sequence[MorphRecord],
    record_indices: Sequence[int],
    morph_indices: Sequence[int],
    acc_indices: Sequence[int],
    cri_indices: Sequence[int],
) -> List[Dict]:
    rows: List[Dict] = []

    for out_i, rec_i in enumerate(record_indices):
        r = records[rec_i]

        rows.append({
            "embedding_index": out_i,
            "dataset": dataset_name,
            "method": method_name,
            "scenario": scenario_name,
            "split": split_name,
            "morph_image_path": str(r.morph_path),
            "morph_filename": r.morph_filename,
            "morph_stem": r.morph_stem,
            "accomplice_stem": r.acc_stem,
            "criminal_stem": r.cri_stem,
            "accomplice_id": r.acc_id,
            "criminal_id": r.cri_id,
            "accomplice_image_path": str(r.acc_path),
            "criminal_image_path": str(r.cri_path),
            "morph_cache_index": int(morph_indices[out_i]),
            "accomplice_cache_index": int(acc_indices[out_i]),
            "criminal_cache_index": int(cri_indices[out_i]),
        })

    return rows


def compute_mean_std_from_npy_files(
    npy_paths: Sequence[Path],
    eps: float,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total_count = 0
    total_sum: Optional[np.ndarray] = None
    total_sumsq: Optional[np.ndarray] = None

    for path in npy_paths:
        arr = np.load(path, mmap_mode="r")

        if arr.ndim < 2:
            raise ValueError(
                f"Expected at least 2D z_sem array, got {arr.shape}: {path}"
            )

        for start in tqdm(range(0, arr.shape[0], chunk_size), desc=f"Stats {path.name}"):
            end = min(start + chunk_size, arr.shape[0])
            chunk = np.asarray(arr[start:end], dtype=np.float64)

            if chunk.size == 0:
                continue

            flat = chunk.reshape(chunk.shape[0], -1)

            s = flat.sum(axis=0)
            ss = np.square(flat).sum(axis=0)

            if total_sum is None:
                total_sum = s
                total_sumsq = ss
            else:
                total_sum += s
                total_sumsq += ss

            total_count += flat.shape[0]

    if total_count == 0 or total_sum is None or total_sumsq is None:
        raise ValueError("Cannot compute normalization statistics from zero rows.")

    mean = total_sum / total_count
    var = total_sumsq / total_count - np.square(mean)
    var = np.maximum(var, eps ** 2)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)


def export_norm_files(out_dir: Path, chunk_size: int, eps: float) -> None:
    """
    Create only:
        train_zsem_mean.npy
        train_zsem_std.npy

    These are computed only from the training source embeddings:
        train_acc_zsem.npy + train_cri_zsem.npy

    Morph embeddings are deliberately not included.
    """
    remove_stale_extra_norm_files(out_dir)

    train_acc = out_dir / "train_acc_zsem.npy"
    train_cri = out_dir / "train_cri_zsem.npy"

    mean, std = compute_mean_std_from_npy_files(
        [train_acc, train_cri],
        eps=eps,
        chunk_size=chunk_size,
    )

    atomic_np_save(out_dir / "train_zsem_mean.npy", mean)
    atomic_np_save(out_dir / "train_zsem_std.npy", std)


def export_test_stochastic_arrays(
    out_dir: Path,
    records: Sequence[MorphRecord],
    test_record_indices: Sequence[int],
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_size: int,
    stochastic_T: int,
    dataset_name: str,
    method_name: str,
    scenario_name: str,
) -> None:
    """
    Export stochastic codes only for TEST MORPHS.

    We do NOT store stochastic codes for:
        - train morphs
        - train accomplices
        - train criminals
        - test accomplices
        - test criminals
    """
    remove_stale_source_test_xt_files(out_dir)

    test_records = [records[i] for i in test_record_indices]
    morph_paths = [r.morph_path for r in test_records]

    encode_stochastic_to_file(
        image_paths=morph_paths,
        out_path=out_dir / "test_morph_xt.npy",
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        stochastic_T=stochastic_T,
        desc=(
            f"{dataset_name} {method_name} {scenario_name} "
            f"test morph xT T={stochastic_T}"
        ),
    )


def export_scenario(
    out_dir: Path,
    dataset_name: str,
    method_name: str,
    scenario_name: str,
    records: Sequence[MorphRecord],
    train_record_indices: Sequence[int],
    test_record_indices: Sequence[int],
    excluded_record_indices: Sequence[int],
    morph_cache: SemanticCache,
    bf_cache: SemanticCache,
    morph_discovery_stats: Dict[str, int],
    split_source_summary: Dict,
    encode_stochastic: bool,
    chunk_size: int,
    norm_eps: float,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    image_size: int,
    stochastic_T: int,
) -> None:
    ensure_dir(out_dir)
    remove_stale_train_xt_files(out_dir)
    remove_stale_source_test_xt_files(out_dir)

    train_record_indices = list(train_record_indices)
    test_record_indices = list(test_record_indices)
    excluded_record_indices = list(excluded_record_indices)

    if not train_record_indices:
        raise ValueError(f"Scenario has zero train morphs: {out_dir}")

    if not test_record_indices:
        print(f"[warning] Scenario has zero test morphs: {out_dir}")

    train_morph_idx, train_acc_idx, train_cri_idx = cache_indices_for_records(
        records,
        train_record_indices,
        morph_cache,
        bf_cache,
    )
    test_morph_idx, test_acc_idx, test_cri_idx = cache_indices_for_records(
        records,
        test_record_indices,
        morph_cache,
        bf_cache,
    )

    print("\n" + "-" * 100)
    print(f"Exporting {dataset_name} / {method_name} / {scenario_name}")
    print(f"  train morphs:    {len(train_record_indices)}")
    print(f"  test morphs:     {len(test_record_indices)}")
    print(f"  excluded morphs: {len(excluded_record_indices)}")
    print("-" * 100)

    save_indexed_array(
        morph_cache.zsem_path,
        train_morph_idx,
        out_dir / "train_morph_zsem.npy",
        chunk_size,
        desc=f"{scenario_name} train morph zsem",
    )
    save_indexed_array(
        bf_cache.zsem_path,
        train_acc_idx,
        out_dir / "train_acc_zsem.npy",
        chunk_size,
        desc=f"{scenario_name} train accomplice zsem",
    )
    save_indexed_array(
        bf_cache.zsem_path,
        train_cri_idx,
        out_dir / "train_cri_zsem.npy",
        chunk_size,
        desc=f"{scenario_name} train criminal zsem",
    )

    save_indexed_array(
        morph_cache.zsem_path,
        test_morph_idx,
        out_dir / "test_morph_zsem.npy",
        chunk_size,
        desc=f"{scenario_name} test morph zsem",
    )
    save_indexed_array(
        bf_cache.zsem_path,
        test_acc_idx,
        out_dir / "test_acc_zsem.npy",
        chunk_size,
        desc=f"{scenario_name} test accomplice zsem",
    )
    save_indexed_array(
        bf_cache.zsem_path,
        test_cri_idx,
        out_dir / "test_cri_zsem.npy",
        chunk_size,
        desc=f"{scenario_name} test criminal zsem",
    )

    train_rows = split_metadata_rows(
        dataset_name=dataset_name,
        method_name=method_name,
        scenario_name=scenario_name,
        split_name="train",
        records=records,
        record_indices=train_record_indices,
        morph_indices=train_morph_idx,
        acc_indices=train_acc_idx,
        cri_indices=train_cri_idx,
    )
    test_rows = split_metadata_rows(
        dataset_name=dataset_name,
        method_name=method_name,
        scenario_name=scenario_name,
        split_name="test",
        records=records,
        record_indices=test_record_indices,
        morph_indices=test_morph_idx,
        acc_indices=test_acc_idx,
        cri_indices=test_cri_idx,
    )

    fieldnames = [
        "embedding_index",
        "dataset",
        "method",
        "scenario",
        "split",
        "morph_image_path",
        "morph_filename",
        "morph_stem",
        "accomplice_stem",
        "criminal_stem",
        "accomplice_id",
        "criminal_id",
        "accomplice_image_path",
        "criminal_image_path",
        "morph_cache_index",
        "accomplice_cache_index",
        "criminal_cache_index",
    ]

    atomic_csv_write(train_rows, out_dir / "train_metadata.csv", fieldnames)
    atomic_csv_write(test_rows, out_dir / "test_metadata.csv", fieldnames)

    export_norm_files(out_dir, chunk_size=chunk_size, eps=norm_eps)

    if encode_stochastic:
        export_test_stochastic_arrays(
            out_dir=out_dir,
            records=records,
            test_record_indices=test_record_indices,
            model=model,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            stochastic_T=stochastic_T,
            dataset_name=dataset_name,
            method_name=method_name,
            scenario_name=scenario_name,
        )
    else:
        for name in ["test_morph_xt.npy", "test_acc_xt.npy", "test_cri_xt.npy"]:
            p = out_dir / name
            if p.exists():
                print(
                    "[cleanup] Removing stale stochastic file because "
                    f"--no-stochastic was set: {p}"
                )
                p.unlink()

    split_summary = {
        "dataset": dataset_name,
        "method": method_name,
        "scenario": scenario_name,
        "out_dir": str(out_dir),
        "semantic_only_for_train": True,
        "stochastic_only_for_test_morphs": bool(encode_stochastic),
        "stochastic_T": stochastic_T if encode_stochastic else None,
        "num_records_total_valid": len(records),
        "num_train_morphs": len(train_record_indices),
        "num_test_morphs": len(test_record_indices),
        "num_excluded_morphs": len(excluded_record_indices),
        "morph_discovery_stats": morph_discovery_stats,
        "split_source": split_source_summary,
        "normalization": {
            "files": ["train_zsem_mean.npy", "train_zsem_std.npy"],
            "computed_over": "train_acc_zsem.npy + train_cri_zsem.npy",
            "morph_embeddings_included": False,
        },
    }

    atomic_json_dump(split_summary, out_dir / "split_summary.json")


def process_mad22(args, model, device: torch.device) -> None:
    print("\n" + "=" * 100)
    print("PROCESSING MAD22")
    print("=" * 100)

    mad_root = Path(args.mad22_root)
    mad_out = Path(args.out_root) / "MAD22_embeddings"
    ensure_dir(mad_out)

    bonafide_dir = mad_root / "BonaFide"

    if not bonafide_dir.exists():
        raise FileNotFoundError(f"MAD22 BonaFide directory not found: {bonafide_dir}")

    mad_split_json = Path(args.mad22_identity_json)

    if not mad_split_json.exists():
        raise FileNotFoundError(
            f"MAD22 identity split JSON not found: {mad_split_json}. "
            "Pass --mad22-identity-json with the correct path."
        )

    bf_paths = discover_images(bonafide_dir)
    bf_lookup = path_lookup_from_images(bf_paths)

    print(f"MAD22 bona fide images found: {len(bf_paths)}")

    bf_cache = encode_images_to_semantic_cache(
        image_paths=bf_paths,
        cache_dir=mad_out / "_bonafide_cache",
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        desc="MAD22 bona fide zsem cache",
        force_recompute=args.force_recompute,
    )

    global_summary = {
        "dataset": "MAD22",
        "root": str(mad_root),
        "out_dir": str(mad_out),
        "identity_split_json": str(mad_split_json),
        "methods": [],
    }

    for method_name in args.mad22_methods:
        method_dir = mad_root / method_name

        if not method_dir.exists():
            print(f"[warning] MAD22 method directory not found, skipping: {method_dir}")
            continue

        print("\n" + "=" * 100)
        print(f"MAD22 METHOD: {method_name}")
        print("=" * 100)

        method_out = mad_out / method_name
        ensure_dir(method_out)

        records, stats = build_mad_morph_records(method_dir, bf_lookup)

        if not records:
            print(f"[warning] No valid records for MAD22 method {method_name}, skipping.")
            continue

        morph_cache = encode_images_to_semantic_cache(
            image_paths=[r.morph_path for r in records],
            cache_dir=method_out / "all_morph_encoded",
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            desc=f"MAD22 {method_name} all morph zsem cache",
            force_recompute=args.force_recompute,
        )

        rnd_train, rnd_test, rnd_excluded = random_morph_split(
            n=len(records),
            train_pct=args.random_train_pct,
            seed=args.seed,
        )

        export_scenario(
            out_dir=method_out / "random_80_20",
            dataset_name="MAD22",
            method_name=method_name,
            scenario_name="random_80_20",
            records=records,
            train_record_indices=rnd_train,
            test_record_indices=rnd_test,
            excluded_record_indices=rnd_excluded,
            morph_cache=morph_cache,
            bf_cache=bf_cache,
            morph_discovery_stats=stats,
            split_source_summary={
                "type": "random_morph_level_split_no_identity_restriction",
                "train_pct": args.random_train_pct,
                "seed": args.seed,
            },
            encode_stochastic=not args.no_stochastic,
            chunk_size=args.copy_chunk_size,
            norm_eps=args.norm_eps,
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            stochastic_T=args.stochastic_T,
        )

        id_train, id_test, id_excluded, loaded_split = identity_disjoint_mad_split(
            records,
            mad_split_json,
        )

        export_scenario(
            out_dir=method_out / "identity_disjoint_80_20",
            dataset_name="MAD22",
            method_name=method_name,
            scenario_name="identity_disjoint_80_20",
            records=records,
            train_record_indices=id_train,
            test_record_indices=id_test,
            excluded_record_indices=id_excluded,
            morph_cache=morph_cache,
            bf_cache=bf_cache,
            morph_discovery_stats=stats,
            split_source_summary={
                "type": "identity_disjoint_split_from_json",
                "json_path": str(mad_split_json),
                "json_train_id_count": len(loaded_split.get("train_ids", [])),
                "json_test_id_count": len(loaded_split.get("test_ids", [])),
                "excluded_cross_identity_split_morphs": len(id_excluded),
            },
            encode_stochastic=not args.no_stochastic,
            chunk_size=args.copy_chunk_size,
            norm_eps=args.norm_eps,
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            stochastic_T=args.stochastic_T,
        )

        global_summary["methods"].append({
            "method": method_name,
            "method_dir": str(method_dir),
            "method_out": str(method_out),
            "valid_records": len(records),
            "discovery_stats": stats,
            "random_80_20": {
                "train": len(rnd_train),
                "test": len(rnd_test),
                "excluded": len(rnd_excluded),
            },
            "identity_disjoint_80_20": {
                "train": len(id_train),
                "test": len(id_test),
                "excluded": len(id_excluded),
            },
        })

    atomic_json_dump(global_summary, mad_out / "MAD22_processing_summary.json")
    atomic_json_dump({"status": "complete"}, mad_out / "_MAD22_DONE.json")

    print("\n" + "=" * 100)
    print("MAD22 PROCESSING COMPLETE. All MAD22 outputs have been written before SMDD starts.")
    print("=" * 100)


def process_smdd(args, model, device: torch.device) -> None:
    print("\n" + "=" * 100)
    print("PROCESSING SMDD")
    print("=" * 100)

    smdd_root = Path(args.smdd_root)
    smdd_out = Path(args.out_root) / "SMDD_embeddings"
    ensure_dir(smdd_out)

    morph_dir = Path(args.smdd_morph_dir) if args.smdd_morph_dir else smdd_root / "m15k_t"
    bonafide_dir = (
        Path(args.smdd_bonafide_dir)
        if args.smdd_bonafide_dir
        else smdd_root / "os25k_m_t"
    )
    smdd_split_json = Path(args.smdd_split_json)

    if not morph_dir.exists():
        raise FileNotFoundError(f"SMDD morph directory not found: {morph_dir}")

    if not bonafide_dir.exists():
        raise FileNotFoundError(f"SMDD bona fide directory not found: {bonafide_dir}")

    if not smdd_split_json.exists():
        raise FileNotFoundError(
            f"SMDD split JSON not found: {smdd_split_json}. "
            "Pass --smdd-split-json with the correct path."
        )

    bf_paths = discover_images(bonafide_dir)
    bf_lookup = path_lookup_from_images(bf_paths)

    print(f"SMDD bona fide images found: {len(bf_paths)}")

    bf_cache = encode_images_to_semantic_cache(
        image_paths=bf_paths,
        cache_dir=smdd_out / "_bonafide_cache",
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        desc="SMDD bona fide zsem cache",
        force_recompute=args.force_recompute,
    )

    method_name = "SMDD"
    method_out = smdd_out / method_name
    ensure_dir(method_out)

    records, stats = build_smdd_morph_records(morph_dir, bf_lookup)

    if not records:
        raise RuntimeError("No valid SMDD morph records found.")

    morph_cache = encode_images_to_semantic_cache(
        image_paths=[r.morph_path for r in records],
        cache_dir=method_out / "all_morph_encoded",
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        desc="SMDD all morph zsem cache",
        force_recompute=args.force_recompute,
    )

    train_idx, test_idx, excluded_idx, loaded_split = fixed_test_smdd_split(
        records,
        smdd_split_json,
    )

    scenario_name = "identity_disjoint_fixed_1000"

    export_scenario(
        out_dir=method_out / scenario_name,
        dataset_name="SMDD",
        method_name=method_name,
        scenario_name=scenario_name,
        records=records,
        train_record_indices=train_idx,
        test_record_indices=test_idx,
        excluded_record_indices=excluded_idx,
        morph_cache=morph_cache,
        bf_cache=bf_cache,
        morph_discovery_stats=stats,
        split_source_summary={
            "type": "fixed_test_source_image_split_from_json",
            "json_path": str(smdd_split_json),
            "target_test_morphs": loaded_split.get("target_test_morphs"),
            "exact_target_satisfied": loaded_split.get("exact_target_satisfied"),
            "json_train_bonafide_count": len(loaded_split.get("train_bonafide_ids", [])),
            "json_test_bonafide_count": len(loaded_split.get("test_bonafide_ids", [])),
            "excluded_cross_source_split_morphs": len(excluded_idx),
        },
        encode_stochastic=not args.no_stochastic,
        chunk_size=args.copy_chunk_size,
        norm_eps=args.norm_eps,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        stochastic_T=args.stochastic_T,
    )

    summary = {
        "dataset": "SMDD",
        "root": str(smdd_root),
        "morph_dir": str(morph_dir),
        "bonafide_dir": str(bonafide_dir),
        "out_dir": str(smdd_out),
        "split_json": str(smdd_split_json),
        "valid_records": len(records),
        "discovery_stats": stats,
        scenario_name: {
            "train": len(train_idx),
            "test": len(test_idx),
            "excluded": len(excluded_idx),
        },
    }

    atomic_json_dump(summary, smdd_out / "SMDD_processing_summary.json")
    atomic_json_dump({"status": "complete"}, smdd_out / "_SMDD_DONE.json")

    print("\n" + "=" * 100)
    print("SMDD PROCESSING COMPLETE.")
    print("=" * 100)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Encode MAD22 and SMDD into DiffAE semantic embedding datasets. "
            "Stochastic x_T is exported only for test morphs. "
            "Normalization is computed only from train source embeddings."
        )
    )

    p.add_argument(
        "--diffae-root",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/diffae/",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc/last.ckpt",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    p.add_argument(
        "--mad22-root",
        type=str,
        default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/MAD22/original_sorted",
    )
    p.add_argument(
        "--smdd-root",
        type=str,
        default="/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD",
    )
    p.add_argument("--smdd-morph-dir", type=str, default=None)
    p.add_argument("--smdd-bonafide-dir", type=str, default=None)

    p.add_argument(
        "--mad22-identity-json",
        type=str,
        default="mad22_optimal_identity_split_80_seed42.json",
    )
    p.add_argument(
        "--smdd-split-json",
        type=str,
        default="smdd_fixed_1000_test_seed42.json",
    )

    p.add_argument(
        "--out-root",
        type=str,
        default=".",
        help="Will create MAD22_embeddings and SMDD_embeddings inside this folder.",
    )

    p.add_argument(
        "--mad22-methods",
        nargs="*",
        default=["FaceMorpher", "MIPGAN_I", "MIPGAN_II", "OpenCV", "Webmorph"],
        help="MAD22 method folders to process.",
    )

    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument(
        "--stochastic-T",
        type=int,
        default=20,
        help="T used for test morph stochastic encoding. Default: 20.",
    )
    p.add_argument(
        "--no-stochastic",
        action="store_true",
        help="Only store z_sem arrays; skip test morph stochastic x_T arrays.",
    )

    p.add_argument("--random-train-pct", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-chunk-size", type=int, default=2048)
    p.add_argument("--norm-eps", type=float, default=1e-6)

    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore existing semantic caches and recompute z_sem.",
    )
    p.add_argument("--skip-mad22", action="store_true")
    p.add_argument("--skip-smdd", action="store_true")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    device = torch.device(args.device)

    print("\nInitializing DiffAE model")
    print("=" * 100)
    print(f"DiffAE root: {args.diffae_root}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Device:      {device}")
    print(f"Stochastic T for test morphs: {args.stochastic_T}")
    print("Normalization: train sources only, i.e. train_acc_zsem + train_cri_zsem")

    model = load_diffae_ffhq256_autoencoder(
        diffae_root=Path(args.diffae_root),
        checkpoint_path=Path(args.checkpoint),
        device=device,
    )

    print("Model loaded successfully.")

    if not args.skip_mad22:
        process_mad22(args, model, device)
    else:
        print("Skipping MAD22 because --skip-mad22 was set.")

    if not args.skip_smdd:
        process_smdd(args, model, device)
    else:
        print("Skipping SMDD because --skip-smdd was set.")

    print("\nAll requested processing is complete.")


if __name__ == "__main__":
    main()

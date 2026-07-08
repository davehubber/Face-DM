#!/usr/bin/env python3
"""
Fix normalization files for MAD22_embeddings and SMDD_embeddings.

This script replaces, in each split folder:

    train_zsem_mean.npy
    train_zsem_std.npy

with statistics computed from the unique base bona fide training set, instead of
from repeated train_acc_zsem/train_cri_zsem rows.

It does NOT re-encode images.
It does NOT modify train/test morph/source embedding arrays.
It only overwrites the two normalization files.

Expected structure:

    <embeddings-root>/MAD22_embeddings/
        _bonafide_cache/
            all_zsem.npy
            all_metadata.csv
            stem_to_index.json
        FaceMorpher/
            random_80_20/
            identity_disjoint_80_20/
        MIPGAN_I/
        MIPGAN_II/
        OpenCV/
        Webmorph/

    <embeddings-root>/SMDD_embeddings/
        _bonafide_cache/
            all_zsem.npy
            all_metadata.csv
            stem_to_index.json
        SMDD/
            identity_disjoint_fixed_1000/

Default split JSON names:
    mad22_optimal_identity_split_80_seed42.json
    smdd_fixed_1000_test_seed42.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm


MAD22_METHODS_DEFAULT = ["FaceMorpher", "MIPGAN_I", "MIPGAN_II", "OpenCV", "Webmorph"]
MAD22_SCENARIOS_DEFAULT = ["random_80_20", "identity_disjoint_80_20"]
SMDD_SCENARIO_DEFAULT = "identity_disjoint_fixed_1000"


def ensure_exists(path: Path, kind: str = "path") -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind}: {path}")


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except Exception:
        return

    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_np_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)
    fsync_dir(path.parent)


def load_json(path: Path) -> Dict:
    ensure_exists(path, "JSON file")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    ensure_exists(path, "CSV file")

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mad_identity_from_stem(stem: str) -> str:
    """
    MAD22 examples:
        009_08 -> 009
        144_03 -> 144
    """
    return stem.split("_")[0]


def load_stem_to_index(cache_dir: Path) -> Dict[str, int]:
    path = cache_dir / "stem_to_index.json"
    ensure_exists(path, "stem_to_index.json")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return {str(k): int(v) for k, v in raw.items()}


def load_cache_metadata(cache_dir: Path) -> List[Dict[str, str]]:
    path = cache_dir / "all_metadata.csv"
    return read_csv_rows(path)


def selected_indices_from_stems(
    stems: Sequence[str],
    stem_to_index: Dict[str, int],
    label: str,
) -> List[int]:
    unique_stems = sorted(set(str(s) for s in stems))
    missing = [s for s in unique_stems if s not in stem_to_index]

    if missing:
        preview = missing[:20]
        raise RuntimeError(
            f"{label}: {len(missing)} selected stems were not found in the bona fide cache. "
            f"First missing stems: {preview}"
        )

    return sorted(stem_to_index[s] for s in unique_stems)


def compute_mean_std_from_cache_indices(
    cache_zsem_path: Path,
    selected_indices: Sequence[int],
    chunk_size: int,
    eps: float,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    ensure_exists(cache_zsem_path, "cache z_sem file")

    selected_indices = sorted(set(int(i) for i in selected_indices))

    if not selected_indices:
        raise ValueError(f"{label}: no selected bona fide indices for statistics.")

    zsem = np.load(cache_zsem_path, mmap_mode="r")

    if zsem.ndim != 2:
        raise ValueError(
            f"{label}: expected z_sem cache to have shape (N, D), got {zsem.shape}."
        )

    n_total = zsem.shape[0]

    bad = [i for i in selected_indices if i < 0 or i >= n_total]
    if bad:
        raise IndexError(
            f"{label}: selected indices outside cache range 0..{n_total - 1}. "
            f"First bad indices: {bad[:20]}"
        )

    total_count = 0
    total_sum = np.zeros((zsem.shape[1],), dtype=np.float64)
    total_sumsq = np.zeros((zsem.shape[1],), dtype=np.float64)

    for start in tqdm(
        range(0, len(selected_indices), chunk_size),
        desc=f"Stats {label}",
    ):
        end = min(start + chunk_size, len(selected_indices))
        idx = np.array(selected_indices[start:end], dtype=np.int64)

        chunk = np.asarray(zsem[idx], dtype=np.float64)

        total_sum += chunk.sum(axis=0)
        total_sumsq += np.square(chunk).sum(axis=0)
        total_count += chunk.shape[0]

    mean = total_sum / total_count
    var = total_sumsq / total_count - np.square(mean)
    var = np.maximum(var, eps ** 2)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)


def backup_existing_norm_files(split_dir: Path) -> None:
    for name in ["train_zsem_mean.npy", "train_zsem_std.npy"]:
        src = split_dir / name

        if not src.exists():
            continue

        dst = split_dir / f"{name}.backup_before_unique_bonafide_fix"

        if dst.exists():
            continue

        shutil.copy2(src, dst)


def write_norm_files(
    split_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    selected_stems: Sequence[str],
    selected_indices: Sequence[int],
    source_description: Dict,
    dry_run: bool,
    backup: bool,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[dry-run] Would overwrite: {split_dir / 'train_zsem_mean.npy'}")
        print(f"[dry-run] Would overwrite: {split_dir / 'train_zsem_std.npy'}")
        return

    if backup:
        backup_existing_norm_files(split_dir)

    atomic_np_save(split_dir / "train_zsem_mean.npy", mean)
    atomic_np_save(split_dir / "train_zsem_std.npy", std)

    summary = {
        "normalization_files": [
            "train_zsem_mean.npy",
            "train_zsem_std.npy",
        ],
        "computed_from": source_description,
        "num_unique_bonafide_stems": len(set(selected_stems)),
        "num_unique_cache_indices": len(set(selected_indices)),
        "mean_shape": list(mean.shape),
        "std_shape": list(std.shape),
        "note": (
            "These statistics were recomputed from the unique base bona fide "
            "training set. Repeated train morph source occurrences were not used."
        ),
    }

    atomic_json_dump(summary, split_dir / "normalization_source_summary.json")

    print(f"[updated] {split_dir}")
    print(f"          unique bona fide embeddings used: {len(set(selected_indices))}")


def mad22_identity_disjoint_train_stems(
    mad22_cache_metadata: List[Dict[str, str]],
    mad22_split_json: Path,
) -> Tuple[List[str], Dict]:
    data = load_json(mad22_split_json)

    train_ids = set(str(x) for x in data.get("train_ids", []))

    if not train_ids:
        raise ValueError(
            f"MAD22 split JSON does not contain non-empty train_ids: {mad22_split_json}"
        )

    stems = []

    for row in mad22_cache_metadata:
        stem = row["stem"]
        identity = mad_identity_from_stem(stem)

        if identity in train_ids:
            stems.append(stem)

    source = {
        "dataset": "MAD22",
        "scenario": "identity_disjoint_80_20",
        "policy": "all_bonafide_images_whose_identity_is_in_train_ids",
        "split_json": str(mad22_split_json),
        "num_train_ids": len(train_ids),
    }

    return stems, source


def mad22_random_train_stems_from_metadata(
    split_dir: Path,
    mad22_cache_metadata: List[Dict[str, str]],
) -> Tuple[List[str], Dict]:
    """
    For random_80_20, there is no identity-disjoint training identity set.

    The default policy here is:
        1. Read train_metadata.csv.
        2. Collect the unique identities that appear as accomplice/criminal
           sources in the training morphs.
        3. Use all bona fide images in the MAD22 cache whose identity belongs
           to those source identities.

    This avoids repeated influence from the same source image, and it uses the
    base bona fide identity pool implied by the random training split.
    """
    train_metadata = split_dir / "train_metadata.csv"
    rows = read_csv_rows(train_metadata)

    train_source_identities: Set[str] = set()

    for row in rows:
        acc_id = row.get("accomplice_id", "")
        cri_id = row.get("criminal_id", "")

        if acc_id:
            train_source_identities.add(str(acc_id))

        if cri_id:
            train_source_identities.add(str(cri_id))

    if not train_source_identities:
        raise ValueError(f"No train source identities found in {train_metadata}")

    stems = []

    for row in mad22_cache_metadata:
        stem = row["stem"]
        identity = mad_identity_from_stem(stem)

        if identity in train_source_identities:
            stems.append(stem)

    source = {
        "dataset": "MAD22",
        "scenario": "random_80_20",
        "policy": (
            "all_bonafide_images_whose_identity_appears_as_train_accomplice_or_"
            "criminal_in_train_metadata"
        ),
        "train_metadata": str(train_metadata),
        "num_train_source_identities": len(train_source_identities),
    }

    return stems, source


def smdd_train_stems_from_split_json(smdd_split_json: Path) -> Tuple[List[str], Dict]:
    data = load_json(smdd_split_json)

    train_ids = [str(x) for x in data.get("train_bonafide_ids", [])]

    if not train_ids:
        raise ValueError(
            f"SMDD split JSON does not contain non-empty train_bonafide_ids: {smdd_split_json}"
        )

    source = {
        "dataset": "SMDD",
        "scenario": "identity_disjoint_fixed_1000",
        "policy": "all_bonafide_images_listed_in_train_bonafide_ids",
        "split_json": str(smdd_split_json),
        "num_train_bonafide_ids": len(set(train_ids)),
        "target_test_morphs": data.get("target_test_morphs"),
        "exact_target_satisfied": data.get("exact_target_satisfied"),
    }

    return train_ids, source


def update_one_split(
    split_dir: Path,
    cache_dir: Path,
    selected_stems: Sequence[str],
    source_description: Dict,
    chunk_size: int,
    eps: float,
    dry_run: bool,
    backup: bool,
) -> None:
    ensure_exists(split_dir, "split directory")
    ensure_exists(cache_dir, "bona fide cache directory")

    stem_to_index = load_stem_to_index(cache_dir)
    selected_indices = selected_indices_from_stems(
        stems=selected_stems,
        stem_to_index=stem_to_index,
        label=str(split_dir),
    )

    cache_zsem_path = cache_dir / "all_zsem.npy"

    mean, std = compute_mean_std_from_cache_indices(
        cache_zsem_path=cache_zsem_path,
        selected_indices=selected_indices,
        chunk_size=chunk_size,
        eps=eps,
        label=split_dir.name,
    )

    write_norm_files(
        split_dir=split_dir,
        mean=mean,
        std=std,
        selected_stems=selected_stems,
        selected_indices=selected_indices,
        source_description=source_description,
        dry_run=dry_run,
        backup=backup,
    )


def update_mad22(
    embeddings_root: Path,
    mad22_split_json: Path,
    methods: Sequence[str],
    scenarios: Sequence[str],
    chunk_size: int,
    eps: float,
    dry_run: bool,
    backup: bool,
) -> None:
    mad22_root = embeddings_root / "MAD22_embeddings"
    mad22_cache_dir = mad22_root / "_bonafide_cache"

    ensure_exists(mad22_root, "MAD22_embeddings directory")
    ensure_exists(mad22_cache_dir, "MAD22 bona fide cache directory")

    mad22_cache_metadata = load_cache_metadata(mad22_cache_dir)

    print("\n" + "=" * 100)
    print("Updating MAD22 normalization files")
    print("=" * 100)

    for method in methods:
        method_dir = mad22_root / method

        if not method_dir.exists():
            print(f"[warning] MAD22 method folder not found, skipping: {method_dir}")
            continue

        for scenario in scenarios:
            split_dir = method_dir / scenario

            if not split_dir.exists():
                print(f"[warning] MAD22 split folder not found, skipping: {split_dir}")
                continue

            if scenario == "identity_disjoint_80_20":
                selected_stems, source = mad22_identity_disjoint_train_stems(
                    mad22_cache_metadata=mad22_cache_metadata,
                    mad22_split_json=mad22_split_json,
                )

            elif scenario == "random_80_20":
                selected_stems, source = mad22_random_train_stems_from_metadata(
                    split_dir=split_dir,
                    mad22_cache_metadata=mad22_cache_metadata,
                )

            else:
                print(f"[warning] Unknown MAD22 scenario, skipping: {split_dir}")
                continue

            print(f"\nMAD22 / {method} / {scenario}")
            print(f"Selected unique base bona fide stems: {len(set(selected_stems))}")

            update_one_split(
                split_dir=split_dir,
                cache_dir=mad22_cache_dir,
                selected_stems=selected_stems,
                source_description=source,
                chunk_size=chunk_size,
                eps=eps,
                dry_run=dry_run,
                backup=backup,
            )


def update_smdd(
    embeddings_root: Path,
    smdd_split_json: Path,
    chunk_size: int,
    eps: float,
    dry_run: bool,
    backup: bool,
) -> None:
    smdd_root = embeddings_root / "SMDD_embeddings"
    smdd_cache_dir = smdd_root / "_bonafide_cache"
    split_dir = smdd_root / "SMDD" / SMDD_SCENARIO_DEFAULT

    ensure_exists(smdd_root, "SMDD_embeddings directory")
    ensure_exists(smdd_cache_dir, "SMDD bona fide cache directory")
    ensure_exists(split_dir, "SMDD split directory")

    selected_stems, source = smdd_train_stems_from_split_json(smdd_split_json)

    print("\n" + "=" * 100)
    print("Updating SMDD normalization files")
    print("=" * 100)
    print(f"SMDD / SMDD / {SMDD_SCENARIO_DEFAULT}")
    print(f"Selected unique base bona fide stems: {len(set(selected_stems))}")

    update_one_split(
        split_dir=split_dir,
        cache_dir=smdd_cache_dir,
        selected_stems=selected_stems,
        source_description=source,
        chunk_size=chunk_size,
        eps=eps,
        dry_run=dry_run,
        backup=backup,
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Replace train_zsem_mean.npy and train_zsem_std.npy in MAD22/SMDD "
            "embedding split folders using the unique base bona fide training set."
        )
    )

    p.add_argument(
        "--embeddings-root",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM",
        help=(
            "Folder containing MAD22_embeddings and SMDD_embeddings. "
            "Default: /nas-ctm01/homes/dacordeiro/Face-DM"
        ),
    )

    p.add_argument(
        "--mad22-split-json",
        type=str,
        default="mad22_optimal_identity_split_80_seed42.json",
        help="Path to the MAD22 identity split JSON.",
    )

    p.add_argument(
        "--smdd-split-json",
        type=str,
        default="smdd_fixed_1000_test_seed42.json",
        help="Path to the SMDD fixed-test split JSON.",
    )

    p.add_argument(
        "--mad22-methods",
        nargs="*",
        default=MAD22_METHODS_DEFAULT,
        help="MAD22 method folders to update.",
    )

    p.add_argument(
        "--mad22-scenarios",
        nargs="*",
        default=MAD22_SCENARIOS_DEFAULT,
        help="MAD22 scenarios to update.",
    )

    p.add_argument(
        "--skip-mad22",
        action="store_true",
        help="Do not update MAD22 normalization files.",
    )

    p.add_argument(
        "--skip-smdd",
        action="store_true",
        help="Do not update SMDD normalization files.",
    )

    p.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Number of bona fide embeddings processed at a time.",
    )

    p.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Minimum std floor.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report what would be updated, but do not overwrite files.",
    )

    p.add_argument(
        "--no-backup",
        action="store_true",
        help=(
            "Do not create backup copies of existing train_zsem_mean/std files. "
            "By default, backups are created once."
        ),
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()

    embeddings_root = Path(args.embeddings_root)
    mad22_split_json = Path(args.mad22_split_json)
    smdd_split_json = Path(args.smdd_split_json)

    backup = not args.no_backup

    print("\nNormalization correction script")
    print("=" * 100)
    print(f"Embeddings root: {embeddings_root}")
    print(f"MAD22 split JSON: {mad22_split_json}")
    print(f"SMDD split JSON:  {smdd_split_json}")
    print(f"Dry run: {args.dry_run}")
    print(f"Backup old files: {backup}")

    if not args.skip_mad22:
        update_mad22(
            embeddings_root=embeddings_root,
            mad22_split_json=mad22_split_json,
            methods=args.mad22_methods,
            scenarios=args.mad22_scenarios,
            chunk_size=args.chunk_size,
            eps=args.eps,
            dry_run=args.dry_run,
            backup=backup,
        )
    else:
        print("\nSkipping MAD22.")

    if not args.skip_smdd:
        update_smdd(
            embeddings_root=embeddings_root,
            smdd_split_json=smdd_split_json,
            chunk_size=args.chunk_size,
            eps=args.eps,
            dry_run=args.dry_run,
            backup=backup,
        )
    else:
        print("\nSkipping SMDD.")

    print("\nDone.")


if __name__ == "__main__":
    main()

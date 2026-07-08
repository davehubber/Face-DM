#!/usr/bin/env python3
"""
Find an identity-level MAD22 split that minimizes excluded morphs.

A morph such as:
    009_08-vs-144_08.jpg

is interpreted as an edge between identities:
    009 and 144

Given a requested percentage of identities reserved for training:
    - train morph: both identities are in train
    - test morph: both identities are in test
    - excluded morph: one identity is train and the other is test

The optimization objective is:

    minimize total excluded morphs across the selected morphing methods

This is a fixed-size graph partitioning problem. The script uses repeated
simulated annealing + local swap search, which is appropriate when the exact
combinatorial search would be too large.

Example:
python optimize_mad22_identity_split.py \
    --root /nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/MAD22/original_sorted \
    --train-id-pct 80 \
    --seed 42 \
    --num-restarts 200 \
    --anneal-steps 20000 \
    --save-json mad22_optimal_identity_split_80_seed42.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

MORPH_RE = re.compile(r"^(.+?)-vs-(.+)$", re.IGNORECASE)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def identity_from_bonafide_name(path: Path) -> str:
    """
    Example:
        009_08.jpg -> 009
        144_03.jpg -> 144
    """
    return path.stem.split("_")[0]


def parse_morph_identities(path: Path) -> Optional[Tuple[str, str]]:
    """
    Example:
        009_08-vs-144_08.jpg -> ("009", "144")
    """
    match = MORPH_RE.match(path.stem)
    if not match:
        return None

    left_name, right_name = match.group(1), match.group(2)

    left_id = left_name.split("_")[0]
    right_id = right_name.split("_")[0]

    if not left_id or not right_id:
        return None

    return left_id, right_id


def collect_bonafide_identities(
    bonafide_dir: Path,
) -> Tuple[Set[str], Dict[str, List[Path]]]:
    identity_to_files: Dict[str, List[Path]] = defaultdict(list)

    for path in sorted(bonafide_dir.rglob("*")):
        if not is_image(path):
            continue

        identity = identity_from_bonafide_name(path)
        identity_to_files[identity].append(path)

    return set(identity_to_files.keys()), identity_to_files


def resolve_method_dirs(root: Path, methods: Optional[List[str]]) -> List[Path]:
    if methods is None:
        return [
            p for p in sorted(root.iterdir())
            if p.is_dir() and p.name != "BonaFide" and not p.name.startswith(".")
        ]

    return [root / method for method in methods]


def collect_weighted_morph_graph(
    method_dirs: List[Path],
    all_bonafide_ids: Set[str],
) -> Tuple[Counter[Tuple[str, str]], Dict[str, Dict[str, int]]]:
    """
    Builds a weighted graph over identities.

    Each valid morph contributes weight 1 to the edge between its two identities.
    If the same identity pair appears in several methods, the weight accumulates.

    Returns:
        pair_weights:
            Counter mapping (id_a, id_b) -> number of morphs using this pair.
        parsing_stats_by_method:
            Per-method counts for valid/invalid/unknown names.
    """
    pair_weights: Counter[Tuple[str, str]] = Counter()
    parsing_stats_by_method: Dict[str, Dict[str, int]] = {}

    for method_dir in method_dirs:
        stats = {
            "total_images": 0,
            "valid_morphs": 0,
            "invalid_name": 0,
            "unknown_identity": 0,
        }

        if not method_dir.exists():
            print(f"[WARNING] Method folder does not exist, skipping: {method_dir}")
            parsing_stats_by_method[method_dir.name] = stats
            continue

        for path in sorted(method_dir.rglob("*")):
            if not is_image(path):
                continue

            stats["total_images"] += 1

            parsed = parse_morph_identities(path)
            if parsed is None:
                stats["invalid_name"] += 1
                continue

            id_a, id_b = parsed

            if id_a not in all_bonafide_ids or id_b not in all_bonafide_ids:
                stats["unknown_identity"] += 1
                continue

            edge = tuple(sorted((id_a, id_b)))
            pair_weights[edge] += 1
            stats["valid_morphs"] += 1

        parsing_stats_by_method[method_dir.name] = stats

    return pair_weights, parsing_stats_by_method


def build_adjacency(
    pair_weights: Counter[Tuple[str, str]],
) -> Dict[str, Dict[str, int]]:
    adj: Dict[str, Dict[str, int]] = defaultdict(dict)

    for (u, v), w in pair_weights.items():
        if u == v:
            continue

        adj[u][v] = adj[u].get(v, 0) + w
        adj[v][u] = adj[v].get(u, 0) + w

    return adj


def compute_cut_weight(
    train_ids: Set[str],
    pair_weights: Counter[Tuple[str, str]],
) -> int:
    cut = 0

    for (u, v), w in pair_weights.items():
        if (u in train_ids) != (v in train_ids):
            cut += w

    return cut


def swap_delta(
    train_ids: Set[str],
    a_train: str,
    b_test: str,
    adj: Dict[str, Dict[str, int]],
) -> int:
    """
    Computes the change in cut weight if:
        a_train moves from train to test
        b_test moves from test to train

    Negative delta means the swap improves the split.
    """
    affected_edges = {}

    for u in (a_train, b_test):
        for v, w in adj.get(u, {}).items():
            edge = tuple(sorted((u, v)))
            affected_edges[edge] = w

    before = 0
    after = 0

    for (u, v), w in affected_edges.items():
        u_before = u in train_ids
        v_before = v in train_ids

        u_after = u_before
        v_after = v_before

        if u == a_train:
            u_after = False
        elif u == b_test:
            u_after = True

        if v == a_train:
            v_after = False
        elif v == b_test:
            v_after = True

        if u_before != v_before:
            before += w

        if u_after != v_after:
            after += w

    return after - before


def exact_optimize_split(
    identities: List[str],
    n_train: int,
    pair_weights: Counter[Tuple[str, str]],
) -> Tuple[Set[str], int]:
    """
    Exact combinatorial optimization.

    Only feasible for small numbers of identities.
    """
    best_train_ids: Optional[Set[str]] = None
    best_cut: Optional[int] = None

    for combo in combinations(identities, n_train):
        train_ids = set(combo)
        cut = compute_cut_weight(train_ids, pair_weights)

        if best_cut is None or cut < best_cut:
            best_cut = cut
            best_train_ids = train_ids

    assert best_train_ids is not None
    assert best_cut is not None

    return best_train_ids, best_cut


def heuristic_optimize_split(
    identities: List[str],
    n_train: int,
    pair_weights: Counter[Tuple[str, str]],
    seed: int,
    num_restarts: int,
    anneal_steps: int,
    local_sweeps: int,
) -> Tuple[Set[str], int]:
    """
    Repeated simulated annealing followed by greedy local swap search.

    The split size is fixed throughout.
    """
    rng = random.Random(seed)
    adj = build_adjacency(pair_weights)

    if n_train == 0:
        return set(), 0

    if n_train == len(identities):
        return set(identities), 0

    total_weight = sum(pair_weights.values())
    start_temp = max(1.0, total_weight / max(1, len(identities)))
    end_temp = 0.01

    best_global_train: Optional[Set[str]] = None
    best_global_cut: Optional[int] = None

    for restart_idx in range(num_restarts):
        train_list = rng.sample(identities, n_train)
        train_ids = set(train_list)

        test_list = [identity for identity in identities if identity not in train_ids]

        current_cut = compute_cut_weight(train_ids, pair_weights)

        best_restart_train = set(train_ids)
        best_restart_cut = current_cut

        for step in range(anneal_steps):
            if anneal_steps <= 1:
                temp = end_temp
            else:
                frac = step / (anneal_steps - 1)
                temp = start_temp * ((end_temp / start_temp) ** frac)

            train_idx = rng.randrange(len(train_list))
            test_idx = rng.randrange(len(test_list))

            a_train = train_list[train_idx]
            b_test = test_list[test_idx]

            delta = swap_delta(train_ids, a_train, b_test, adj)

            accept = False
            if delta <= 0:
                accept = True
            else:
                prob = math.exp(-delta / max(temp, 1e-12))
                accept = rng.random() < prob

            if accept:
                train_ids.remove(a_train)
                train_ids.add(b_test)

                train_list[train_idx] = b_test
                test_list[test_idx] = a_train

                current_cut += delta

                if current_cut < best_restart_cut:
                    best_restart_cut = current_cut
                    best_restart_train = set(train_ids)

        train_ids = set(best_restart_train)

        for _ in range(local_sweeps):
            train_list = sorted(train_ids)
            test_list = [identity for identity in identities if identity not in train_ids]

            best_delta = 0
            best_swap: Optional[Tuple[str, str]] = None

            for a_train in train_list:
                for b_test in test_list:
                    delta = swap_delta(train_ids, a_train, b_test, adj)

                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (a_train, b_test)

            if best_swap is None:
                break

            a_train, b_test = best_swap
            train_ids.remove(a_train)
            train_ids.add(b_test)
            best_restart_cut += best_delta

        final_cut = compute_cut_weight(train_ids, pair_weights)

        if best_global_cut is None or final_cut < best_global_cut:
            best_global_cut = final_cut
            best_global_train = set(train_ids)

        print(
            f"[restart {restart_idx + 1:>4}/{num_restarts}] "
            f"best_excluded_so_far={best_global_cut}"
        )

    assert best_global_train is not None
    assert best_global_cut is not None

    return best_global_train, best_global_cut


def count_method_morphs(
    method_dir: Path,
    train_ids: Set[str],
    test_ids: Set[str],
    all_bonafide_ids: Set[str],
) -> Dict[str, int]:
    counts = {
        "total_images": 0,
        "train_morphs": 0,
        "test_morphs": 0,
        "excluded_morphs": 0,
        "invalid_name": 0,
        "unknown_identity": 0,
    }

    for path in sorted(method_dir.rglob("*")):
        if not is_image(path):
            continue

        counts["total_images"] += 1

        parsed = parse_morph_identities(path)
        if parsed is None:
            counts["invalid_name"] += 1
            continue

        id_a, id_b = parsed

        if id_a not in all_bonafide_ids or id_b not in all_bonafide_ids:
            counts["unknown_identity"] += 1
            continue

        if id_a in train_ids and id_b in train_ids:
            counts["train_morphs"] += 1
        elif id_a in test_ids and id_b in test_ids:
            counts["test_morphs"] += 1
        else:
            counts["excluded_morphs"] += 1

    counts["usable_morphs"] = counts["train_morphs"] + counts["test_morphs"]

    return counts


def run(
    root: Path,
    train_id_pct: float,
    seed: int,
    methods: Optional[List[str]],
    optimizer: str,
    exact_combination_limit: int,
    num_restarts: int,
    anneal_steps: int,
    local_sweeps: int,
) -> Dict:
    bonafide_dir = root / "BonaFide"
    if not bonafide_dir.exists():
        raise FileNotFoundError(f"Could not find BonaFide folder: {bonafide_dir}")

    all_ids_set, identity_to_files = collect_bonafide_identities(bonafide_dir)

    if len(all_ids_set) == 0:
        raise RuntimeError(f"No bona fide identities found in: {bonafide_dir}")

    identities = sorted(all_ids_set)

    if not 0.0 <= train_id_pct <= 100.0:
        raise ValueError("--train-id-pct must be between 0 and 100")

    n_train = round(len(identities) * train_id_pct / 100.0)

    method_dirs = resolve_method_dirs(root, methods)

    pair_weights, parsing_stats_by_method = collect_weighted_morph_graph(
        method_dirs=method_dirs,
        all_bonafide_ids=all_ids_set,
    )

    n_combinations = math.comb(len(identities), n_train)

    print("\nOptimization setup")
    print("=" * 80)
    print(f"Total identities: {len(identities)}")
    print(f"Requested train identity percentage: {train_id_pct:.2f}%")
    print(f"Train identities to select: {n_train}")
    print(f"Test identities: {len(identities) - n_train}")
    print(f"Valid morph edges/pairs: {len(pair_weights)}")
    print(f"Total valid morphs used in objective: {sum(pair_weights.values())}")
    print(f"Possible identity splits with this size: {n_combinations}")

    if optimizer == "exact":
        if n_combinations > exact_combination_limit:
            raise RuntimeError(
                f"Exact optimization would require checking {n_combinations} splits, "
                f"which is above --exact-combination-limit={exact_combination_limit}. "
                f"Use --optimizer heuristic instead."
            )

        print("\nRunning exact optimization...")
        train_ids, best_cut = exact_optimize_split(
            identities=identities,
            n_train=n_train,
            pair_weights=pair_weights,
        )
        optimizer_used = "exact"

    else:
        print("\nRunning heuristic optimization...")
        train_ids, best_cut = heuristic_optimize_split(
            identities=identities,
            n_train=n_train,
            pair_weights=pair_weights,
            seed=seed,
            num_restarts=num_restarts,
            anneal_steps=anneal_steps,
            local_sweeps=local_sweeps,
        )
        optimizer_used = "heuristic"

    test_ids = set(identities) - train_ids

    results = {
        "root": str(root),
        "seed": seed,
        "train_id_pct": train_id_pct,
        "optimizer": optimizer_used,
        "objective": "minimize_total_excluded_morphs_across_selected_methods",
        "num_total_identities": len(identities),
        "num_train_identities": len(train_ids),
        "num_test_identities": len(test_ids),
        "num_possible_splits_with_this_size": n_combinations,
        "optimized_excluded_morphs_objective": best_cut,
        "num_total_bonafide_images": sum(len(v) for v in identity_to_files.values()),
        "num_train_bonafide_images": sum(len(identity_to_files[i]) for i in train_ids),
        "num_test_bonafide_images": sum(len(identity_to_files[i]) for i in test_ids),
        "methods": {},
        "parsing_stats_by_method": parsing_stats_by_method,
        "train_ids": sorted(train_ids),
        "test_ids": sorted(test_ids),
    }

    for method_dir in method_dirs:
        if not method_dir.exists():
            continue

        results["methods"][method_dir.name] = count_method_morphs(
            method_dir=method_dir,
            train_ids=train_ids,
            test_ids=test_ids,
            all_bonafide_ids=all_ids_set,
        )

    return results


def print_results(results: Dict) -> None:
    print("\nMAD22 optimized identity-disjoint morph split")
    print("=" * 80)

    print(f"Root: {results['root']}")
    print(f"Seed: {results['seed']}")
    print(f"Optimizer: {results['optimizer']}")
    print(f"Training identity percentage: {results['train_id_pct']:.2f}%")

    print("\nBona fide identities")
    print("-" * 80)
    print(f"Total identities: {results['num_total_identities']}")
    print(f"Train identities: {results['num_train_identities']}")
    print(f"Test identities:  {results['num_test_identities']}")

    print("\nBona fide images")
    print("-" * 80)
    print(f"Total bona fide images: {results['num_total_bonafide_images']}")
    print(f"Train bona fide images: {results['num_train_bonafide_images']}")
    print(f"Test bona fide images:  {results['num_test_bonafide_images']}")

    print("\nOptimized objective")
    print("-" * 80)
    print(
        "Excluded morphs under optimized split: "
        f"{results['optimized_excluded_morphs_objective']}"
    )

    print("\nMorph counts by method")
    print("-" * 80)

    header = (
        f"{'Method':<18}"
        f"{'Total':>10}"
        f"{'Train':>10}"
        f"{'Test':>10}"
        f"{'Excluded':>12}"
        f"{'Usable':>10}"
        f"{'Invalid':>10}"
        f"{'UnknownID':>12}"
    )
    print(header)
    print("-" * len(header))

    total_all = defaultdict(int)

    for method, c in results["methods"].items():
        print(
            f"{method:<18}"
            f"{c['total_images']:>10}"
            f"{c['train_morphs']:>10}"
            f"{c['test_morphs']:>10}"
            f"{c['excluded_morphs']:>12}"
            f"{c['usable_morphs']:>10}"
            f"{c['invalid_name']:>10}"
            f"{c['unknown_identity']:>12}"
        )

        for k, v in c.items():
            total_all[k] += v

    print("-" * len(header))
    print(
        f"{'ALL_METHODS':<18}"
        f"{total_all['total_images']:>10}"
        f"{total_all['train_morphs']:>10}"
        f"{total_all['test_morphs']:>10}"
        f"{total_all['excluded_morphs']:>12}"
        f"{total_all['usable_morphs']:>10}"
        f"{total_all['invalid_name']:>10}"
        f"{total_all['unknown_identity']:>12}"
    )

    print("\nMeaning:")
    print("  Train    = both source identities are in the training identity split")
    print("  Test     = both source identities are in the testing identity split")
    print("  Excluded = one source identity is train and the other is test")
    print("  Invalid  = filename could not be parsed as id_expr-vs-id_expr")
    print("  UnknownID = parsed identity was not found in BonaFide")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/MAD22/original_sorted"
        ),
        help="Path to MAD22/original_sorted.",
    )

    parser.add_argument(
        "--train-id-pct",
        type=float,
        required=True,
        help="Percentage of bona fide identities reserved for training, e.g. 80.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the optimization.",
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional method folders to include in the optimization and report. "
            "Default: all folders except BonaFide. "
            "Example: --methods FaceMorpher MIPGAN_I MIPGAN_II OpenCV Webmorph"
        ),
    )

    parser.add_argument(
        "--optimizer",
        choices=["heuristic", "exact"],
        default="heuristic",
        help=(
            "heuristic is recommended for normal use. "
            "exact is only feasible when the number of identity combinations is small."
        ),
    )

    parser.add_argument(
        "--exact-combination-limit",
        type=int,
        default=1_000_000,
        help="Maximum number of combinations allowed for --optimizer exact.",
    )

    parser.add_argument(
        "--num-restarts",
        type=int,
        default=200,
        help="Number of independent heuristic restarts.",
    )

    parser.add_argument(
        "--anneal-steps",
        type=int,
        default=20_000,
        help="Number of simulated annealing swap attempts per restart.",
    )

    parser.add_argument(
        "--local-sweeps",
        type=int,
        default=20,
        help="Number of greedy local-improvement sweeps after each restart.",
    )

    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save results, including exact train/test identity lists.",
    )

    args = parser.parse_args()

    results = run(
        root=args.root,
        train_id_pct=args.train_id_pct,
        seed=args.seed,
        methods=args.methods,
        optimizer=args.optimizer,
        exact_combination_limit=args.exact_combination_limit,
        num_restarts=args.num_restarts,
        anneal_steps=args.anneal_steps,
        local_sweeps=args.local_sweeps,
    )

    print_results(results)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved split/count details to: {args.save_json}")


if __name__ == "__main__":
    main()

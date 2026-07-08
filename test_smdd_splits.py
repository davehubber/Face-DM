#!/usr/bin/env python3
"""
SMDD split optimizer with a fixed number of test morphs.

Goal:
    Select test-side bona fide source images so that:

        test_morphs == --target-test-morphs

    Then assign every other bona fide image to train.

    Among feasible splits, maximize train_morphs.
    Since total valid morphs and test_morphs are fixed, this is equivalent to:

        minimize excluded_morphs

SMDD setup assumed:

Root:
    /nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD

Morphs:
    /nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/m15k_t

Bona fides:
    /nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD/os25k_m_t

Example morph filename:
    morphed_img218974_img496816.png

This morph is interpreted as being built from:
    img218974.png
    img496816.png

A morph is:
    - test: both source images are in the selected test source-image set
    - train: both source images are outside the selected test source-image set
    - excluded: one source image is test and the other is train

Example:
python optimize_smdd_fixed_test_morphs.py \
    --target-test-morphs 1000 \
    --seed 42 \
    --num-restarts 300 \
    --anneal-steps 100000 \
    --polish-steps 100000 \
    --save-json smdd_fixed_1000_test_seed42.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

STRICT_SMDD_RE = re.compile(r"^morphed_(img\d+)_(img\d+)$", re.IGNORECASE)
FALLBACK_SMDD_RE = re.compile(r"^morphed_(.+?)_(.+)$", re.IGNORECASE)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def parse_smdd_morph_sources(path: Path) -> Optional[Tuple[str, str]]:
    """
    Example:
        morphed_img218974_img496816.png

    Returns:
        ("img218974", "img496816")
    """
    stem = path.stem

    match = STRICT_SMDD_RE.match(stem)
    if match:
        return match.group(1), match.group(2)

    match = FALLBACK_SMDD_RE.match(stem)
    if match:
        return match.group(1), match.group(2)

    return None


def collect_bonafides(bonafide_dir: Path) -> Dict[str, Path]:
    """
    Returns:
        {
            "img218974": Path(".../img218974.png"),
            ...
        }
    """
    bonafides: Dict[str, Path] = {}

    for path in sorted(bonafide_dir.rglob("*")):
        if not is_image(path):
            continue

        stem = path.stem

        if stem in bonafides:
            print(f"[WARNING] Duplicate bona fide stem found: {stem}")
            print(f"          Keeping:  {bonafides[stem]}")
            print(f"          Ignoring: {path}")
            continue

        bonafides[stem] = path

    return bonafides


def collect_morph_graph(
    morph_dir: Path,
    bonafide_ids: Set[str],
) -> Tuple[Counter[Tuple[str, str]], Dict[str, int]]:
    """
    Builds a weighted graph over bona fide image IDs.

    Each valid morph contributes one weighted edge between its two source images.
    """
    pair_weights: Counter[Tuple[str, str]] = Counter()

    stats = {
        "total_morph_images": 0,
        "valid_morphs": 0,
        "invalid_name": 0,
        "unknown_bonafide": 0,
        "self_pair_morphs": 0,
    }

    for path in sorted(morph_dir.rglob("*")):
        if not is_image(path):
            continue

        stats["total_morph_images"] += 1

        parsed = parse_smdd_morph_sources(path)
        if parsed is None:
            stats["invalid_name"] += 1
            continue

        src_a, src_b = parsed

        if src_a not in bonafide_ids or src_b not in bonafide_ids:
            stats["unknown_bonafide"] += 1
            continue

        if src_a == src_b:
            stats["self_pair_morphs"] += 1

        edge = tuple(sorted((src_a, src_b)))
        pair_weights[edge] += 1
        stats["valid_morphs"] += 1

    return pair_weights, stats


def build_adjacency(
    pair_weights: Counter[Tuple[str, str]],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    adj: Dict[str, Dict[str, int]] = defaultdict(dict)
    degree: Dict[str, int] = defaultdict(int)

    for (u, v), w in pair_weights.items():
        if u == v:
            continue

        adj[u][v] = adj[u].get(v, 0) + w
        adj[v][u] = adj[v].get(u, 0) + w

        degree[u] += w
        degree[v] += w

    return adj, degree


def compute_split_metrics(
    test_ids: Set[str],
    pair_weights: Counter[Tuple[str, str]],
) -> Dict[str, int]:
    metrics = {
        "train_morphs": 0,
        "test_morphs": 0,
        "excluded_morphs": 0,
    }

    for (u, v), w in pair_weights.items():
        u_test = u in test_ids
        v_test = v in test_ids

        if u_test and v_test:
            metrics["test_morphs"] += w
        elif not u_test and not v_test:
            metrics["train_morphs"] += w
        else:
            metrics["excluded_morphs"] += w

    metrics["usable_morphs"] = metrics["train_morphs"] + metrics["test_morphs"]

    return metrics


def sum_edges_to_set(
    node: str,
    node_set: Set[str],
    adj: Dict[str, Dict[str, int]],
) -> int:
    return sum(w for neighbor, w in adj.get(node, {}).items() if neighbor in node_set)


def add_node_delta(
    node: str,
    test_ids: Set[str],
    adj: Dict[str, Dict[str, int]],
    degree: Dict[str, int],
) -> Dict[str, int]:
    """
    Delta when node moves from train -> test.

    Edges from node to current test nodes:
        excluded -> test

    Edges from node to current train nodes:
        train -> excluded
    """
    edges_to_test = sum_edges_to_set(node, test_ids, adj)
    edges_to_train = degree.get(node, 0) - edges_to_test

    return {
        "test_morphs": edges_to_test,
        "train_morphs": -edges_to_train,
        "excluded_morphs": edges_to_train - edges_to_test,
    }


def remove_node_delta(
    node: str,
    test_ids: Set[str],
    adj: Dict[str, Dict[str, int]],
    degree: Dict[str, int],
) -> Dict[str, int]:
    """
    Delta when node moves from test -> train.

    Edges from node to remaining test nodes:
        test -> excluded

    Edges from node to train nodes:
        excluded -> train
    """
    edges_to_test_including_self = sum_edges_to_set(node, test_ids, adj)
    edges_to_test = edges_to_test_including_self

    edges_to_train = degree.get(node, 0) - edges_to_test

    return {
        "test_morphs": -edges_to_test,
        "train_morphs": edges_to_train,
        "excluded_morphs": edges_to_test - edges_to_train,
    }


def apply_delta(metrics: Dict[str, int], delta: Dict[str, int]) -> Dict[str, int]:
    new_metrics = dict(metrics)

    for key, value in delta.items():
        new_metrics[key] += value

    new_metrics["usable_morphs"] = (
        new_metrics["train_morphs"] + new_metrics["test_morphs"]
    )

    return new_metrics


def objective_value(
    metrics: Dict[str, int],
    target_test_morphs: int,
    penalty_weight: int,
) -> float:
    """
    Penalized objective used during heuristic search.

    The final feasible criterion is still exact:
        test_morphs == target_test_morphs
    """
    test_gap = abs(metrics["test_morphs"] - target_test_morphs)

    # Primary pressure: hit exactly the target number of test morphs.
    # Secondary pressure: once near/at target, minimize excluded morphs.
    return penalty_weight * test_gap + metrics["excluded_morphs"]


def is_feasible(metrics: Dict[str, int], target_test_morphs: int) -> bool:
    return metrics["test_morphs"] == target_test_morphs


def better_feasible(
    candidate_metrics: Dict[str, int],
    best_metrics: Optional[Dict[str, int]],
) -> bool:
    """
    Among feasible splits, prefer:
        1. more train morphs
        2. fewer excluded morphs

    With fixed test_morphs and fixed total morphs, these are equivalent,
    but both are kept for clarity.
    """
    if best_metrics is None:
        return True

    if candidate_metrics["train_morphs"] > best_metrics["train_morphs"]:
        return True

    if (
        candidate_metrics["train_morphs"] == best_metrics["train_morphs"]
        and candidate_metrics["excluded_morphs"] < best_metrics["excluded_morphs"]
    ):
        return True

    return False


def better_penalized_fallback(
    candidate_metrics: Dict[str, int],
    best_metrics: Optional[Dict[str, int]],
    target_test_morphs: int,
) -> bool:
    """
    Used only if no exact feasible split is found.

    Prefer:
        1. smaller absolute gap to target test morphs
        2. more train morphs
        3. fewer excluded morphs
    """
    if best_metrics is None:
        return True

    candidate_gap = abs(candidate_metrics["test_morphs"] - target_test_morphs)
    best_gap = abs(best_metrics["test_morphs"] - target_test_morphs)

    if candidate_gap < best_gap:
        return True

    if candidate_gap == best_gap:
        if candidate_metrics["train_morphs"] > best_metrics["train_morphs"]:
            return True

        if (
            candidate_metrics["train_morphs"] == best_metrics["train_morphs"]
            and candidate_metrics["excluded_morphs"] < best_metrics["excluded_morphs"]
        ):
            return True

    return False


def grow_seed_to_target(
    nodes: List[str],
    target_test_morphs: int,
    adj: Dict[str, Dict[str, int]],
    degree: Dict[str, int],
    rng: random.Random,
) -> Set[str]:
    """
    Builds an initial test set by growing a connected/dense region until the
    induced number of test morphs is near the target.

    This is only a seed. The annealing stage can add/remove nodes afterwards.
    """
    if not nodes:
        return set()

    start = rng.choice(nodes)

    test_ids = {start}
    current_test_morphs = 0

    candidates = set(adj.get(start, {}).keys())

    while current_test_morphs < target_test_morphs and len(test_ids) < len(nodes):
        candidates -= test_ids

        if candidates:
            scored = []

            for node in candidates:
                gain = sum_edges_to_set(node, test_ids, adj)
                scored.append((gain, degree.get(node, 0), node))

            max_gain = max(item[0] for item in scored)

            # Prefer nodes that actually increase the test morph count.
            best = [item for item in scored if item[0] == max_gain]

            _, _, chosen = rng.choice(best)
        else:
            remaining = [node for node in nodes if node not in test_ids]
            chosen = rng.choice(remaining)

        gain = sum_edges_to_set(chosen, test_ids, adj)

        test_ids.add(chosen)
        current_test_morphs += gain

        candidates.update(adj.get(chosen, {}).keys())

        # If we overshoot badly, stop and let the optimizer correct it.
        if current_test_morphs > target_test_morphs * 1.25:
            break

    return test_ids


def random_seed_near_target(
    nodes: List[str],
    target_test_morphs: int,
    avg_degree: float,
    rng: random.Random,
) -> Set[str]:
    """
    Rough random seed with a size estimated from expected internal edges.
    This is deliberately approximate and only used to diversify restarts.
    """
    n = len(nodes)

    if n == 0:
        return set()

    if avg_degree <= 0:
        k = 1
    else:
        # Very rough estimate for internal edges in a random subset:
        # E_inside approx total_edges * (k / n)^2
        # So k approx n * sqrt(target / total_edges).
        # We use avg_degree only to avoid passing total_edges here.
        total_edges_approx = n * avg_degree / 2
        frac = math.sqrt(max(1.0, target_test_morphs) / max(1.0, total_edges_approx))
        k = round(n * frac)

    k = max(1, min(n - 1, k))

    return set(rng.sample(nodes, k))


def propose_move(
    nodes: List[str],
    test_ids: Set[str],
    rng: random.Random,
) -> Tuple[str, Optional[str]]:
    """
    Returns one of:
        ("add", node)
        ("remove", node)
    """
    n_test = len(test_ids)
    n_total = len(nodes)

    if n_test == 0:
        candidates = [node for node in nodes if node not in test_ids]
        return "add", rng.choice(candidates)

    if n_test == n_total:
        return "remove", rng.choice(tuple(test_ids))

    # Randomly choose add/remove.
    # This allows test set size to vary freely.
    if rng.random() < 0.5:
        candidates = [node for node in nodes if node not in test_ids]
        return "add", rng.choice(candidates)

    return "remove", rng.choice(tuple(test_ids))


def optimize_fixed_test_morphs(
    nodes: List[str],
    pair_weights: Counter[Tuple[str, str]],
    target_test_morphs: int,
    seed: int,
    num_restarts: int,
    anneal_steps: int,
    polish_steps: int,
    penalty_weight: Optional[int],
) -> Tuple[Set[str], Dict[str, int], bool]:
    """
    Searches for a test source-image set with exactly target_test_morphs
    internal morphs.

    Among exact feasible splits, keeps the one with the most train morphs.
    """
    if target_test_morphs < 0:
        raise ValueError("--target-test-morphs must be >= 0")

    total_valid_morphs = sum(pair_weights.values())

    if target_test_morphs > total_valid_morphs:
        raise ValueError(
            f"--target-test-morphs={target_test_morphs} is larger than "
            f"the number of valid morphs ({total_valid_morphs})."
        )

    rng = random.Random(seed)
    adj, degree = build_adjacency(pair_weights)

    if penalty_weight is None:
        penalty_weight = max(1, total_valid_morphs * 10)

    avg_degree = sum(degree.values()) / max(1, len(nodes))

    start_temp = max(1.0, total_valid_morphs * 0.10)
    end_temp = 0.01

    best_feasible_test_ids: Optional[Set[str]] = None
    best_feasible_metrics: Optional[Dict[str, int]] = None

    best_fallback_test_ids: Optional[Set[str]] = None
    best_fallback_metrics: Optional[Dict[str, int]] = None

    def update_bests(test_ids: Set[str], metrics: Dict[str, int]) -> None:
        nonlocal best_feasible_test_ids
        nonlocal best_feasible_metrics
        nonlocal best_fallback_test_ids
        nonlocal best_fallback_metrics

        if is_feasible(metrics, target_test_morphs):
            if better_feasible(metrics, best_feasible_metrics):
                best_feasible_test_ids = set(test_ids)
                best_feasible_metrics = dict(metrics)

        if better_penalized_fallback(
            candidate_metrics=metrics,
            best_metrics=best_fallback_metrics,
            target_test_morphs=target_test_morphs,
        ):
            best_fallback_test_ids = set(test_ids)
            best_fallback_metrics = dict(metrics)

    for restart_idx in range(num_restarts):
        # Alternate seed strategies.
        if restart_idx % 2 == 0:
            test_ids = grow_seed_to_target(
                nodes=nodes,
                target_test_morphs=target_test_morphs,
                adj=adj,
                degree=degree,
                rng=rng,
            )
        else:
            test_ids = random_seed_near_target(
                nodes=nodes,
                target_test_morphs=target_test_morphs,
                avg_degree=avg_degree,
                rng=rng,
            )

        metrics = compute_split_metrics(test_ids, pair_weights)
        obj = objective_value(
            metrics=metrics,
            target_test_morphs=target_test_morphs,
            penalty_weight=penalty_weight,
        )

        update_bests(test_ids, metrics)

        best_restart_test_ids = set(test_ids)
        best_restart_metrics = dict(metrics)
        best_restart_obj = obj

        for step in range(anneal_steps):
            frac = step / max(1, anneal_steps - 1)
            temp = start_temp * ((end_temp / start_temp) ** frac)

            move, node = propose_move(nodes, test_ids, rng)

            if node is None:
                continue

            if move == "add":
                delta = add_node_delta(
                    node=node,
                    test_ids=test_ids,
                    adj=adj,
                    degree=degree,
                )
            else:
                delta = remove_node_delta(
                    node=node,
                    test_ids=test_ids,
                    adj=adj,
                    degree=degree,
                )

            new_metrics = apply_delta(metrics, delta)
            new_obj = objective_value(
                metrics=new_metrics,
                target_test_morphs=target_test_morphs,
                penalty_weight=penalty_weight,
            )

            obj_delta = new_obj - obj

            if obj_delta <= 0:
                accept = True
            else:
                accept_prob = math.exp(-obj_delta / max(temp, 1e-12))
                accept = rng.random() < accept_prob

            if accept:
                if move == "add":
                    test_ids.add(node)
                else:
                    test_ids.remove(node)

                metrics = new_metrics
                obj = new_obj

                update_bests(test_ids, metrics)

                if obj < best_restart_obj:
                    best_restart_obj = obj
                    best_restart_test_ids = set(test_ids)
                    best_restart_metrics = dict(metrics)

        # Greedy polish from the best penalized point in this restart.
        test_ids = set(best_restart_test_ids)
        metrics = dict(best_restart_metrics)
        obj = objective_value(
            metrics=metrics,
            target_test_morphs=target_test_morphs,
            penalty_weight=penalty_weight,
        )

        for _ in range(polish_steps):
            move, node = propose_move(nodes, test_ids, rng)

            if node is None:
                continue

            if move == "add":
                delta = add_node_delta(
                    node=node,
                    test_ids=test_ids,
                    adj=adj,
                    degree=degree,
                )
            else:
                delta = remove_node_delta(
                    node=node,
                    test_ids=test_ids,
                    adj=adj,
                    degree=degree,
                )

            new_metrics = apply_delta(metrics, delta)
            new_obj = objective_value(
                metrics=new_metrics,
                target_test_morphs=target_test_morphs,
                penalty_weight=penalty_weight,
            )

            # During polish, only accept strict improvement in penalized objective.
            if new_obj < obj:
                if move == "add":
                    test_ids.add(node)
                else:
                    test_ids.remove(node)

                metrics = new_metrics
                obj = new_obj

                update_bests(test_ids, metrics)

        if best_feasible_metrics is not None:
            status = (
                f"best_train={best_feasible_metrics['train_morphs']} "
                f"best_test={best_feasible_metrics['test_morphs']} "
                f"best_excluded={best_feasible_metrics['excluded_morphs']}"
            )
        else:
            assert best_fallback_metrics is not None
            gap = abs(best_fallback_metrics["test_morphs"] - target_test_morphs)
            status = (
                "no_exact_split_yet "
                f"best_test={best_fallback_metrics['test_morphs']} "
                f"gap={gap} "
                f"train={best_fallback_metrics['train_morphs']} "
                f"excluded={best_fallback_metrics['excluded_morphs']}"
            )

        print(f"[restart {restart_idx + 1:>4}/{num_restarts}] {status}")

    if best_feasible_test_ids is not None and best_feasible_metrics is not None:
        return best_feasible_test_ids, best_feasible_metrics, True

    assert best_fallback_test_ids is not None
    assert best_fallback_metrics is not None

    return best_fallback_test_ids, best_fallback_metrics, False


def run(
    root: Path,
    morph_dir: Path,
    bonafide_dir: Path,
    target_test_morphs: int,
    seed: int,
    num_restarts: int,
    anneal_steps: int,
    polish_steps: int,
    penalty_weight: Optional[int],
) -> Dict:
    if not morph_dir.exists():
        raise FileNotFoundError(f"Could not find morph folder: {morph_dir}")

    if not bonafide_dir.exists():
        raise FileNotFoundError(f"Could not find bona fide folder: {bonafide_dir}")

    bonafides = collect_bonafides(bonafide_dir)
    all_bonafide_ids = set(bonafides.keys())

    if len(all_bonafide_ids) == 0:
        raise RuntimeError(f"No bona fide images found in: {bonafide_dir}")

    pair_weights, morph_stats = collect_morph_graph(
        morph_dir=morph_dir,
        bonafide_ids=all_bonafide_ids,
    )

    active_ids = sorted({
        src
        for pair in pair_weights.keys()
        for src in pair
    })

    inactive_ids = sorted(all_bonafide_ids - set(active_ids))

    if len(active_ids) == 0:
        raise RuntimeError("No valid morph pairs found. Check filenames and paths.")

    total_valid_morphs = sum(pair_weights.values())

    print("\nSMDD fixed-test-morph optimization setup")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Morph folder: {morph_dir}")
    print(f"Bona fide folder: {bonafide_dir}")
    print(f"Total bona fide images: {len(all_bonafide_ids)}")
    print(f"Bona fide images referenced by morphs: {len(active_ids)}")
    print(f"Bona fide images not referenced by morphs: {len(inactive_ids)}")
    print(f"Valid morphs used in objective: {total_valid_morphs}")
    print(f"Target test morphs: {target_test_morphs}")

    print("\nRunning fixed-test-morph split search...")

    test_active_ids, metrics, exact_found = optimize_fixed_test_morphs(
        nodes=active_ids,
        pair_weights=pair_weights,
        target_test_morphs=target_test_morphs,
        seed=seed,
        num_restarts=num_restarts,
        anneal_steps=anneal_steps,
        polish_steps=polish_steps,
        penalty_weight=penalty_weight,
    )

    train_active_ids = set(active_ids) - test_active_ids

    # Inactive bona fide images do not appear in any morph.
    # Assign them to train by default, since they cannot create test morphs.
    train_ids = train_active_ids | set(inactive_ids)
    test_ids = set(test_active_ids)

    final_metrics = compute_split_metrics(test_ids, pair_weights)

    counts = {
        "total_morph_images": morph_stats["total_morph_images"],
        "valid_morphs": morph_stats["valid_morphs"],
        "train_morphs": final_metrics["train_morphs"],
        "test_morphs": final_metrics["test_morphs"],
        "excluded_morphs": final_metrics["excluded_morphs"],
        "usable_morphs": final_metrics["usable_morphs"],
        "invalid_name": morph_stats["invalid_name"],
        "unknown_bonafide": morph_stats["unknown_bonafide"],
        "self_pair_morphs": morph_stats["self_pair_morphs"],
    }

    results = {
        "root": str(root),
        "morph_dir": str(morph_dir),
        "bonafide_dir": str(bonafide_dir),
        "seed": seed,
        "target_test_morphs": target_test_morphs,
        "exact_target_satisfied": exact_found,
        "objective": (
            "choose_test_source_images_with_exact_target_test_morphs_"
            "then_maximize_train_morphs"
        ),
        "note": (
            "This is source-image-disjoint. It is identity-disjoint only if the "
            "SMDD image IDs correspond one-to-one with identities, or if each "
            "identity appears only once in the source set."
        ),
        "num_total_bonafide_images": len(all_bonafide_ids),
        "num_active_bonafide_images_referenced_by_morphs": len(active_ids),
        "num_inactive_bonafide_images_not_referenced_by_morphs": len(inactive_ids),
        "num_train_bonafide_images": len(train_ids),
        "num_test_bonafide_images": len(test_ids),
        "num_train_active_bonafide_images": len(train_active_ids),
        "num_test_active_bonafide_images": len(test_active_ids),
        "morph_counts": counts,
        "train_bonafide_ids": sorted(train_ids),
        "test_bonafide_ids": sorted(test_ids),
        "train_active_bonafide_ids": sorted(train_active_ids),
        "test_active_bonafide_ids": sorted(test_active_ids),
    }

    return results


def print_results(results: Dict) -> None:
    counts = results["morph_counts"]

    print("\nSMDD fixed-test-morph optimized split")
    print("=" * 80)

    print(f"Seed: {results['seed']}")
    print(f"Target test morphs: {results['target_test_morphs']}")
    print(f"Exact target satisfied: {results['exact_target_satisfied']}")

    print("\nBona fide images")
    print("-" * 80)
    print(f"Total bona fide images: {results['num_total_bonafide_images']}")
    print(
        "Referenced by morphs: "
        f"{results['num_active_bonafide_images_referenced_by_morphs']}"
    )
    print(
        "Not referenced by morphs: "
        f"{results['num_inactive_bonafide_images_not_referenced_by_morphs']}"
    )
    print(f"Train bona fide images: {results['num_train_bonafide_images']}")
    print(f"Test bona fide images:  {results['num_test_bonafide_images']}")

    print("\nActive bona fide images")
    print("-" * 80)
    print(f"Train active bona fide images: {results['num_train_active_bonafide_images']}")
    print(f"Test active bona fide images:  {results['num_test_active_bonafide_images']}")

    print("\nMorph counts")
    print("-" * 80)
    print(f"Total morph images: {counts['total_morph_images']}")
    print(f"Valid morphs:       {counts['valid_morphs']}")
    print(f"Train morphs:       {counts['train_morphs']}")
    print(f"Test morphs:        {counts['test_morphs']}")
    print(f"Excluded morphs:    {counts['excluded_morphs']}")
    print(f"Usable morphs:      {counts['usable_morphs']}")
    print(f"Invalid names:      {counts['invalid_name']}")
    print(f"Unknown bonafides:  {counts['unknown_bonafide']}")
    print(f"Self-pair morphs:   {counts['self_pair_morphs']}")

    if counts["valid_morphs"] > 0:
        train_pct = 100.0 * counts["train_morphs"] / counts["valid_morphs"]
        test_pct = 100.0 * counts["test_morphs"] / counts["valid_morphs"]
        excluded_pct = 100.0 * counts["excluded_morphs"] / counts["valid_morphs"]
        usable_pct = 100.0 * counts["usable_morphs"] / counts["valid_morphs"]

        print(f"\nTrain percentage over valid morphs:    {train_pct:.2f}%")
        print(f"Test percentage over valid morphs:     {test_pct:.2f}%")
        print(f"Excluded percentage over valid morphs: {excluded_pct:.2f}%")
        print(f"Usable percentage over valid morphs:   {usable_pct:.2f}%")

    if not results["exact_target_satisfied"]:
        print("\n[WARNING]")
        print(
            "The optimizer did not find a split with exactly "
            f"{results['target_test_morphs']} test morphs."
        )
        print(
            "The reported split is the closest/best fallback found. "
            "Try increasing --num-restarts, --anneal-steps, and --polish-steps."
        )

    print("\nMeaning:")
    print("  Train    = both source bona fide images are outside the test source set")
    print("  Test     = both source bona fide images are inside the test source set")
    print("  Excluded = one source image is test and the other is train")
    print("  Invalid  = filename could not be parsed as morphed_imgA_imgB")
    print("  Unknown  = parsed source image was not found in os25k_m_t")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/nas-ctm01/datasets/public/BIOMETRICS/Face_Morphing/SMDD"),
        help="Path to the SMDD root folder.",
    )

    parser.add_argument(
        "--morph-dir",
        type=Path,
        default=None,
        help="Path to SMDD morph folder. Default: ROOT/m15k_t.",
    )

    parser.add_argument(
        "--bonafide-dir",
        type=Path,
        default=None,
        help="Path to SMDD bona fide source folder. Default: ROOT/os25k_m_t.",
    )

    parser.add_argument(
        "--target-test-morphs",
        type=int,
        default=1000,
        help="Exact number of test morphs to target. Default: 1000.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--num-restarts",
        type=int,
        default=300,
        help="Number of independent optimization restarts.",
    )

    parser.add_argument(
        "--anneal-steps",
        type=int,
        default=100_000,
        help="Number of simulated annealing add/remove attempts per restart.",
    )

    parser.add_argument(
        "--polish-steps",
        type=int,
        default=100_000,
        help="Number of greedy improvement attempts per restart.",
    )

    parser.add_argument(
        "--penalty-weight",
        type=int,
        default=None,
        help=(
            "Penalty per missing/excess test morph during search. "
            "Default: 10 * number of valid morphs."
        ),
    )

    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the split and counts as JSON.",
    )

    args = parser.parse_args()

    morph_dir = args.morph_dir if args.morph_dir is not None else args.root / "m15k_t"
    bonafide_dir = (
        args.bonafide_dir
        if args.bonafide_dir is not None
        else args.root / "os25k_m_t"
    )

    results = run(
        root=args.root,
        morph_dir=morph_dir,
        bonafide_dir=bonafide_dir,
        target_test_morphs=args.target_test_morphs,
        seed=args.seed,
        num_restarts=args.num_restarts,
        anneal_steps=args.anneal_steps,
        polish_steps=args.polish_steps,
        penalty_weight=args.penalty_weight,
    )

    print_results(results)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)

        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved split/count details to: {args.save_json}")


if __name__ == "__main__":
    main()

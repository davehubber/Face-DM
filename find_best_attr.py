import argparse
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


ATTRIBUTES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]


def load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt_path}")

    print(f"Loading classifier checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]

    if isinstance(ckpt, dict):
        return ckpt

    raise RuntimeError(f"Unexpected checkpoint format at: {ckpt_path}")


def load_classifier_tensors(
    ckpt_path: Path,
    use_ema_classifier: bool,
    latent_stats_path: Optional[Path],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads:
      - classifier weight: [40, 512]
      - classifier bias:   [40]
      - conds_mean:        [1, 512]
      - conds_std:         [1, 512]

    The released DiffAE classifier was trained on normalized z_sem:
        z_norm = (z_raw - conds_mean) / conds_std

    Therefore, your raw z_sem must be normalized with THESE stats,
    not with your own train split mean/std.
    """
    sd = load_state_dict(ckpt_path)

    prefix = "ema_classifier" if use_ema_classifier else "classifier"

    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"

    if weight_key not in sd or bias_key not in sd:
        available = "\n".join(sorted(sd.keys())[:80])
        raise KeyError(
            f"Could not find {weight_key} and {bias_key} in checkpoint.\n"
            f"First available keys:\n{available}"
        )

    weight = sd[weight_key].float()
    bias = sd[bias_key].float()

    if "conds_mean" in sd and "conds_std" in sd:
        mean = sd["conds_mean"].float()
        std = sd["conds_std"].float()
        print("Found conds_mean and conds_std inside classifier checkpoint.")
    else:
        if latent_stats_path is None:
            raise KeyError(
                "The classifier checkpoint does not contain conds_mean/conds_std, "
                "and --latent-stats was not provided.\n\n"
                "Provide the DiffAE latent statistics file, usually:\n"
                "  checkpoints/ffhq256_autoenc/latent.pkl"
            )

        latent_stats_path = Path(latent_stats_path).resolve()
        if not latent_stats_path.exists():
            raise FileNotFoundError(f"Latent stats file not found: {latent_stats_path}")

        print(f"Loading latent stats from: {latent_stats_path}")
        stats = torch.load(str(latent_stats_path), map_location="cpu")
        mean = stats["conds_mean"].float()
        std = stats["conds_std"].float()

    mean = mean.reshape(1, -1)
    std = std.reshape(1, -1)

    if weight.shape != (40, 512):
        raise ValueError(f"Expected classifier weight shape [40, 512], got {tuple(weight.shape)}")
    if bias.shape != (40,):
        raise ValueError(f"Expected classifier bias shape [40], got {tuple(bias.shape)}")
    if mean.shape != (1, 512):
        raise ValueError(f"Expected conds_mean shape [1, 512], got {tuple(mean.shape)}")
    if std.shape != (1, 512):
        raise ValueError(f"Expected conds_std shape [1, 512], got {tuple(std.shape)}")

    std = torch.where(std == 0, torch.ones_like(std), std)

    return weight, bias, mean, std


def compute_logits_for_all_embeddings(
    z_path: Path,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Loads raw DiffAE z_sem embeddings from .npy and computes classifier logits.

    Output:
        logits: np.ndarray [N, 40], float32

    Important:
        Input must be raw DiffAE z_sem, not your own z-scored embeddings.
    """
    z_path = Path(z_path).resolve()
    if not z_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {z_path}")

    print(f"Loading embeddings with mmap: {z_path}")
    z = np.load(str(z_path), mmap_mode="r")

    if z.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [N, 512], got {z.shape}")
    if z.shape[1] != 512:
        raise ValueError(f"Expected 512-D DiffAE z_sem embeddings, got shape {z.shape}")

    n = z.shape[0]
    print(f"Number of embeddings: {n:,}")
    print("Assumption: these are RAW DiffAE z_sem embeddings.")

    weight = weight.to(device)
    bias = bias.to(device)
    mean = mean.to(device)
    std = std.to(device)

    logits_out = np.empty((n, 40), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Classifying embeddings"):
            end = min(start + batch_size, n)

            z_batch = torch.from_numpy(np.asarray(z[start:end])).float().to(device)
            z_norm = (z_batch - mean) / std

            logits = F.linear(z_norm, weight, bias)
            logits_out[start:end] = logits.detach().cpu().numpy().astype(np.float32)

    return logits_out


def sample_unique_unordered_pairs(
    n: int,
    num_pairs: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples random unordered pairs (i, j) with:
      - i != j
      - no repeated pair
      - (i, j) and (j, i) treated as the same pair

    Returns:
      i_idx, j_idx arrays, each [num_pairs], with i_idx < j_idx.
    """
    if n < 2:
        raise ValueError("Need at least 2 embeddings to create pairs.")

    max_pairs = n * (n - 1) // 2
    if num_pairs > max_pairs:
        raise ValueError(
            f"Requested {num_pairs:,} pairs, but only {max_pairs:,} unique unordered pairs exist."
        )

    rng = np.random.default_rng(seed)
    collected = []

    total_unique = 0
    oversample = max(10_000, int(num_pairs * 1.05))

    pbar = tqdm(total=num_pairs, desc="Sampling unique unordered pairs")

    while total_unique < num_pairs:
        remaining = num_pairs - total_unique
        draw = max(oversample, int(remaining * 1.10))

        a = rng.integers(0, n, size=draw, dtype=np.int64)
        b = rng.integers(0, n, size=draw, dtype=np.int64)

        keep = a != b
        a = a[keep]
        b = b[keep]

        lo = np.minimum(a, b)
        hi = np.maximum(a, b)

        # Encode unordered pair uniquely. Since lo < hi, this is collision-free.
        codes = lo * np.int64(n) + hi
        codes = np.unique(codes)

        collected.append(codes)

        merged = np.unique(np.concatenate(collected))
        if len(merged) > num_pairs:
            merged = merged[:num_pairs]

        pbar.update(len(merged) - total_unique)
        total_unique = len(merged)
        collected = [merged]

    pbar.close()

    final_codes = collected[0]
    i_idx = final_codes // np.int64(n)
    j_idx = final_codes % np.int64(n)

    assert np.all(i_idx < j_idx)
    assert len(i_idx) == num_pairs
    assert len(np.unique(final_codes)) == num_pairs

    return i_idx.astype(np.int64), j_idx.astype(np.int64)


def analyze_pairs(
    logits: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    pair_batch_size: int,
) -> Dict[str, np.ndarray]:
    """
    Computes attribute-wise separation metrics over sampled pairs.

    Main metrics:
      - mean_abs_logit_diff:
          Average absolute difference in raw classifier logits.

      - mean_abs_prob_diff:
          Average absolute difference in sigmoid(logit).
          This is bounded [0, 1] and easier to interpret.

      - binary_disagreement_rate:
          Fraction of pairs where one embedding is predicted positive
          and the other negative for the attribute.

      - prevalence:
          Fraction of individual embeddings predicted positive for the attribute.

      - prevalence_balance:
          1.0 means prevalence is 50%.
          0.0 means prevalence is 0% or 100%.
          This helps penalize extremely rare or extremely common attributes.

      - practical_score:
          mean_abs_prob_diff * prevalence_balance.
          This favors attributes that both separate pairs and are not too rare/common.
    """
    n, num_attrs = logits.shape
    assert num_attrs == 40

    probs = 1.0 / (1.0 + np.exp(-logits))
    positives = logits > 0

    prevalence = positives.mean(axis=0).astype(np.float64)
    mean_probability = probs.mean(axis=0).astype(np.float64)
    prevalence_balance = (1.0 - np.abs(2.0 * prevalence - 1.0)).astype(np.float64)

    sum_abs_logit_diff = np.zeros(num_attrs, dtype=np.float64)
    sum_abs_prob_diff = np.zeros(num_attrs, dtype=np.float64)
    sum_binary_disagree = np.zeros(num_attrs, dtype=np.float64)

    # For additional interpretability, store approximate percentiles.
    # Exact percentiles over 1M x 40 is still fine in memory, but this avoids
    # holding all pair differences at once.
    all_prob_diffs = np.empty((len(i_idx), num_attrs), dtype=np.float32)

    for start in tqdm(range(0, len(i_idx), pair_batch_size), desc="Analyzing pair differences"):
        end = min(start + pair_batch_size, len(i_idx))
        ii = i_idx[start:end]
        jj = j_idx[start:end]

        logit_diff = np.abs(logits[ii] - logits[jj]).astype(np.float32)
        prob_diff = np.abs(probs[ii] - probs[jj]).astype(np.float32)
        disagree = positives[ii] != positives[jj]

        sum_abs_logit_diff += logit_diff.sum(axis=0)
        sum_abs_prob_diff += prob_diff.sum(axis=0)
        sum_binary_disagree += disagree.sum(axis=0)

        all_prob_diffs[start:end] = prob_diff

    m = float(len(i_idx))

    mean_abs_logit_diff = sum_abs_logit_diff / m
    mean_abs_prob_diff = sum_abs_prob_diff / m
    binary_disagreement_rate = sum_binary_disagree / m

    median_abs_prob_diff = np.percentile(all_prob_diffs, 50, axis=0)
    p90_abs_prob_diff = np.percentile(all_prob_diffs, 90, axis=0)

    practical_score = mean_abs_prob_diff * prevalence_balance

    return {
        "prevalence": prevalence,
        "mean_probability": mean_probability,
        "prevalence_balance": prevalence_balance,
        "mean_abs_logit_diff": mean_abs_logit_diff,
        "mean_abs_prob_diff": mean_abs_prob_diff,
        "median_abs_prob_diff": median_abs_prob_diff,
        "p90_abs_prob_diff": p90_abs_prob_diff,
        "binary_disagreement_rate": binary_disagreement_rate,
        "practical_score": practical_score,
    }


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def write_report(
    out_path: Path,
    embedding_path: Path,
    ckpt_path: Path,
    num_embeddings: int,
    num_pairs: int,
    seed: int,
    use_ema_classifier: bool,
    metrics: Dict[str, np.ndarray],
) -> None:
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order_practical = np.argsort(-metrics["practical_score"])
    order_prob_sep = np.argsort(-metrics["mean_abs_prob_diff"])
    order_disagree = np.argsort(-metrics["binary_disagreement_rate"])

    best_practical = order_practical[0]
    best_prob_sep = order_prob_sep[0]
    best_disagree = order_disagree[0]

    rare_mask = metrics["prevalence"] < 0.05
    common_mask = metrics["prevalence"] > 0.95

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("DIFFAE FFHQ256 ATTRIBUTE PAIR-SEPARATION REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write("Goal\n")
        f.write("----\n")
        f.write(
            "Estimate which of the 40 CelebA/CelebA-HQ attributes most distinguishes random pairs "
            "of DiffAE FFHQ256 semantic embeddings.\n\n"
        )

        f.write("Input configuration\n")
        f.write("-------------------\n")
        f.write(f"Embedding file:              {embedding_path}\n")
        f.write(f"Classifier checkpoint:       {ckpt_path}\n")
        f.write(f"Classifier version:          {'EMA classifier' if use_ema_classifier else 'raw classifier'}\n")
        f.write(f"Number of embeddings:        {num_embeddings:,}\n")
        f.write(f"Number of sampled pairs:     {num_pairs:,}\n")
        f.write(f"Random seed:                 {seed}\n")
        f.write("Pair constraints:            no self-pairs; no repeated unordered pairs; (z1,z2)==(z2,z1)\n")
        f.write("Embedding assumption:        raw DiffAE z_sem, not your own z-scored embeddings\n")
        f.write("Classifier normalization:    DiffAE/classifier conds_mean and conds_std\n\n")

        f.write("Metric definitions\n")
        f.write("------------------\n")
        f.write("prevalence:                  fraction of embeddings with logit > 0 for this attribute\n")
        f.write("mean_probability:            mean sigmoid(logit) over embeddings\n")
        f.write("mean_abs_prob_diff:          mean |sigmoid(logit_i) - sigmoid(logit_j)| over sampled pairs\n")
        f.write("median_abs_prob_diff:        median probability-score difference over sampled pairs\n")
        f.write("p90_abs_prob_diff:           90th percentile probability-score difference over sampled pairs\n")
        f.write("binary_disagreement_rate:    fraction of pairs where one is positive and the other negative\n")
        f.write("mean_abs_logit_diff:         mean absolute raw logit difference\n")
        f.write("prevalence_balance:          1 - abs(2*prevalence - 1); best when prevalence is near 50%\n")
        f.write("practical_score:             mean_abs_prob_diff * prevalence_balance\n\n")

        f.write("Main ranking: best practical attribute separators\n")
        f.write("-------------------------------------------------\n")
        f.write(
            f"{'Rank':>4}  {'Attribute':<24}  "
            f"{'Practical':>10}  {'MeanProbDiff':>12}  {'BinDisagree':>12}  "
            f"{'Prevalence':>11}  {'Balance':>8}  {'MeanProb':>9}  "
            f"{'MedianDiff':>10}  {'P90Diff':>8}  {'MeanLogitDiff':>13}\n"
        )
        f.write("-" * 140 + "\n")

        for rank, idx in enumerate(order_practical, start=1):
            f.write(
                f"{rank:4d}  {ATTRIBUTES[idx]:<24}  "
                f"{metrics['practical_score'][idx]:10.4f}  "
                f"{metrics['mean_abs_prob_diff'][idx]:12.4f}  "
                f"{metrics['binary_disagreement_rate'][idx]:12.4f}  "
                f"{fmt_pct(metrics['prevalence'][idx]):>11}  "
                f"{metrics['prevalence_balance'][idx]:8.4f}  "
                f"{metrics['mean_probability'][idx]:9.4f}  "
                f"{metrics['median_abs_prob_diff'][idx]:10.4f}  "
                f"{metrics['p90_abs_prob_diff'][idx]:8.4f}  "
                f"{metrics['mean_abs_logit_diff'][idx]:13.4f}\n"
            )

        f.write("\n")
        f.write("Alternative rankings\n")
        f.write("--------------------\n")

        f.write("\nTop 10 by raw continuous separation, using mean_abs_prob_diff:\n")
        for rank, idx in enumerate(order_prob_sep[:10], start=1):
            f.write(
                f"{rank:2d}. {ATTRIBUTES[idx]:<24} "
                f"mean_abs_prob_diff={metrics['mean_abs_prob_diff'][idx]:.4f}, "
                f"prevalence={fmt_pct(metrics['prevalence'][idx])}, "
                f"binary_disagreement={fmt_pct(metrics['binary_disagreement_rate'][idx])}\n"
            )

        f.write("\nTop 10 by binary disagreement rate:\n")
        for rank, idx in enumerate(order_disagree[:10], start=1):
            f.write(
                f"{rank:2d}. {ATTRIBUTES[idx]:<24} "
                f"binary_disagreement={fmt_pct(metrics['binary_disagreement_rate'][idx])}, "
                f"prevalence={fmt_pct(metrics['prevalence'][idx])}, "
                f"mean_abs_prob_diff={metrics['mean_abs_prob_diff'][idx]:.4f}\n"
            )

        f.write("\nPresence / rarity notes\n")
        f.write("-----------------------\n")
        if rare_mask.any():
            f.write("Attributes predicted present in <5% of embeddings:\n")
            for idx in np.where(rare_mask)[0]:
                f.write(f"  - {ATTRIBUTES[idx]:<24} prevalence={fmt_pct(metrics['prevalence'][idx])}\n")
        else:
            f.write("No attributes were predicted present in <5% of embeddings.\n")

        if common_mask.any():
            f.write("\nAttributes predicted present in >95% of embeddings:\n")
            for idx in np.where(common_mask)[0]:
                f.write(f"  - {ATTRIBUTES[idx]:<24} prevalence={fmt_pct(metrics['prevalence'][idx])}\n")
        else:
            f.write("\nNo attributes were predicted present in >95% of embeddings.\n")

        f.write("\n")
        f.write("Simple informative summary\n")
        f.write("--------------------------\n")
        f.write(
            f"Best practical separator: {ATTRIBUTES[best_practical]} "
            f"(practical_score={metrics['practical_score'][best_practical]:.4f}, "
            f"mean_abs_prob_diff={metrics['mean_abs_prob_diff'][best_practical]:.4f}, "
            f"binary_disagreement={fmt_pct(metrics['binary_disagreement_rate'][best_practical])}, "
            f"prevalence={fmt_pct(metrics['prevalence'][best_practical])}).\n"
        )
        f.write(
            f"Strongest continuous separator without prevalence penalty: {ATTRIBUTES[best_prob_sep]} "
            f"(mean_abs_prob_diff={metrics['mean_abs_prob_diff'][best_prob_sep]:.4f}, "
            f"prevalence={fmt_pct(metrics['prevalence'][best_prob_sep])}).\n"
        )
        f.write(
            f"Strongest threshold-based separator: {ATTRIBUTES[best_disagree]} "
            f"(binary_disagreement={fmt_pct(metrics['binary_disagreement_rate'][best_disagree])}, "
            f"prevalence={fmt_pct(metrics['prevalence'][best_disagree])}).\n"
        )
        f.write(
            "\nRecommended interpretation: for deterministic pair ordering, prefer attributes with high "
            "mean_abs_prob_diff or binary_disagreement_rate, but avoid attributes with extremely low or "
            "extremely high prevalence unless you specifically want a rare/common feature rule. The "
            "practical_score is the most useful single summary because it rewards separation while "
            "penalizing attributes that are too rare or too universal.\n"
        )

    print(f"\nSaved report to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze which DiffAE/CelebA attribute classifier outputs most distinguish "
            "random unordered pairs of FFHQ256 DiffAE semantic embeddings."
        )
    )

    parser.add_argument(
        "--train-zsem",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_train.npy",
        help="Path to raw DiffAE z_sem training split, shape [N, 512].",
    )
    parser.add_argument(
        "--classifier-ckpt",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc_cls/last.ckpt",
        help="Path to released DiffAE FFHQ256 attribute classifier checkpoint.",
    )
    parser.add_argument(
        "--latent-stats",
        type=str,
        default=None,
        help=(
            "Optional path to checkpoints/ffhq256_autoenc/latent.pkl. "
            "Usually not needed if conds_mean/conds_std are inside the classifier checkpoint."
        ),
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_train_attribute_pair_report.txt",
        help="Output .txt report path.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=1_000_000,
        help="Number of unique unordered random pairs to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pair sampling.",
    )
    parser.add_argument(
        "--classify-batch-size",
        type=int,
        default=8192,
        help="Batch size for computing classifier logits.",
    )
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=65536,
        help="Batch size for pair-difference analysis.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for classifier forward pass.",
    )
    parser.add_argument(
        "--use-ema-classifier",
        action="store_true",
        help=(
            "Use ema_classifier weights instead of classifier weights. "
            "Default matches the official manipulation notebook more closely: classifier weights."
        ),
    )

    args = parser.parse_args()

    train_zsem_path = Path(args.train_zsem).resolve()
    classifier_ckpt_path = Path(args.classifier_ckpt).resolve()
    latent_stats_path = Path(args.latent_stats).resolve() if args.latent_stats is not None else None
    out_report_path = Path(args.out_report).resolve()
    device = torch.device(args.device)

    print("=" * 100)
    print("DiffAE FFHQ256 attribute pair-separation analysis")
    print("=" * 100)
    print(f"Train z_sem:        {train_zsem_path}")
    print(f"Classifier ckpt:    {classifier_ckpt_path}")
    print(f"Latent stats:       {latent_stats_path}")
    print(f"Output report:      {out_report_path}")
    print(f"Num pairs:          {args.num_pairs:,}")
    print(f"Seed:               {args.seed}")
    print(f"Device:             {device}")
    print(f"Use EMA classifier: {args.use_ema_classifier}")
    print("=" * 100)

    weight, bias, mean, std = load_classifier_tensors(
        ckpt_path=classifier_ckpt_path,
        use_ema_classifier=args.use_ema_classifier,
        latent_stats_path=latent_stats_path,
    )

    logits = compute_logits_for_all_embeddings(
        z_path=train_zsem_path,
        weight=weight,
        bias=bias,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.classify_batch_size,
    )

    num_embeddings = logits.shape[0]

    i_idx, j_idx = sample_unique_unordered_pairs(
        n=num_embeddings,
        num_pairs=args.num_pairs,
        seed=args.seed,
    )

    metrics = analyze_pairs(
        logits=logits,
        i_idx=i_idx,
        j_idx=j_idx,
        pair_batch_size=args.pair_batch_size,
    )

    write_report(
        out_path=out_report_path,
        embedding_path=train_zsem_path,
        ckpt_path=classifier_ckpt_path,
        num_embeddings=num_embeddings,
        num_pairs=args.num_pairs,
        seed=args.seed,
        use_ema_classifier=args.use_ema_classifier,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()

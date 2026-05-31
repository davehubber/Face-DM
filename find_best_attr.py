import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


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

    The classifier expects raw DiffAE z_sem normalized as:
        z_norm = (z_raw - conds_mean) / conds_std

    Do not use your own train split mean/std here.
    """
    sd = load_state_dict(ckpt_path)

    prefix = "ema_classifier" if use_ema_classifier else "classifier"

    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"

    if weight_key not in sd or bias_key not in sd:
        available = "\n".join(sorted(sd.keys())[:100])
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


def classify_embeddings_mmap(
    z_path: Path,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Computes classifier logits for every clean raw z_sem embedding.

    Returns:
        clean_logits: [N, 40], float32
    """
    z_path = Path(z_path).resolve()
    if not z_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {z_path}")

    z = np.load(str(z_path), mmap_mode="r")

    if z.ndim != 2 or z.shape[1] != 512:
        raise ValueError(f"Expected raw DiffAE z_sem shape [N, 512], got {z.shape}")

    n = z.shape[0]
    logits_out = np.empty((n, 40), dtype=np.float32)

    weight = weight.to(device)
    bias = bias.to(device)
    mean = mean.to(device)
    std = std.to(device)

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Classifying clean embeddings"):
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
      i_idx, j_idx, both [num_pairs], with i_idx < j_idx.
    """
    if n < 2:
        raise ValueError("Need at least 2 embeddings.")

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


def analyze_average_midpoint_behavior(
    z_path: Path,
    clean_logits: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    pair_batch_size: int,
    min_clean_logit_gap: float,
    min_clean_prob_gap: float,
) -> Dict[str, np.ndarray]:
    """
    For each pair:
        z_avg = 0.5 * (z_i + z_j)

    Then compares classifier score of z_avg against the midpoint of the
    clean scores.

    For logits, midpoint behavior should be exact up to numerical error.

    For sigmoid probabilities, midpoint behavior is not guaranteed because
    sigmoid is nonlinear.

    Main per-attribute metrics:
      - clean prevalence
      - clean separation
      - mean absolute midpoint error for logits
      - mean absolute midpoint error for probabilities
      - normalized position of average score between the two clean scores

    Normalized position:
        low = min(score_i, score_j)
        high = max(score_i, score_j)
        position = (score_avg - low) / (high - low)

    Interpretation:
        position = 0.5  -> exactly equidistant
        position > 0.5  -> closer to the high-score embedding
        position < 0.5  -> closer to the low-score embedding
    """
    z = np.load(str(z_path), mmap_mode="r")

    weight = weight.to(device)
    bias = bias.to(device)
    mean = mean.to(device)
    std = std.to(device)

    n_attrs = 40
    n_pairs = len(i_idx)

    clean_probs = sigmoid_np(clean_logits)
    clean_positive = clean_logits > 0

    prevalence = clean_positive.mean(axis=0).astype(np.float64)
    mean_clean_prob = clean_probs.mean(axis=0).astype(np.float64)

    sum_clean_abs_logit_gap = np.zeros(n_attrs, dtype=np.float64)
    sum_clean_abs_prob_gap = np.zeros(n_attrs, dtype=np.float64)

    sum_abs_logit_mid_error = np.zeros(n_attrs, dtype=np.float64)
    max_abs_logit_mid_error = np.zeros(n_attrs, dtype=np.float64)

    sum_abs_prob_mid_error = np.zeros(n_attrs, dtype=np.float64)
    sum_signed_prob_mid_error = np.zeros(n_attrs, dtype=np.float64)
    max_abs_prob_mid_error = np.zeros(n_attrs, dtype=np.float64)

    sum_logit_pos_minus_half = np.zeros(n_attrs, dtype=np.float64)
    sum_abs_logit_pos_minus_half = np.zeros(n_attrs, dtype=np.float64)
    count_valid_logit_pos = np.zeros(n_attrs, dtype=np.float64)

    sum_prob_pos_minus_half = np.zeros(n_attrs, dtype=np.float64)
    sum_abs_prob_pos_minus_half = np.zeros(n_attrs, dtype=np.float64)
    count_valid_prob_pos = np.zeros(n_attrs, dtype=np.float64)

    count_prob_closer_to_high = np.zeros(n_attrs, dtype=np.float64)
    count_prob_closer_to_low = np.zeros(n_attrs, dtype=np.float64)
    count_prob_exact_mid = np.zeros(n_attrs, dtype=np.float64)

    # Store only probability normalized deviations for percentiles.
    # Shape [1_000_000, 40] float32 is about 160 MB.
    prob_pos_minus_half_all = np.empty((n_pairs, n_attrs), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, n_pairs, pair_batch_size), desc="Classifying averaged embeddings"):
            end = min(start + pair_batch_size, n_pairs)

            ii = i_idx[start:end]
            jj = j_idx[start:end]

            z_i = torch.from_numpy(np.asarray(z[ii])).float().to(device)
            z_j = torch.from_numpy(np.asarray(z[jj])).float().to(device)
            z_avg = 0.5 * (z_i + z_j)

            z_avg_norm = (z_avg - mean) / std
            avg_logits_t = F.linear(z_avg_norm, weight, bias)
            avg_logits = avg_logits_t.detach().cpu().numpy().astype(np.float32)
            avg_probs = sigmoid_np(avg_logits).astype(np.float32)

            l_i = clean_logits[ii]
            l_j = clean_logits[jj]
            p_i = clean_probs[ii]
            p_j = clean_probs[jj]

            expected_logit_mid = 0.5 * (l_i + l_j)
            expected_prob_mid = 0.5 * (p_i + p_j)

            clean_abs_logit_gap = np.abs(l_i - l_j)
            clean_abs_prob_gap = np.abs(p_i - p_j)

            logit_mid_error = avg_logits - expected_logit_mid
            prob_mid_error = avg_probs - expected_prob_mid

            sum_clean_abs_logit_gap += clean_abs_logit_gap.sum(axis=0)
            sum_clean_abs_prob_gap += clean_abs_prob_gap.sum(axis=0)

            abs_logit_mid_error = np.abs(logit_mid_error)
            sum_abs_logit_mid_error += abs_logit_mid_error.sum(axis=0)
            max_abs_logit_mid_error = np.maximum(
                max_abs_logit_mid_error,
                abs_logit_mid_error.max(axis=0),
            )

            abs_prob_mid_error = np.abs(prob_mid_error)
            sum_abs_prob_mid_error += abs_prob_mid_error.sum(axis=0)
            sum_signed_prob_mid_error += prob_mid_error.sum(axis=0)
            max_abs_prob_mid_error = np.maximum(
                max_abs_prob_mid_error,
                abs_prob_mid_error.max(axis=0),
            )

            # Normalized position in logit space.
            low_l = np.minimum(l_i, l_j)
            high_l = np.maximum(l_i, l_j)
            gap_l = high_l - low_l
            valid_l = gap_l > min_clean_logit_gap

            pos_l = np.zeros_like(avg_logits, dtype=np.float32)
            pos_l[valid_l] = (avg_logits[valid_l] - low_l[valid_l]) / gap_l[valid_l]
            dev_l = pos_l - 0.5

            sum_logit_pos_minus_half += np.where(valid_l, dev_l, 0.0).sum(axis=0)
            sum_abs_logit_pos_minus_half += np.where(valid_l, np.abs(dev_l), 0.0).sum(axis=0)
            count_valid_logit_pos += valid_l.sum(axis=0)

            # Normalized position in probability space.
            low_p = np.minimum(p_i, p_j)
            high_p = np.maximum(p_i, p_j)
            gap_p = high_p - low_p
            valid_p = gap_p > min_clean_prob_gap

            pos_p = np.full_like(avg_probs, 0.5, dtype=np.float32)
            pos_p[valid_p] = (avg_probs[valid_p] - low_p[valid_p]) / gap_p[valid_p]
            dev_p = pos_p - 0.5

            prob_pos_minus_half_all[start:end] = np.where(valid_p, dev_p, np.nan)

            sum_prob_pos_minus_half += np.where(valid_p, dev_p, 0.0).sum(axis=0)
            sum_abs_prob_pos_minus_half += np.where(valid_p, np.abs(dev_p), 0.0).sum(axis=0)
            count_valid_prob_pos += valid_p.sum(axis=0)

            eps = 1e-7
            count_prob_closer_to_high += ((dev_p > eps) & valid_p).sum(axis=0)
            count_prob_closer_to_low += ((dev_p < -eps) & valid_p).sum(axis=0)
            count_prob_exact_mid += ((np.abs(dev_p) <= eps) & valid_p).sum(axis=0)

    m = float(n_pairs)

    mean_abs_logit_mid_error = sum_abs_logit_mid_error / m
    mean_abs_prob_mid_error = sum_abs_prob_mid_error / m
    mean_signed_prob_mid_error = sum_signed_prob_mid_error / m

    mean_clean_abs_logit_gap = sum_clean_abs_logit_gap / m
    mean_clean_abs_prob_gap = sum_clean_abs_prob_gap / m

    safe_logit_counts = np.maximum(count_valid_logit_pos, 1.0)
    safe_prob_counts = np.maximum(count_valid_prob_pos, 1.0)

    mean_logit_pos_minus_half = sum_logit_pos_minus_half / safe_logit_counts
    mean_abs_logit_pos_minus_half = sum_abs_logit_pos_minus_half / safe_logit_counts

    mean_prob_pos_minus_half = sum_prob_pos_minus_half / safe_prob_counts
    mean_abs_prob_pos_minus_half = sum_abs_prob_pos_minus_half / safe_prob_counts

    prob_closer_to_high_rate = count_prob_closer_to_high / safe_prob_counts
    prob_closer_to_low_rate = count_prob_closer_to_low / safe_prob_counts
    prob_exact_mid_rate = count_prob_exact_mid / safe_prob_counts

    # Percentiles of probability-space normalized imbalance.
    p50_abs_prob_pos_dev = np.zeros(n_attrs, dtype=np.float64)
    p90_abs_prob_pos_dev = np.zeros(n_attrs, dtype=np.float64)
    p99_abs_prob_pos_dev = np.zeros(n_attrs, dtype=np.float64)

    for a in range(n_attrs):
        vals = np.abs(prob_pos_minus_half_all[:, a])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            p50_abs_prob_pos_dev[a] = np.nan
            p90_abs_prob_pos_dev[a] = np.nan
            p99_abs_prob_pos_dev[a] = np.nan
        else:
            p50_abs_prob_pos_dev[a] = np.percentile(vals, 50)
            p90_abs_prob_pos_dev[a] = np.percentile(vals, 90)
            p99_abs_prob_pos_dev[a] = np.percentile(vals, 99)

    return {
        "prevalence": prevalence,
        "mean_clean_prob": mean_clean_prob,
        "mean_clean_abs_logit_gap": mean_clean_abs_logit_gap,
        "mean_clean_abs_prob_gap": mean_clean_abs_prob_gap,
        "mean_abs_logit_mid_error": mean_abs_logit_mid_error,
        "max_abs_logit_mid_error": max_abs_logit_mid_error,
        "mean_abs_prob_mid_error": mean_abs_prob_mid_error,
        "mean_signed_prob_mid_error": mean_signed_prob_mid_error,
        "max_abs_prob_mid_error": max_abs_prob_mid_error,
        "valid_logit_position_rate": count_valid_logit_pos / m,
        "mean_logit_pos_minus_half": mean_logit_pos_minus_half,
        "mean_abs_logit_pos_minus_half": mean_abs_logit_pos_minus_half,
        "valid_prob_position_rate": count_valid_prob_pos / m,
        "mean_prob_pos_minus_half": mean_prob_pos_minus_half,
        "mean_abs_prob_pos_minus_half": mean_abs_prob_pos_minus_half,
        "p50_abs_prob_pos_dev": p50_abs_prob_pos_dev,
        "p90_abs_prob_pos_dev": p90_abs_prob_pos_dev,
        "p99_abs_prob_pos_dev": p99_abs_prob_pos_dev,
        "prob_closer_to_high_rate": prob_closer_to_high_rate,
        "prob_closer_to_low_rate": prob_closer_to_low_rate,
        "prob_exact_mid_rate": prob_exact_mid_rate,
    }


def write_report(
    out_path: Path,
    train_zsem_path: Path,
    classifier_ckpt_path: Path,
    num_embeddings: int,
    num_pairs: int,
    seed: int,
    use_ema_classifier: bool,
    min_clean_logit_gap: float,
    min_clean_prob_gap: float,
    metrics: Dict[str, np.ndarray],
) -> None:
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    order_prob_imbalance = np.argsort(-metrics["mean_abs_prob_pos_minus_half"])
    order_prob_mid_error = np.argsort(-metrics["mean_abs_prob_mid_error"])
    order_clean_prob_gap = np.argsort(-metrics["mean_clean_abs_prob_gap"])

    best_imbalance = order_prob_imbalance[0]
    best_prob_mid_error = order_prob_mid_error[0]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write("DIFFAE FFHQ256 ATTRIBUTE AVERAGE-MIDPOINT REPORT\n")
        f.write("=" * 110 + "\n\n")

        f.write("Goal\n")
        f.write("----\n")
        f.write(
            "For random pairs of raw DiffAE FFHQ256 semantic embeddings z1 and z2, test whether the "
            "attribute score of z_avg = 0.5*z1 + 0.5*z2 is equidistant from the attribute scores "
            "of z1 and z2.\n\n"
        )

        f.write("Important mathematical expectation\n")
        f.write("----------------------------------\n")
        f.write(
            "For raw classifier logits, the average embedding should be exactly at the midpoint. "
            "This is because DiffAE classifier scoring is affine normalization followed by a linear "
            "classifier. Therefore, logit(z_avg) should equal 0.5*logit(z1) + 0.5*logit(z2), up to "
            "small floating-point error.\n\n"
        )
        f.write(
            "For sigmoid probabilities, midpoint behavior is not guaranteed. sigmoid(logit(z_avg)) "
            "does not generally equal 0.5*sigmoid(logit(z1)) + 0.5*sigmoid(logit(z2)). Probability-space "
            "imbalance can therefore appear even when logit-space behavior is perfectly balanced.\n\n"
        )

        f.write("Input configuration\n")
        f.write("-------------------\n")
        f.write(f"Embedding file:                  {train_zsem_path}\n")
        f.write(f"Classifier checkpoint:           {classifier_ckpt_path}\n")
        f.write(f"Classifier version:              {'EMA classifier' if use_ema_classifier else 'raw classifier'}\n")
        f.write(f"Number of embeddings:            {num_embeddings:,}\n")
        f.write(f"Number of sampled pairs:         {num_pairs:,}\n")
        f.write(f"Random seed:                     {seed}\n")
        f.write("Pair constraints:                no self-pairs; no repeated unordered pairs; (z1,z2)==(z2,z1)\n")
        f.write("Embedding assumption:            raw DiffAE z_sem, not your own z-scored embeddings\n")
        f.write("Classifier normalization:        DiffAE/classifier conds_mean and conds_std\n")
        f.write(f"Minimum clean logit gap:         {min_clean_logit_gap}\n")
        f.write(f"Minimum clean probability gap:   {min_clean_prob_gap}\n\n")

        f.write("Metric definitions\n")
        f.write("------------------\n")
        f.write("prevalence:                      fraction of clean embeddings with logit > 0\n")
        f.write("mean_clean_abs_prob_gap:         mean |prob(z1)-prob(z2)| across sampled pairs\n")
        f.write("mean_abs_logit_mid_error:        mean |logit(z_avg) - midpoint(logit(z1),logit(z2))|\n")
        f.write("max_abs_logit_mid_error:         maximum absolute logit midpoint error\n")
        f.write("mean_abs_prob_mid_error:         mean |prob(z_avg) - midpoint(prob(z1),prob(z2))|\n")
        f.write("mean_prob_pos_minus_half:        mean normalized probability position minus 0.5\n")
        f.write("mean_abs_prob_pos_minus_half:    mean absolute normalized probability imbalance\n")
        f.write("prob_closer_to_high_rate:        fraction where prob(z_avg) is closer to the higher-prob clean embedding\n")
        f.write("prob_closer_to_low_rate:         fraction where prob(z_avg) is closer to the lower-prob clean embedding\n\n")

        f.write("Full per-attribute table\n")
        f.write("------------------------\n")
        f.write(
            f"{'Attr':<24}  {'Prev':>8}  {'MeanProb':>9}  "
            f"{'CleanProbGap':>12}  {'LogitMidErrMean':>16}  {'LogitMidErrMax':>15}  "
            f"{'ProbMidErrMean':>15}  {'ProbMidErrSigned':>17}  "
            f"{'ProbPos-0.5':>12}  {'AbsProbPosDev':>13}  "
            f"{'P50AbsDev':>10}  {'P90AbsDev':>10}  {'P99AbsDev':>10}  "
            f"{'CloserHigh':>11}  {'CloserLow':>10}\n"
        )
        f.write("-" * 190 + "\n")

        for idx, attr in enumerate(ATTRIBUTES):
            f.write(
                f"{attr:<24}  "
                f"{fmt_pct(metrics['prevalence'][idx]):>8}  "
                f"{metrics['mean_clean_prob'][idx]:9.4f}  "
                f"{metrics['mean_clean_abs_prob_gap'][idx]:12.4f}  "
                f"{metrics['mean_abs_logit_mid_error'][idx]:16.8e}  "
                f"{metrics['max_abs_logit_mid_error'][idx]:15.8e}  "
                f"{metrics['mean_abs_prob_mid_error'][idx]:15.6f}  "
                f"{metrics['mean_signed_prob_mid_error'][idx]:17.6f}  "
                f"{metrics['mean_prob_pos_minus_half'][idx]:12.6f}  "
                f"{metrics['mean_abs_prob_pos_minus_half'][idx]:13.6f}  "
                f"{metrics['p50_abs_prob_pos_dev'][idx]:10.6f}  "
                f"{metrics['p90_abs_prob_pos_dev'][idx]:10.6f}  "
                f"{metrics['p99_abs_prob_pos_dev'][idx]:10.6f}  "
                f"{fmt_pct(metrics['prob_closer_to_high_rate'][idx]):>11}  "
                f"{fmt_pct(metrics['prob_closer_to_low_rate'][idx]):>10}\n"
            )

        f.write("\n")
        f.write("Top attributes by probability-space imbalance\n")
        f.write("---------------------------------------------\n")
        for rank, idx in enumerate(order_prob_imbalance[:10], start=1):
            f.write(
                f"{rank:2d}. {ATTRIBUTES[idx]:<24} "
                f"mean_abs_prob_pos_dev={metrics['mean_abs_prob_pos_minus_half'][idx]:.6f}, "
                f"mean_prob_pos_minus_half={metrics['mean_prob_pos_minus_half'][idx]:+.6f}, "
                f"closer_high={fmt_pct(metrics['prob_closer_to_high_rate'][idx])}, "
                f"closer_low={fmt_pct(metrics['prob_closer_to_low_rate'][idx])}, "
                f"prevalence={fmt_pct(metrics['prevalence'][idx])}\n"
            )

        f.write("\n")
        f.write("Top attributes by probability midpoint error\n")
        f.write("--------------------------------------------\n")
        for rank, idx in enumerate(order_prob_mid_error[:10], start=1):
            f.write(
                f"{rank:2d}. {ATTRIBUTES[idx]:<24} "
                f"mean_abs_prob_mid_error={metrics['mean_abs_prob_mid_error'][idx]:.6f}, "
                f"signed_prob_mid_error={metrics['mean_signed_prob_mid_error'][idx]:+.6f}, "
                f"mean_clean_prob_gap={metrics['mean_clean_abs_prob_gap'][idx]:.4f}, "
                f"prevalence={fmt_pct(metrics['prevalence'][idx])}\n"
            )

        f.write("\n")
        f.write("Top attributes by clean probability separation\n")
        f.write("----------------------------------------------\n")
        for rank, idx in enumerate(order_clean_prob_gap[:10], start=1):
            f.write(
                f"{rank:2d}. {ATTRIBUTES[idx]:<24} "
                f"mean_clean_abs_prob_gap={metrics['mean_clean_abs_prob_gap'][idx]:.4f}, "
                f"mean_abs_prob_pos_dev={metrics['mean_abs_prob_pos_minus_half'][idx]:.6f}, "
                f"prevalence={fmt_pct(metrics['prevalence'][idx])}\n"
            )

        f.write("\n")
        f.write("Simple informative summary\n")
        f.write("--------------------------\n")

        max_logit_err = metrics["max_abs_logit_mid_error"].max()
        mean_logit_err = metrics["mean_abs_logit_mid_error"].mean()

        f.write(
            f"Across all attributes, the average maximum absolute logit midpoint error was approximately "
            f"{mean_logit_err:.8e}, and the worst observed maximum logit midpoint error was "
            f"{max_logit_err:.8e}.\n"
        )

        f.write(
            "This should be extremely small. If it is, then there is no useful imbalance in raw classifier-logit "
            "space: the averaged embedding is exactly halfway between both clean embeddings for every attribute.\n"
        )

        f.write(
            f"\nThe strongest probability-space imbalance was observed for {ATTRIBUTES[best_imbalance]} "
            f"(mean_abs_prob_pos_dev={metrics['mean_abs_prob_pos_minus_half'][best_imbalance]:.6f}, "
            f"mean_prob_pos_minus_half={metrics['mean_prob_pos_minus_half'][best_imbalance]:+.6f}, "
            f"closer_high={fmt_pct(metrics['prob_closer_to_high_rate'][best_imbalance])}, "
            f"closer_low={fmt_pct(metrics['prob_closer_to_low_rate'][best_imbalance])}).\n"
        )

        f.write(
            f"The largest direct probability midpoint error was observed for {ATTRIBUTES[best_prob_mid_error]} "
            f"(mean_abs_prob_mid_error={metrics['mean_abs_prob_mid_error'][best_prob_mid_error]:.6f}).\n"
        )

        f.write(
            "\nInterpretation: if your downstream rule uses raw logits, averaging gives no asymmetry to exploit. "
            "If your downstream rule uses sigmoid probabilities, apparent imbalance may appear, but this comes from "
            "the nonlinear sigmoid transform rather than from information preserved asymmetrically in the averaged "
            "embedding. Therefore, probability-space imbalance should be interpreted cautiously.\n"
        )

    print(f"\nSaved report to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether DiffAE attribute classifier scores of averaged embeddings are equidistant "
            "from the scores of the two clean embeddings."
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
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_train_attribute_average_midpoint_report.txt",
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
        help="Batch size for computing clean classifier logits.",
    )
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=16384,
        help="Batch size for classifying averaged embeddings.",
    )
    parser.add_argument(
        "--min-clean-logit-gap",
        type=float,
        default=1e-6,
        help="Minimum clean logit gap needed to compute normalized logit position.",
    )
    parser.add_argument(
        "--min-clean-prob-gap",
        type=float,
        default=1e-5,
        help="Minimum clean probability gap needed to compute normalized probability position.",
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

    print("=" * 110)
    print("DiffAE FFHQ256 attribute average-midpoint analysis")
    print("=" * 110)
    print(f"Train z_sem:              {train_zsem_path}")
    print(f"Classifier ckpt:          {classifier_ckpt_path}")
    print(f"Latent stats:             {latent_stats_path}")
    print(f"Output report:            {out_report_path}")
    print(f"Num pairs:                {args.num_pairs:,}")
    print(f"Seed:                     {args.seed}")
    print(f"Device:                   {device}")
    print(f"Use EMA classifier:       {args.use_ema_classifier}")
    print(f"Min clean logit gap:      {args.min_clean_logit_gap}")
    print(f"Min clean probability gap:{args.min_clean_prob_gap}")
    print("=" * 110)

    weight, bias, mean, std = load_classifier_tensors(
        ckpt_path=classifier_ckpt_path,
        use_ema_classifier=args.use_ema_classifier,
        latent_stats_path=latent_stats_path,
    )

    clean_logits = classify_embeddings_mmap(
        z_path=train_zsem_path,
        weight=weight,
        bias=bias,
        mean=mean,
        std=std,
        device=device,
        batch_size=args.classify_batch_size,
    )

    num_embeddings = clean_logits.shape[0]

    i_idx, j_idx = sample_unique_unordered_pairs(
        n=num_embeddings,
        num_pairs=args.num_pairs,
        seed=args.seed,
    )

    metrics = analyze_average_midpoint_behavior(
        z_path=train_zsem_path,
        clean_logits=clean_logits,
        i_idx=i_idx,
        j_idx=j_idx,
        weight=weight,
        bias=bias,
        mean=mean,
        std=std,
        device=device,
        pair_batch_size=args.pair_batch_size,
        min_clean_logit_gap=args.min_clean_logit_gap,
        min_clean_prob_gap=args.min_clean_prob_gap,
    )

    write_report(
        out_path=out_report_path,
        train_zsem_path=train_zsem_path,
        classifier_ckpt_path=classifier_ckpt_path,
        num_embeddings=num_embeddings,
        num_pairs=args.num_pairs,
        seed=args.seed,
        use_ema_classifier=args.use_ema_classifier,
        min_clean_logit_gap=args.min_clean_logit_gap,
        min_clean_prob_gap=args.min_clean_prob_gap,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
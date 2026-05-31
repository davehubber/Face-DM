import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# Official CelebA/CelebA-HQ attribute order used by the DiffAE classifier.
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

ATTR_TO_ID = {name: i for i, name in enumerate(ATTRIBUTES)}


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    ckpt_path = Path(ckpt_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found:\n  {ckpt_path}\n\n"
            "Expected something like:\n"
            "  /nas-ctm01/homes/dacordeiro/diffae/checkpoints/ffhq256_autoenc_cls/last.ckpt"
        )

    print(f"Loading classifier checkpoint:\n  {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]

    if isinstance(ckpt, dict):
        return ckpt

    raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")


def load_classifier_tensors(
    ckpt_path: Path,
    latent_stats_path: Optional[Path],
    use_ema_classifier: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads the released DiffAE attribute classifier tensors.

    Returns:
        weight: [40, 512]
        bias:   [40]
        mean:   [1, 512]
        std:    [1, 512]

    Important:
        Your input embeddings should be RAW DiffAE z_sem.
        The script normalizes them internally using the classifier/DiffAE stats:
            z_norm = (z_raw - conds_mean) / conds_std

        Do NOT use your own train_mean/train_std for these classifier scores.
    """
    sd = load_state_dict(ckpt_path)

    prefix = "ema_classifier" if use_ema_classifier else "classifier"

    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"

    if weight_key not in sd or bias_key not in sd:
        available = "\n".join(sorted(sd.keys())[:120])
        raise KeyError(
            f"Could not find classifier tensors in checkpoint.\n"
            f"Missing keys: {weight_key}, {bias_key}\n\n"
            f"First available keys:\n{available}"
        )

    weight = sd[weight_key].float()
    bias = sd[bias_key].float()

    if "conds_mean" in sd and "conds_std" in sd:
        mean = sd["conds_mean"].float()
        std = sd["conds_std"].float()
        print("Found conds_mean and conds_std inside the classifier checkpoint.")
    else:
        if latent_stats_path is None:
            raise KeyError(
                "The classifier checkpoint does not contain conds_mean/conds_std, "
                "and --latent-stats was not provided.\n\n"
                "Provide the DiffAE latent statistics file, usually:\n"
                "  /nas-ctm01/homes/dacordeiro/diffae/checkpoints/ffhq256_autoenc/latent.pkl"
            )

        latent_stats_path = Path(latent_stats_path).resolve()
        if not latent_stats_path.exists():
            raise FileNotFoundError(f"Latent stats file not found:\n  {latent_stats_path}")

        print(f"Loading latent stats:\n  {latent_stats_path}")
        stats = torch.load(str(latent_stats_path), map_location="cpu")
        mean = stats["conds_mean"].float()
        std = stats["conds_std"].float()

    mean = mean.reshape(1, -1)
    std = std.reshape(1, -1)
    std = torch.where(std == 0, torch.ones_like(std), std)

    if weight.shape != (40, 512):
        raise ValueError(f"Expected classifier weight shape [40, 512], got {tuple(weight.shape)}")
    if bias.shape != (40,):
        raise ValueError(f"Expected classifier bias shape [40], got {tuple(bias.shape)}")
    if mean.shape != (1, 512):
        raise ValueError(f"Expected conds_mean shape [1, 512], got {tuple(mean.shape)}")
    if std.shape != (1, 512):
        raise ValueError(f"Expected conds_std shape [1, 512], got {tuple(std.shape)}")

    return weight, bias, mean, std


def compute_selected_attribute_scores(
    split_path: Path,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    attr_names,
    batch_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Computes selected attribute logits and probabilities for one split.

    Returns:
        {
            "Smiling": {"logit": [N], "prob": [N]},
            "Young":   {"logit": [N], "prob": [N]},
        }
    """
    split_path = Path(split_path).resolve()

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found:\n  {split_path}")

    z = np.load(str(split_path), mmap_mode="r")

    if z.ndim != 2:
        raise ValueError(f"Expected split array shape [N, 512], got {z.shape} at {split_path}")
    if z.shape[1] != 512:
        raise ValueError(f"Expected 512-D DiffAE z_sem, got {z.shape} at {split_path}")

    n = z.shape[0]

    attr_ids = [ATTR_TO_ID[name] for name in attr_names]

    outputs = {
        name: {
            "logit": np.empty((n,), dtype=np.float32),
            "prob": np.empty((n,), dtype=np.float32),
        }
        for name in attr_names
    }

    weight = weight.to(device)
    bias = bias.to(device)
    mean = mean.to(device)
    std = std.to(device)

    print(f"\nProcessing split:\n  {split_path}")
    print(f"Shape: {z.shape}")
    print(f"Attributes: {', '.join(attr_names)}")

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"Scoring {split_path.stem}"):
            end = min(start + batch_size, n)

            z_batch = torch.from_numpy(np.asarray(z[start:end])).float().to(device)

            # Correct classifier preprocessing:
            # raw DiffAE z_sem -> authors/classifier latent normalization.
            z_norm = (z_batch - mean) / std

            logits = F.linear(z_norm, weight, bias)
            selected_logits = logits[:, attr_ids]
            selected_probs = torch.sigmoid(selected_logits)

            selected_logits_np = selected_logits.detach().cpu().numpy().astype(np.float32)
            selected_probs_np = selected_probs.detach().cpu().numpy().astype(np.float32)

            for local_attr_i, attr_name in enumerate(attr_names):
                outputs[attr_name]["logit"][start:end] = selected_logits_np[:, local_attr_i]
                outputs[attr_name]["prob"][start:end] = selected_probs_np[:, local_attr_i]

    return outputs


def save_scores_for_split(
    split_path: Path,
    scores: Dict[str, Dict[str, np.ndarray]],
    overwrite: bool,
) -> Dict[str, Path]:
    """
    Saves:
      ffhq256_diffae_zsem_train_smiling_logit.npy
      ffhq256_diffae_zsem_train_smiling_prob.npy
      ffhq256_diffae_zsem_train_young_logit.npy
      ffhq256_diffae_zsem_train_young_prob.npy

    Also saves:
      ffhq256_diffae_zsem_train_attribute_scores_smiling_young.npz
    """
    split_path = Path(split_path).resolve()
    out_dir = split_path.parent
    split_stem = split_path.stem

    saved = {}

    for attr_name, attr_scores in scores.items():
        attr_lower = attr_name.lower()

        logit_path = out_dir / f"{split_stem}_{attr_lower}_logit.npy"
        prob_path = out_dir / f"{split_stem}_{attr_lower}_prob.npy"

        if not overwrite:
            for p in [logit_path, prob_path]:
                if p.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite existing file:\n  {p}\n"
                        "Use --overwrite if you intentionally want to replace it."
                    )

        np.save(logit_path, attr_scores["logit"].astype(np.float32))
        np.save(prob_path, attr_scores["prob"].astype(np.float32))

        saved[f"{attr_lower}_logit"] = logit_path
        saved[f"{attr_lower}_prob"] = prob_path

    npz_path = out_dir / f"{split_stem}_attribute_scores_smiling_young.npz"
    if npz_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file:\n  {npz_path}\n"
            "Use --overwrite if you intentionally want to replace it."
        )

    np.savez_compressed(
        npz_path,
        smiling_logit=scores["Smiling"]["logit"].astype(np.float32),
        smiling_prob=scores["Smiling"]["prob"].astype(np.float32),
        young_logit=scores["Young"]["logit"].astype(np.float32),
        young_prob=scores["Young"]["prob"].astype(np.float32),
        attributes=np.array(["Smiling", "Young"]),
        note=np.array([
            "Logits are the recommended scores for deterministic sorting. "
            "Probabilities are sigmoid(logit), saved only for inspection."
        ]),
    )
    saved["combined_npz"] = npz_path

    return saved


def append_summary_for_split(
    lines,
    split_name: str,
    split_path: Path,
    scores: Dict[str, Dict[str, np.ndarray]],
    saved_paths: Dict[str, Path],
):
    lines.append("=" * 100)
    lines.append(f"SPLIT: {split_name}")
    lines.append("=" * 100)
    lines.append(f"Input split file: {split_path}")
    lines.append(f"Number of embeddings: {len(next(iter(scores.values()))['logit']):,}")
    lines.append("")

    for attr_name in ["Smiling", "Young"]:
        logits = scores[attr_name]["logit"]
        probs = scores[attr_name]["prob"]

        prevalence_logit_positive = float((logits > 0).mean())
        lines.append(f"{attr_name}")
        lines.append("-" * len(attr_name))
        lines.append(f"  logit file: {saved_paths[attr_name.lower() + '_logit']}")
        lines.append(f"  prob file:  {saved_paths[attr_name.lower() + '_prob']}")
        lines.append(f"  logit mean: {float(logits.mean()):.6f}")
        lines.append(f"  logit std:  {float(logits.std()):.6f}")
        lines.append(f"  logit min:  {float(logits.min()):.6f}")
        lines.append(f"  logit max:  {float(logits.max()):.6f}")
        lines.append(f"  prob mean:  {float(probs.mean()):.6f}")
        lines.append(f"  prob std:   {float(probs.std()):.6f}")
        lines.append(f"  predicted-positive prevalence, logit > 0: {100.0 * prevalence_logit_positive:.2f}%")
        lines.append("")

    lines.append(f"Combined NPZ: {saved_paths['combined_npz']}")
    lines.append("")


def infer_split_paths(base_path: Path):
    """
    Given:
        /.../ffhq256_diffae_zsem.npy

    Returns:
        train -> /.../ffhq256_diffae_zsem_train.npy
        val   -> /.../ffhq256_diffae_zsem_val.npy
        test  -> /.../ffhq256_diffae_zsem_test.npy

    Also works if the provided base path already contains _train/_val/_test.
    """
    base_path = Path(base_path).resolve()
    parent = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_val", "").replace("_test", "")

    return {
        "train": parent / f"{stem}_train.npy",
        "val": parent / f"{stem}_val.npy",
        "test": parent / f"{stem}_test.npy",
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute DiffAE released classifier scores for Smiling and Young "
            "on your train/val/test raw FFHQ256 z_sem splits."
        )
    )

    parser.add_argument(
        "--base-zsem",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy",
        help=(
            "Base DiffAE z_sem path. The script will infer _train/_val/_test split files "
            "from this path."
        ),
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
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size for classifier scoring.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--use-ema-classifier",
        action="store_true",
        help=(
            "Use ema_classifier weights instead of classifier weights. "
            "Default uses classifier weights, matching the official manipulation notebook style."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing score files.",
    )

    args = parser.parse_args()

    base_zsem = Path(args.base_zsem).resolve()
    classifier_ckpt = Path(args.classifier_ckpt).resolve()
    latent_stats = Path(args.latent_stats).resolve() if args.latent_stats is not None else None
    device = torch.device(args.device)

    print("=" * 100)
    print("Precomputing DiffAE attribute scores: Smiling and Young")
    print("=" * 100)
    print(f"Base z_sem:          {base_zsem}")
    print(f"Classifier ckpt:     {classifier_ckpt}")
    print(f"Latent stats:        {latent_stats}")
    print(f"Device:              {device}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Use EMA classifier:  {args.use_ema_classifier}")
    print(f"Overwrite:           {args.overwrite}")
    print("=" * 100)

    split_paths = infer_split_paths(base_zsem)

    for split_name, split_path in split_paths.items():
        if not split_path.exists():
            raise FileNotFoundError(
                f"Could not find inferred {split_name} split:\n  {split_path}\n\n"
                "Check --base-zsem or verify that your split files exist."
            )

    weight, bias, mean, std = load_classifier_tensors(
        ckpt_path=classifier_ckpt,
        latent_stats_path=latent_stats,
        use_ema_classifier=args.use_ema_classifier,
    )

    attr_names = ["Smiling", "Young"]
    summary_lines = []

    summary_lines.append("DIFFAE ATTRIBUTE SCORE PRECOMPUTATION SUMMARY")
    summary_lines.append("=" * 100)
    summary_lines.append("Scores computed: Smiling and Young")
    summary_lines.append("Recommended score for deterministic sorting: *_logit.npy")
    summary_lines.append("Probability files are sigmoid(logit), saved only for inspection.")
    summary_lines.append("Input assumption: raw DiffAE FFHQ256 z_sem, shape [N, 512].")
    summary_lines.append("Classifier preprocessing: z_norm = (z_raw - classifier_mean) / classifier_std.")
    summary_lines.append("Do not use your own train_mean/train_std for these classifier scores.")
    summary_lines.append("")
    summary_lines.append(f"Base z_sem path:      {base_zsem}")
    summary_lines.append(f"Classifier checkpoint:{classifier_ckpt}")
    summary_lines.append(f"Latent stats path:    {latent_stats}")
    summary_lines.append(f"Classifier version:   {'EMA classifier' if args.use_ema_classifier else 'raw classifier'}")
    summary_lines.append("")

    for split_name in ["train", "val", "test"]:
        split_path = split_paths[split_name]

        scores = compute_selected_attribute_scores(
            split_path=split_path,
            weight=weight,
            bias=bias,
            mean=mean,
            std=std,
            attr_names=attr_names,
            batch_size=args.batch_size,
            device=device,
        )

        saved_paths = save_scores_for_split(
            split_path=split_path,
            scores=scores,
            overwrite=args.overwrite,
        )

        print(f"\nSaved {split_name} score files:")
        for key, path in saved_paths.items():
            print(f"  {key}: {path}")

        append_summary_for_split(
            lines=summary_lines,
            split_name=split_name,
            split_path=split_path,
            scores=scores,
            saved_paths=saved_paths,
        )

    summary_path = base_zsem.parent / "ffhq256_diffae_zsem_attribute_scores_smiling_young_summary.txt"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing summary:\n  {summary_path}\n"
            "Use --overwrite if you intentionally want to replace it."
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n" + "=" * 100)
    print(f"Done. Summary saved to:\n  {summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
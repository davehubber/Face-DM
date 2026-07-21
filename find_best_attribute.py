import argparse
from pathlib import Path

import numpy as np
import torch


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


def load_classifier(classifier_ckpt: Path, device: torch.device):
    classifier_ckpt = Path(classifier_ckpt).resolve()

    if not classifier_ckpt.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_ckpt}")

    ckpt = torch.load(classifier_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    def get_tensor(*keys):
        for key in keys:
            if key in state_dict:
                return state_dict[key]
        return None

    weight = get_tensor("ema_classifier.weight", "classifier.weight")
    bias = get_tensor("ema_classifier.bias", "classifier.bias")
    conds_mean = get_tensor("conds_mean")
    conds_std = get_tensor("conds_std")

    if weight is None or bias is None:
        keys = [k for k in state_dict.keys() if "classifier" in k]
        raise KeyError(f"Could not find classifier weights. Classifier keys found: {keys[:50]}")

    if conds_mean is None or conds_std is None:
        keys = [k for k in state_dict.keys() if "cond" in k or "classifier" in k]
        raise KeyError(
            "This script expects conds_mean and conds_std inside the classifier checkpoint. "
            f"Relevant keys found: {keys[:50]}"
        )

    classifier = torch.nn.Linear(weight.shape[1], weight.shape[0])
    classifier.weight.data.copy_(weight.float())
    classifier.bias.data.copy_(bias.float())
    classifier = classifier.to(device)
    classifier.eval()

    conds_mean = torch.as_tensor(conds_mean).float().reshape(1, -1).to(device)
    conds_std = torch.as_tensor(conds_std).float().reshape(1, -1).to(device)

    return classifier, conds_mean, conds_std


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--classifier-ckpt",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/ffhq256_autoenc_cls/last.ckpt",
    )
    parser.add_argument(
        "--zsem-pairs",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem_test_pairs.npy",
    )
    parser.add_argument(
        "--out-txt",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/attribute_probability_differences_sorted.txt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    zsem_pairs_path = Path(args.zsem_pairs).resolve()
    out_txt_path = Path(args.out_txt).resolve()

    if not zsem_pairs_path.exists():
        raise FileNotFoundError(f"z_sem test pairs file not found: {zsem_pairs_path}")

    zsem_pairs = np.load(zsem_pairs_path).astype(np.float32)

    if zsem_pairs.ndim != 3 or zsem_pairs.shape[1] != 2:
        raise ValueError(f"Expected z_sem pairs with shape [N, 2, D], got {zsem_pairs.shape}")

    num_pairs = zsem_pairs.shape[0]
    z_dim = zsem_pairs.shape[2]

    print(f"Loaded z_sem pairs: {zsem_pairs_path}")
    print(f"Shape: {zsem_pairs.shape}")
    print(f"Number of test pairs: {num_pairs}")
    print(f"z_sem dimension: {z_dim}")

    classifier, conds_mean, conds_std = load_classifier(
        classifier_ckpt=Path(args.classifier_ckpt),
        device=device,
    )

    if classifier.in_features != z_dim:
        raise ValueError(
            f"Classifier expects z_sem dimension {classifier.in_features}, "
            f"but test pairs have dimension {z_dim}"
        )

    all_probs_A = []
    all_probs_B = []

    with torch.no_grad():
        for start in range(0, num_pairs, args.batch_size):
            end = min(start + args.batch_size, num_pairs)

            batch = torch.from_numpy(zsem_pairs[start:end]).float().to(device)  # [B, 2, 512]

            z_A = batch[:, 0, :]
            z_B = batch[:, 1, :]

            z_A_norm = (z_A - conds_mean) / conds_std
            z_B_norm = (z_B - conds_mean) / conds_std

            logits_A = classifier(z_A_norm)
            logits_B = classifier(z_B_norm)

            probs_A = torch.sigmoid(logits_A)
            probs_B = torch.sigmoid(logits_B)

            all_probs_A.append(probs_A.cpu())
            all_probs_B.append(probs_B.cpu())

    probs_A = torch.cat(all_probs_A, dim=0)  # [N, 40]
    probs_B = torch.cat(all_probs_B, dim=0)  # [N, 40]

    abs_diffs = torch.abs(probs_A - probs_B)  # [N, 40]

    mean_abs_diff = abs_diffs.mean(dim=0)
    median_abs_diff = abs_diffs.median(dim=0).values
    max_abs_diff = abs_diffs.max(dim=0).values
    std_abs_diff = abs_diffs.std(dim=0)

    sorted_indices = torch.argsort(mean_abs_diff, descending=True).tolist()

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("DiffAE 40-attribute classifier probability-difference analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"z_sem pairs file: {zsem_pairs_path}\n")
        f.write(f"classifier checkpoint: {Path(args.classifier_ckpt).resolve()}\n")
        f.write(f"number of test pairs: {num_pairs}\n")
        f.write(f"z_sem pair shape: {tuple(zsem_pairs.shape)}\n\n")

        f.write("Criterion used for sorting:\n")
        f.write("  mean_abs_probability_difference = mean(abs(P(attribute|A) - P(attribute|B))) over all test pairs\n\n")

        f.write("Sorted attributes, descending by mean absolute probability difference\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Rank':>4}  "
            f"{'Attr_ID':>7}  "
            f"{'Attribute':<24}  "
            f"{'MeanAbsDiff':>12}  "
            f"{'MedianAbsDiff':>14}  "
            f"{'MaxAbsDiff':>11}  "
            f"{'StdAbsDiff':>11}\n"
        )
        f.write("-" * 80 + "\n")

        for rank, attr_idx in enumerate(sorted_indices, start=1):
            f.write(
                f"{rank:>4}  "
                f"{attr_idx:>7}  "
                f"{ATTRIBUTES[attr_idx]:<24}  "
                f"{float(mean_abs_diff[attr_idx]):>12.6f}  "
                f"{float(median_abs_diff[attr_idx]):>14.6f}  "
                f"{float(max_abs_diff[attr_idx]):>11.6f}  "
                f"{float(std_abs_diff[attr_idx]):>11.6f}\n"
            )

        f.write("\n\nMost different attribute overall:\n")
        best_idx = sorted_indices[0]
        f.write(f"  Rank: 1\n")
        f.write(f"  Attribute ID: {best_idx}\n")
        f.write(f"  Attribute: {ATTRIBUTES[best_idx]}\n")
        f.write(f"  Mean absolute probability difference: {float(mean_abs_diff[best_idx]):.6f}\n")
        f.write(f"  Median absolute probability difference: {float(median_abs_diff[best_idx]):.6f}\n")
        f.write(f"  Maximum absolute probability difference in a pair: {float(max_abs_diff[best_idx]):.6f}\n")

    print("\nTop 10 attributes by mean absolute probability difference")
    print("-" * 80)

    for rank, attr_idx in enumerate(sorted_indices[:10], start=1):
        print(
            f"{rank:>2}. {ATTRIBUTES[attr_idx]:<24} "
            f"mean={float(mean_abs_diff[attr_idx]):.6f} "
            f"median={float(median_abs_diff[attr_idx]):.6f} "
            f"max={float(max_abs_diff[attr_idx]):.6f}"
        )

    print(f"\nSaved full sorted report to: {out_txt_path}")


if __name__ == "__main__":
    main()
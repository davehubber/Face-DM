import argparse
import csv
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/smiling_train_probs/train_smiling_probabilities.csv",
        help="Path to the computed smiling probabilities CSV",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/smiling_train_probs/smiling_extremes_comparison.png",
        help="Output path for the stitched comparison image",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).resolve()
    out_path = Path(args.out_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Probabilities CSV not found at: {csv_path}\n"
            f"Please run your compute_smiling_scores.py script first."
        )

    min_prob = float("inf")
    max_prob = float("-inf")
    min_row = None
    max_row = None

    # 1. Parse the CSV to identify absolute min and max instances
    print(f"Scanning metrics from: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prob = float(row["smiling_probability"])
            if prob > max_prob:
                max_prob = prob
                max_row = row
            if prob < min_prob:
                min_prob = prob
                min_row = row

    if not max_row or not min_row:
        print("Error: No valid rows found in the target CSV file.")
        return

    print("\nExtremes Located:")
    print(
        f"  - Lowest Smiling Score:  {min_prob:.6f} ({min_row['smiling_percentage']}%) -> {min_row['filename']}"
    )
    print(
        f"  - Highest Smiling Score: {max_prob:.6f} ({max_row['smiling_percentage']}%) -> {max_row['filename']}"
    )

    path_min = Path(min_row["image_path"])
    path_max = Path(max_row["image_path"])

    if not path_min.exists() or not path_max.exists():
        raise FileNotFoundError(
            f"Image verification failed. Ensure source files exist:\n"
            f"  Min Path: {path_min}\n  Max Path: {path_max}"
        )

    # 2. Open images and prepare canvas
    img_min = Image.open(path_min).convert("RGB")
    img_max = Image.open(path_max).convert("RGB")

    w_min, h_min = img_min.size
    w_max, h_max = img_max.size

    # Side-by-side stitching setup: [ Lowest Score (Left) | Highest Score (Right) ]
    total_width = w_min + w_max
    max_height = max(h_min, h_max)

    canvas = Image.new("RGB", (total_width, max_height))
    canvas.paste(img_min, (0, 0))
    canvas.paste(img_max, (w_min, 0))

    # 3. Save result
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"\n[SUCCESS] Side-by-side grid exported to: {out_path}")


if __name__ == "__main__":
    main()
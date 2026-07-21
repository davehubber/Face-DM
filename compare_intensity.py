import argparse
import os
import numpy as np
import torch
from utils import get_data


class DatasetArgs:
    """A mock args class to satisfy the interface of get_data in utils.py."""

    def __init__(self, dataset_path, test_images=1000, batch_size=32):
        self.dataset_path = dataset_path
        self.test_images = test_images
        self.batch_size = batch_size
        self.train_samples_per_epoch = 30000  # Unused for validation partition
        self.num_workers = 4


def compute_pair_separations(dataset_path, num_pairs=1000):
    args = DatasetArgs(dataset_path=dataset_path, test_images=num_pairs)
    dataloader = get_data(args, partition="val")

    absolute_differences = []

    print(f"Processing pairs for: {dataset_path}...")
    for batch_idx, (bright_batch, dark_batch) in enumerate(dataloader):
        # Tensors are normalized to [-1, 1]. Unnormalize to [0, 255] for true pixel intensity.
        bright_unnorm = (bright_batch + 1.0) / 2.0 * 255.0
        dark_unnorm = (dark_batch + 1.0) / 2.0 * 255.0

        # Calculate mean intensity per image across channel, height, and width dims
        mean_bright = bright_unnorm.mean(dim=[1, 2, 3])
        mean_dark = dark_unnorm.mean(dim=[1, 2, 3])

        # Compute absolute difference for each pair in the batch
        batch_diffs = torch.abs(mean_bright - mean_dark)
        absolute_differences.extend(batch_diffs.cpu().tolist())

    return np.array(absolute_differences)


def main():
    parser = argparse.ArgumentParser(
        description="Compare mean pixel intensity separation between two paired validation datasets."
    )
    parser.add_argument(
        "--dataset_1", required=True, help="Path to the first image dataset folder"
    )
    parser.add_argument(
        "--dataset_2", required=True, help="Path to the second image dataset folder"
    )
    parser.add_argument(
        "--output_file",
        default="dataset_intensity_comparison.txt",
        help="Path to save the summary report",
    )
    parser.add_argument(
        "--num_pairs",
        default=1000,
        type=int,
        help="Number of validation pairs (default: 1000)",
    )
    args = parser.parse_args()

    # Calculate separations for both datasets
    diffs_1 = compute_pair_separations(args.dataset_1, args.num_pairs)
    diffs_2 = compute_pair_separations(args.dataset_2, args.num_pairs)

    mean_sep_1 = np.mean(diffs_1)
    mean_sep_2 = np.mean(diffs_2)

    # Determine which dataset is more separated
    if mean_sep_1 > mean_sep_2:
        winner = f"Dataset 1 ({args.dataset_1})"
        margin = mean_sep_1 - mean_sep_2
    elif mean_sep_2 > mean_sep_1:
        winner = f"Dataset 2 ({args.dataset_2})"
        margin = mean_sep_2 - mean_sep_1
    else:
        winner = "Neither (They are identical)"
        margin = 0.0

    # Build the report string
    report = (
        f"=== Mean Pixel Intensity Separation Report ===\n"
        f"Evaluated Pairs per Dataset: {args.num_pairs}\n\n"
        f"--- Dataset 1 Summary ---\n"
        f"Path: {args.dataset_1}\n"
        f"Average Pair Separation: {mean_sep_1:.4f}\n"
        f"Max Separation: {np.max(diffs_1):.4f}\n"
        f"Min Separation: {np.min(diffs_1):.4f}\n"
        f"Standard Deviation: {np.std(diffs_1):.4f}\n\n"
        f"--- Dataset 2 Summary ---\n"
        f"Path: {args.dataset_2}\n"
        f"Average Pair Separation: {mean_sep_2:.4f}\n"
        f"Max Separation: {np.max(diffs_2):.4f}\n"
        f"Min Separation: {np.min(diffs_2):.4f}\n"
        f"Standard Deviation: {np.std(diffs_2):.4f}\n\n"
        f"--- Conclusion ---\n"
        f"The 1000 pairs are more separated by mean pixel intensity in:\n"
        f"👉 {winner}\n"
        f"Margin of difference: {margin:.4f} intensity units (0-255 scale).\n"
    )

    # Print to console and write to text file
    print(f"\n{report}")
    with open(args.output_file, "w") as f:
        f.write(report)

    print(f"Results successfully saved to {args.output_file}")


if __name__ == "__main__":
    main()

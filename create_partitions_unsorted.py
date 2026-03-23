import argparse
import os
import random
from typing import List, Sequence, Tuple

import pandas as pd

Pair = Tuple[str, str]


def build_pairs(image_list: Sequence[str], rng: random.Random) -> List[Pair]:
    pairs: List[Pair] = []
    for img1 in image_list:
        for _ in range(5):
            img2 = rng.choice(image_list)
            while img1 == img2:
                img2 = rng.choice(image_list)
            pairs.append((img1, img2))
    return pairs


def pairs_to_rows(pairs: List[Pair], partition_name: str) -> List[dict]:
    rows: List[dict] = []
    for img1, img2 in pairs:
        rows.append({
            "partition": partition_name,
            "Image1": img1,
            "Image2": img2,
        })
    return rows


def generate_partition_csv(
    folder_path: str,
    output_csv: str = "partition_unsorted.csv",
    seed: int = 42,
    test_count: int = 1000,
):
    all_images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])

    if len(all_images) != 8189:
        print(f"Warning: Expected 8189 images, found {len(all_images)}.")

    rng = random.Random(seed)
    rng.shuffle(all_images)

    test_images = all_images[:test_count]
    train_images = all_images[test_count:]

    train_pairs = build_pairs(train_images, rng)
    test_pairs = build_pairs(test_images, rng)

    train_rows = pairs_to_rows(train_pairs, "train")
    test_rows = pairs_to_rows(test_pairs, "test")

    df = pd.DataFrame(train_rows + test_rows)
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved {output_csv} with {len(df)} total unsorted pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="/nas-ctm01/datasets/public/Oxford102Flowers/jpg")
    parser.add_argument("--output_csv", type=str, default="partition_unsorted.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_count", type=int, default=1000)
    args = parser.parse_args()

    generate_partition_csv(
        folder_path=args.folder_path,
        output_csv=args.output_csv,
        seed=args.seed,
        test_count=args.test_count,
    )
import argparse
import os
import random
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import torchvision
from PIL import Image

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


class BrightnessOrderer:
    def __init__(self, image_size: int = 256, batch_size: int = 32, device: str | None = None):
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
        ])

    def _load_image(self, folder_path: str, filename: str) -> torch.Tensor:
        image = Image.open(os.path.join(folder_path, filename)).convert("RGB")
        return self.transform(image)

    def order_partition(self, folder_path: str, pairs: Iterable[Pair], partition_name: str) -> List[dict]:
        pair_list = list(pairs)
        rows: List[dict] = []

        with torch.no_grad():
            for start in range(0, len(pair_list), self.batch_size):
                batch_pairs = pair_list[start:start + self.batch_size]
                batch_1 = torch.stack([self._load_image(folder_path, p1) for p1, _ in batch_pairs]).to(self.device)
                batch_2 = torch.stack([self._load_image(folder_path, p2) for _, p2 in batch_pairs]).to(self.device)

                mean_1 = batch_1.mean(dim=(1, 2, 3)).detach().cpu().tolist()
                mean_2 = batch_2.mean(dim=(1, 2, 3)).detach().cpu().tolist()

                for (img1, img2), b1, b2 in zip(batch_pairs, mean_1, mean_2):
                    # Brightest image must be Image1
                    if b2 > b1:
                        ordered_1, ordered_2 = img2, img1
                        ordered_b1, ordered_b2 = b2, b1
                        swapped = True
                    else:
                        ordered_1, ordered_2 = img1, img2
                        ordered_b1, ordered_b2 = b1, b2
                        swapped = False

                    rows.append({
                        "partition": partition_name,
                        "Image1": ordered_1,
                        "Image2": ordered_2,
                        "AvgPixelIntensity_Image1": ordered_b1,
                        "AvgPixelIntensity_Image2": ordered_b2,
                        "SwappedForBrightnessOrdering": swapped,
                    })

        return rows


def generate_partition_csv(
    folder_path: str,
    output_csv: str = "partition_brightness_img1_brightest.csv",
    seed: int = 42,
    test_count: int = 1000,
    image_size: int = 256,
    batch_size: int = 32,
    device: str | None = None,
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

    print("Computing brightness-based ordering for train pairs...")
    orderer = BrightnessOrderer(image_size=image_size, batch_size=batch_size, device=device)
    train_rows = orderer.order_partition(folder_path, train_pairs, "train")

    print("Computing brightness-based ordering for test pairs...")
    test_rows = orderer.order_partition(folder_path, test_pairs, "test")

    df = pd.DataFrame(train_rows + test_rows)
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved {output_csv} with {len(df)} total ordered pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="/nas-ctm01/datasets/public/Oxford102Flowers/jpg")
    parser.add_argument("--output_csv", type=str, default="partition_brightness_img1_brightest.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_count", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    generate_partition_csv(
        folder_path=args.folder_path,
        output_csv=args.output_csv,
        seed=args.seed,
        test_count=args.test_count,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )
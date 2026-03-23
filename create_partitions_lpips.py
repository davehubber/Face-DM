import argparse
import os
import random
from typing import Iterable, List, Sequence, Tuple

import lpips
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

class LpipsOrderer:
    def __init__(self, image_size: int = 256, batch_size: int = 32, device: str | None = None):
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.model = lpips.LPIPS(net="alex").to(self.device).eval()

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

                average_mix = 0.5 * batch_1 + 0.5 * batch_2
                lpips_1 = self.model(batch_1, average_mix).view(-1).detach().cpu().tolist()
                lpips_2 = self.model(batch_2, average_mix).view(-1).detach().cpu().tolist()

                for (img1, img2), dist1, dist2 in zip(batch_pairs, lpips_1, lpips_2):
                    if dist2 < dist1:
                        ordered_1, ordered_2 = img2, img1
                        ordered_d1, ordered_d2 = dist2, dist1
                        swapped = True
                    else:
                        ordered_1, ordered_2 = img1, img2
                        ordered_d1, ordered_d2 = dist1, dist2
                        swapped = False

                    rows.append({
                        "partition": partition_name,
                        "Image1": ordered_1,
                        "Image2": ordered_2,
                        "LPIPS_Image1_to_average": ordered_d1,
                        "LPIPS_Image2_to_average": ordered_d2,
                        "SwappedForLPIPSOrdering": swapped,
                    })

        return rows


def generate_partition_csv(
    folder_path: str,
    output_csv: str = "partition_lpips.csv",
    seed: int = 42,
    test_count: int = 1000,
    lpips_image_size: int = 256,
    lpips_batch_size: int = 32,
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

    print("Computing LPIPS-based canonical ordering for train pairs...")
    orderer = LpipsOrderer(image_size=lpips_image_size, batch_size=lpips_batch_size, device=device)
    train_rows = orderer.order_partition(folder_path, train_pairs, "train")

    print("Computing LPIPS-based canonical ordering for test pairs...")
    test_rows = orderer.order_partition(folder_path, test_pairs, "test")

    df = pd.DataFrame(train_rows + test_rows)
    df.to_csv(output_csv, index=False)
    print(f"Successfully saved {output_csv} with {len(df)} total ordered pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="/nas-ctm01/datasets/public/Oxford102Flowers/jpg")
    parser.add_argument("--output_csv", type=str, default="partition_lpips.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_count", type=int, default=1000)
    parser.add_argument("--lpips_image_size", type=int, default=256)
    parser.add_argument("--lpips_batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    generate_partition_csv(
        folder_path=args.folder_path,
        output_csv=args.output_csv,
        seed=args.seed,
        test_count=args.test_count,
        lpips_image_size=args.lpips_image_size,
        lpips_batch_size=args.lpips_batch_size,
        device=args.device,
    )

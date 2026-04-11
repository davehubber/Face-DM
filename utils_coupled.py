import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_SEED = 42
VAL_PAIR_SEED = 1234
DEFAULT_PAIRS_PER_IMAGE = 5


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat([
            torch.cat([i for i in images.cpu()], dim=-1),
        ], dim=-2).permute(1, 2, 0).cpu()
    )
    plt.show()


def to_uint8(images):
    images = (images.clamp(-1, 1) + 1) / 2
    return (images * 255).type(torch.uint8)


def save_images(images, obtained_images, original_images, added_images, path, input_images=None, **kwargs):
    if input_images is None:
        input_images = ((original_images.float() + added_images.float()) * 0.5).round().clamp(0, 255).type(torch.uint8)

    _, c, h, w = original_images.shape
    all_images = torch.stack(
        [original_images, added_images, input_images, images, obtained_images], dim=1
    ).view(-1, c, h, w)

    grid = torchvision.utils.make_grid(all_images, nrow=5, padding=2, pad_value=255, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy().astype(np.uint8)
    Image.fromarray(ndarr).save(path)


def list_image_files(dataset_path: str) -> List[str]:
    image_files = sorted(
        f for f in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_files:
        raise ValueError(f"No image files found in {dataset_path}")
    return image_files


def split_image_files(image_files: Sequence[str], train_fraction: float) -> Tuple[List[str], List[str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    shuffled = list(image_files)
    random.Random(SPLIT_SEED).shuffle(shuffled)

    split_idx = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
    return shuffled[:split_idx], shuffled[split_idx:]


def order_by_brightness(image_1: torch.Tensor, image_2: torch.Tensor):
    return (image_1, image_2) if image_1.mean() >= image_2.mean() else (image_2, image_1)


class OnTheFlyPairedDataset(Dataset):
    def __init__(self, dataset_path: str, image_files: Sequence[str], num_pairs: int, transform=None, augment: bool = False, deterministic: bool = False):
        if len(image_files) < 2:
            raise ValueError(f"Split must contain at least 2 images; found {len(image_files)}")
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")

        self.dataset_path = dataset_path
        self.image_files = list(image_files)
        self.num_pairs = int(num_pairs)
        self.transform = transform
        self.augment = augment
        self.deterministic = deterministic

        split_name = "train" if augment else "val"
        mode = "deterministic" if deterministic else "random"
        print(f"Size of {split_name}: {len(self.image_files)} source images, {self.num_pairs} {mode} pairs")

    def __len__(self):
        return self.num_pairs

    def _pair_from_rng(self, rng) -> Tuple[int, int]:
        n = len(self.image_files)
        idx1 = rng.randrange(n)
        idx2 = rng.randrange(n - 1)
        if idx2 >= idx1:
            idx2 += 1
        return idx1, idx2

    def _sample_pair_indices(self, index: int) -> Tuple[int, int]:
        if self.deterministic:
            return self._pair_from_rng(random.Random(VAL_PAIR_SEED + index))
        return self._pair_from_rng(random)

    def __getitem__(self, index):
        idx1, idx2 = self._sample_pair_indices(index)

        image_1 = Image.open(os.path.join(self.dataset_path, self.image_files[idx1])).convert("RGB")
        image_2 = Image.open(os.path.join(self.dataset_path, self.image_files[idx2])).convert("RGB")

        if self.augment:
            if random.random() < 0.5:
                image_1 = TF.hflip(image_1)
            if random.random() < 0.5:
                image_2 = TF.hflip(image_2)

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2


def _default_num_pairs(num_images: int) -> int:
    return max(1, int(num_images) * DEFAULT_PAIRS_PER_IMAGE)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data(args, partition):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    image_files = list_image_files(args.dataset_path)
    train_files, val_files = split_image_files(image_files, args.train_fraction)
    split_files = train_files if partition == "train" else val_files

    if partition == "train":
        num_pairs = args.train_samples_per_epoch or _default_num_pairs(len(split_files))
        dataset = OnTheFlyPairedDataset(
            dataset_path=args.dataset_path,
            image_files=split_files,
            num_pairs=num_pairs,
            transform=transforms,
            augment=True,
            deterministic=False,
        )
    else:
        num_pairs = args.val_samples or _default_num_pairs(len(split_files))
        dataset = OnTheFlyPairedDataset(
            dataset_path=args.dataset_path,
            image_files=split_files,
            num_pairs=num_pairs,
            transform=transforms,
            augment=False,
            deterministic=True,
        )

    configured_workers = getattr(args, "num_workers", None)
    num_workers = min(os.cpu_count() or 1, 8) if configured_workers is None else max(0, int(configured_workers))

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        drop_last=False,
    )


def setup_logging(run_name):
    base_dir = os.path.join("experiments", run_name)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "train_fixed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "eval"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "one_shot"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    return base_dir

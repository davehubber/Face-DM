import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat([
            torch.cat([i for i in images.cpu()], dim=-1),
        ], dim=-2).permute(1, 2, 0).cpu()
    )
    plt.show()



def to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = (images.clamp(-1, 1) + 1) / 2
    return (images * 255).round().type(torch.uint8)



def stack_pair_state(image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
    return torch.cat([image_1, image_2], dim=1)



def split_pair_state(state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if state.shape[1] != 6:
        raise ValueError(f"Expected a 6-channel coupled state, got shape {tuple(state.shape)}")
    return state[:, :3], state[:, 3:]



def duplicate_average_from_pair(image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
    average_image = 0.5 * (image_1 + image_2)
    return torch.cat([average_image, average_image], dim=1)



def duplicate_average_from_state(state: torch.Tensor) -> torch.Tensor:
    image_1, image_2 = split_pair_state(state)
    return duplicate_average_from_pair(image_1, image_2)



def align_prediction_to_targets(
    predicted_state: torch.Tensor,
    target_image_1: torch.Tensor,
    target_image_2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_image_1, predicted_image_2 = split_pair_state(predicted_state)

    loss_12 = (
        (predicted_image_1 - target_image_1).abs().flatten(1).mean(1)
        + (predicted_image_2 - target_image_2).abs().flatten(1).mean(1)
    )
    loss_21 = (
        (predicted_image_1 - target_image_2).abs().flatten(1).mean(1)
        + (predicted_image_2 - target_image_1).abs().flatten(1).mean(1)
    )

    swap_mask = loss_21 < loss_12
    swap_mask_image = swap_mask[:, None, None, None]

    aligned_image_1 = torch.where(swap_mask_image, predicted_image_2, predicted_image_1)
    aligned_image_2 = torch.where(swap_mask_image, predicted_image_1, predicted_image_2)
    return aligned_image_1, aligned_image_2, swap_mask



def save_images(
    images,
    obtained_images,
    original_images,
    added_images,
    path,
    input_images=None,
    **kwargs,
):
    if input_images is None:
        input_images = ((original_images.float() + added_images.float()) * 0.5).round().clamp(0, 255).type(torch.uint8)

    _, c, h, w = original_images.shape
    all_images = torch.stack(
        [original_images, added_images, input_images, images, obtained_images], dim=1
    ).view(-1, c, h, w)

    grid = torchvision.utils.make_grid(all_images, nrow=5, padding=2, pad_value=255, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy().astype(np.uint8)
    Image.fromarray(ndarr).save(path)


class CSVPairedDataset(Dataset):
    def __init__(self, dataset_path, partition, csv_file, transform=None):
        self.dataset_path = dataset_path
        self.partition = partition

        df = pd.read_csv(csv_file, sep=",")
        if "partition" in df.columns:
            df = df[df["partition"] == partition]

        self.image_paths_1 = np.asarray(df["Image1"].values)
        self.image_paths_2 = np.asarray(df["Image2"].values)
        self.transform = transform

        print(f"Size of {partition}: {len(self.image_paths_1)}")

    def __getitem__(self, index):
        x1 = Image.open(os.path.join(self.dataset_path, self.image_paths_1[index])).convert("RGB")
        x2 = Image.open(os.path.join(self.dataset_path, self.image_paths_2[index])).convert("RGB")

        if self.partition == "train":
            if random.random() < 0.5:
                x1 = TF.hflip(x1)
            if random.random() < 0.5:
                x2 = TF.hflip(x2)

        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        return x1, x2

    def __len__(self):
        return len(self.image_paths_1)



def list_image_files(dataset_path: str) -> List[str]:
    image_files = sorted(
        f
        for f in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, f)) and f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not image_files:
        raise ValueError(f"No image files found in {dataset_path}")
    return image_files



def split_image_files(image_files: Sequence[str], train_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    rng = random.Random(seed)
    shuffled = list(image_files)
    rng.shuffle(shuffled)

    split_idx = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_fraction))))
    train_files = shuffled[:split_idx]
    test_files = shuffled[split_idx:]
    return train_files, test_files


class OnTheFlyPairedDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        image_files: Sequence[str],
        partition: str,
        num_pairs: int,
        transform=None,
        base_seed: int = 42,
        augment: bool = False,
        deterministic_pairs: bool = False,
    ):
        if len(image_files) < 2:
            raise ValueError(f"{partition} split must contain at least 2 images; found {len(image_files)}")
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")

        self.dataset_path = dataset_path
        self.image_files = list(image_files)
        self.partition = partition
        self.num_pairs = int(num_pairs)
        self.transform = transform
        self.base_seed = int(base_seed)
        self.augment = augment
        self.deterministic_pairs = deterministic_pairs

        print(
            f"Size of {partition}: {len(self.image_files)} source images, {self.num_pairs} sampled pairs per epoch/pass"
        )

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
        if self.deterministic_pairs:
            rng = random.Random(self.base_seed + index)
            return self._pair_from_rng(rng)
        return self._pair_from_rng(random)

    def __getitem__(self, index):
        idx1, idx2 = self._sample_pair_indices(index)

        x1 = Image.open(os.path.join(self.dataset_path, self.image_files[idx1])).convert("RGB")
        x2 = Image.open(os.path.join(self.dataset_path, self.image_files[idx2])).convert("RGB")

        if self.augment:
            if random.random() < 0.5:
                x1 = TF.hflip(x1)
            if random.random() < 0.5:
                x2 = TF.hflip(x2)

        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        return x1, x2



def _default_num_pairs(num_images: int, pairs_per_image: int) -> int:
    return max(1, int(num_images) * int(pairs_per_image))



def _build_dataset(args, partition: str, transform):
    partition_file = getattr(args, "partition_file", None)
    use_dynamic_pairs = getattr(args, "use_dynamic_pairs", False)

    if partition_file and not use_dynamic_pairs:
        return CSVPairedDataset(args.dataset_path, partition, partition_file, transform=transform)

    image_files = list_image_files(args.dataset_path)
    train_files, test_files = split_image_files(
        image_files=image_files,
        train_fraction=getattr(args, "train_fraction", 0.8),
        seed=getattr(args, "split_seed", 42),
    )

    split_files = train_files if partition == "train" else test_files
    default_pairs_per_image = (
        getattr(args, "train_pairs_per_image", 5)
        if partition == "train"
        else getattr(args, "test_pairs_per_image", 5)
    )

    explicit_num_pairs = (
        getattr(args, "train_samples_per_epoch", None)
        if partition == "train"
        else getattr(args, "test_samples", None)
    )
    num_pairs = explicit_num_pairs or _default_num_pairs(len(split_files), default_pairs_per_image)

    return OnTheFlyPairedDataset(
        dataset_path=args.dataset_path,
        image_files=split_files,
        partition=partition,
        num_pairs=num_pairs,
        transform=transform,
        base_seed=getattr(args, "pair_seed", 1234 if partition == "train" else 4321),
        augment=(partition == "train"),
        deterministic_pairs=(partition != "train"),
    )



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

    dataset = _build_dataset(args, partition, transform=transforms)

    configured_workers = getattr(args, "num_workers", None)
    num_workers = min(os.cpu_count() or 1, 8) if configured_workers is None else max(0, int(configured_workers))

    shuffle = (partition == "train") and isinstance(dataset, CSVPairedDataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        drop_last=False,
    )
    return dataloader



def setup_logging(run_name):
    base_dir = os.path.join("experiments", run_name)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "train_fixed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "eval"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "one_shot"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    return base_dir

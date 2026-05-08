import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

VAL_PAIR_SEED = 1234
_STATS_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

def setup_logging(run_name: str) -> str:
    base_dir = os.path.join("experiments", run_name)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "decode"), exist_ok=True)
    return base_dir

def load_dataset_stats(split_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if split_path in _STATS_CACHE:
        return _STATS_CACHE[split_path]

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Could not find semantic split file: {split_path}")

    pack = torch.load(split_path, map_location="cpu")
    embeddings = pack["z_sem"].to(torch.float32)
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0, unbiased=False).clamp_min(1e-6)

    _STATS_CACHE[split_path] = (mean, std)
    return mean, std

class SemanticPairsDataset(Dataset):
    def __init__(self, dataset_root: str, split: str, num_pairs: int, n_components: int = 256, deterministic: bool = False):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")

        orig_path = os.path.join(dataset_root, "semantic", f"{split}_zsem.pt")
        pca_path = os.path.join(dataset_root, "semantic", f"{split}_zsem_pca{n_components}.pt")
        
        if not os.path.exists(orig_path):
            raise FileNotFoundError(f"Missing original file: {orig_path}")
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"Missing PCA file: {pca_path}")

        orig_pack = torch.load(orig_path, map_location="cpu")
        pca_pack = torch.load(pca_path, map_location="cpu")

        # Load stats for z-scoring
        orig_mean, orig_std = load_dataset_stats(orig_path)
        pca_mean, pca_std = load_dataset_stats(pca_path)

        # Normalize both spaces
        self.embeddings_orig = (orig_pack["z_sem"].to(torch.float32) - orig_mean) / orig_std
        self.embeddings_pca = (pca_pack["z_sem"].to(torch.float32) - pca_mean) / pca_std

        self.sample_ids: List[str] = list(orig_pack["sample_ids"])
        self.source_paths: List[str] = list(orig_pack["source_paths"])
        self.relative_paths: List[str] = list(orig_pack.get("relative_paths", [""] * len(self.sample_ids)))

        self.num_pairs = int(num_pairs)
        self.deterministic = deterministic

        mode = "deterministic" if deterministic else "random"
        print(f"Loaded {split} split: {len(self.embeddings_orig)} embeddings, {self.num_pairs} {mode} pairs")

    def __len__(self) -> int:
        return self.num_pairs

    def _pair_from_rng(self, rng: random.Random) -> Tuple[int, int]:
        n = len(self.embeddings_orig)
        idx1 = rng.randrange(n)
        idx2 = rng.randrange(n - 1)
        if idx2 >= idx1:
            idx2 += 1
        return idx1, idx2

    def _sample_pair_indices(self, index: int) -> Tuple[int, int]:
        if self.deterministic:
            return self._pair_from_rng(random.Random(VAL_PAIR_SEED + index))
        return self._pair_from_rng(random)

    def __getitem__(self, index: int) -> Dict:
        idx_A, idx_B = self._sample_pair_indices(index)

        # Build x_0: Concatenated 256-dimensional PCA embeddings -> 512 dimensions total
        pca_A = self.embeddings_pca[idx_A]
        pca_B = self.embeddings_pca[idx_B]
        x_0 = torch.cat([pca_A, pca_B], dim=-1)

        # Build x_T: Averaged 512-dimensional original embeddings -> 512 dimensions total
        orig_A = self.embeddings_orig[idx_A]
        orig_B = self.embeddings_orig[idx_B]
        x_T = (orig_A + orig_B) / 2.0

        return {
            "x_0": x_0,
            "x_T": x_T,
            "pca_A": pca_A,
            "pca_B": pca_B,
            "orig_A": orig_A,
            "orig_B": orig_B,
            "idx_A": idx_A,
            "idx_B": idx_B,
            "sample_id_A": self.sample_ids[idx_A],
            "sample_id_B": self.sample_ids[idx_B],
            "source_path_A": self.source_paths[idx_A],
            "source_path_B": self.source_paths[idx_B],
        }

def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_semantic_pairs(batch: List[Dict]) -> Dict:
    return {
        "x_0": torch.stack([item["x_0"] for item in batch], dim=0),
        "x_T": torch.stack([item["x_T"] for item in batch], dim=0),
        "pca_A": torch.stack([item["pca_A"] for item in batch], dim=0),
        "pca_B": torch.stack([item["pca_B"] for item in batch], dim=0),
        "idx_A": torch.tensor([item["idx_A"] for item in batch], dtype=torch.long),
        "idx_B": torch.tensor([item["idx_B"] for item in batch], dtype=torch.long),
        "sample_id_A": [item["sample_id_A"] for item in batch],
        "sample_id_B": [item["sample_id_B"] for item in batch],
        "source_path_A": [item["source_path_A"] for item in batch],
        "source_path_B": [item["source_path_B"] for item in batch],
    }

def get_data(args, partition: str):
    if partition not in {"train", "val"}:
        raise ValueError(f"Unsupported partition: {partition}")

    dataset = SemanticPairsDataset(
        dataset_root=args.dataset_root,
        split=partition,
        num_pairs=args.train_samples_per_epoch if partition == "train" else args.val_samples,
        n_components=args.n_components,
        deterministic=(partition == "val"),
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
        collate_fn=collate_semantic_pairs,
    )
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

VAL_PAIR_SEED = 1234
_ZSCORE_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def setup_logging(run_name: str) -> str:
    base_dir = os.path.join("experiments", run_name)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "decode"), exist_ok=True)
    return base_dir


def load_zscore_stats(dataset_root: str) -> Tuple[torch.Tensor, torch.Tensor]:
    cached = _ZSCORE_CACHE.get(dataset_root)
    if cached is not None:
        return cached

    train_split_path = os.path.join(dataset_root, "semantic", "train_zsem.pt")
    if not os.path.exists(train_split_path):
        raise FileNotFoundError(f"Could not find semantic train split file: {train_split_path}")

    train_pack = torch.load(train_split_path, map_location="cpu")
    train_embeddings = train_pack["z_sem"].to(torch.float32)
    mean = train_embeddings.mean(dim=0)
    std = train_embeddings.std(dim=0, unbiased=False).clamp_min(1e-6)

    _ZSCORE_CACHE[dataset_root] = (mean, std)
    return mean, std


class SemanticPairsDataset(Dataset):
    def __init__(self, dataset_root: str, split: str, num_pairs: int, deterministic: bool = False):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")

        split_path = os.path.join(dataset_root, "semantic", f"{split}_zsem.pt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Could not find semantic split file: {split_path}")

        pack = torch.load(split_path, map_location="cpu")
        self.mean, self.std = load_zscore_stats(dataset_root)

        self.raw_embeddings = pack["z_sem"].to(torch.float32)
        self.embeddings = (self.raw_embeddings - self.mean) / self.std

        self.sample_ids: List[str] = list(pack["sample_ids"])
        self.source_paths: List[str] = list(pack["source_paths"])
        self.relative_paths: List[str] = list(pack.get("relative_paths", [""] * len(self.sample_ids)))

        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings tensor, got shape {tuple(self.embeddings.shape)}")
        if len(self.embeddings) < 2:
            raise ValueError(f"Split must contain at least 2 embeddings; found {len(self.embeddings)}")

        self.num_pairs = int(num_pairs)
        self.deterministic = deterministic

        mode = "deterministic" if deterministic else "random"
        print(f"Loaded {split} split: {len(self.embeddings)} embeddings, {self.num_pairs} {mode} pairs")

    def __len__(self) -> int:
        return self.num_pairs

    def _pair_from_rng(self, rng: random.Random) -> Tuple[int, int]:
        n = len(self.embeddings)
        idx1 = rng.randrange(n)
        idx2 = rng.randrange(n - 1)
        if idx2 >= idx1:
            idx2 += 1
        return idx1, idx2

    def _sample_pair_indices(self, index: int) -> Tuple[int, int]:
        if self.deterministic:
            return self._pair_from_rng(random.Random(VAL_PAIR_SEED + index))
        return self._pair_from_rng(random)

    def _ordered_pair(self, idx1: int, idx2: int) -> Dict:
        emb1 = self.embeddings[idx1]
        emb2 = self.embeddings[idx2]

        raw_emb1 = self.raw_embeddings[idx1]
        raw_emb2 = self.raw_embeddings[idx2]

        cos1 = F.cosine_similarity(
            raw_emb1.unsqueeze(0),
            self.mean.unsqueeze(0),
            dim=1,
        ).item()
        cos2 = F.cosine_similarity(
            raw_emb2.unsqueeze(0),
            self.mean.unsqueeze(0),
            dim=1,
        ).item()

        norm1 = torch.linalg.vector_norm(emb1).item()
        norm2 = torch.linalg.vector_norm(emb2).item()

        if cos1 >= cos2:
            dominant_idx, recessive_idx = idx1, idx2
            dominant_embedding, recessive_embedding = emb1, emb2
            dominant_norm, recessive_norm = norm1, norm2
        else:
            dominant_idx, recessive_idx = idx2, idx1
            dominant_embedding, recessive_embedding = emb2, emb1
            dominant_norm, recessive_norm = norm2, norm1

        return {
            "dominant_embedding": dominant_embedding,
            "recessive_embedding": recessive_embedding,
            "dominant_idx": dominant_idx,
            "recessive_idx": recessive_idx,
            "dominant_sample_id": self.sample_ids[dominant_idx],
            "recessive_sample_id": self.sample_ids[recessive_idx],
            "dominant_source_path": self.source_paths[dominant_idx],
            "recessive_source_path": self.source_paths[recessive_idx],
            "dominant_relative_path": self.relative_paths[dominant_idx],
            "recessive_relative_path": self.relative_paths[recessive_idx],
            "dominant_norm": dominant_norm,
            "recessive_norm": recessive_norm,
        }

    def __getitem__(self, index: int) -> Dict:
        idx1, idx2 = self._sample_pair_indices(index)
        return self._ordered_pair(idx1, idx2)


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_semantic_pairs(batch: List[Dict]) -> Dict:
    dominant_embedding = torch.stack([item["dominant_embedding"] for item in batch], dim=0)
    recessive_embedding = torch.stack([item["recessive_embedding"] for item in batch], dim=0)

    return {
        "dominant_embedding": dominant_embedding,
        "recessive_embedding": recessive_embedding,
        "dominant_idx": torch.tensor([item["dominant_idx"] for item in batch], dtype=torch.long),
        "recessive_idx": torch.tensor([item["recessive_idx"] for item in batch], dtype=torch.long),
        "dominant_norm": torch.tensor([item["dominant_norm"] for item in batch], dtype=torch.float32),
        "recessive_norm": torch.tensor([item["recessive_norm"] for item in batch], dtype=torch.float32),
        "dominant_sample_id": [item["dominant_sample_id"] for item in batch],
        "recessive_sample_id": [item["recessive_sample_id"] for item in batch],
        "dominant_source_path": [item["dominant_source_path"] for item in batch],
        "recessive_source_path": [item["recessive_source_path"] for item in batch],
        "dominant_relative_path": [item["dominant_relative_path"] for item in batch],
        "recessive_relative_path": [item["recessive_relative_path"] for item in batch],
    }


def get_data(args, partition: str):
    if partition not in {"train", "val"}:
        raise ValueError(f"Unsupported partition: {partition}")

    dataset = SemanticPairsDataset(
        dataset_root=args.dataset_root,
        split=partition,
        num_pairs=args.train_samples_per_epoch if partition == "train" else args.val_samples,
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

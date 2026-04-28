import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

VAL_PAIR_SEED = 1234
_ZSCORE_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
_COSINE_MEAN_CACHE: Dict[Tuple[str, str], torch.Tensor] = {}


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

def load_cosine_order_mean(dataset_root: str, source: str = "train_val") -> torch.Tensor:
    """
    Loads the raw DiffAE semantic embedding mean used for cosine-based ordering.

    source:
        "train"     -> compute mean only from train_zsem.pt
        "train_val" -> compute mean from train_zsem.pt + val_zsem.pt
    """
    if source not in {"train", "train_val"}:
        raise ValueError(f"Unsupported cosine mean source: {source}")

    cache_key = (dataset_root, source)
    cached = _COSINE_MEAN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    splits = ["train"] if source == "train" else ["train", "val"]

    all_embeddings = []
    for split in splits:
        split_path = os.path.join(dataset_root, "semantic", f"{split}_zsem.pt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Could not find semantic split file: {split_path}")

        pack = torch.load(split_path, map_location="cpu")
        all_embeddings.append(pack["z_sem"].to(torch.float32))

    embeddings = torch.cat(all_embeddings, dim=0)
    mean_embedding = embeddings.mean(dim=0)

    if mean_embedding.norm().item() < 1e-8:
        raise ValueError(
            "The cosine ordering mean has near-zero norm. "
            "Cosine similarity to the mean would be unstable."
        )

    _COSINE_MEAN_CACHE[cache_key] = mean_embedding
    return mean_embedding

class SemanticPairsDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        num_pairs: int,
        deterministic: bool = False,
        pair_ordering: str = "norm",
        cosine_mean_source: str = "train_val",
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")

        split_path = os.path.join(dataset_root, "semantic", f"{split}_zsem.pt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Could not find semantic split file: {split_path}")

        if pair_ordering not in {"norm", "cosine_mean"}:
            raise ValueError(f"Unsupported pair_ordering: {pair_ordering}")

        self.pair_ordering = pair_ordering
        self.cosine_mean_source = cosine_mean_source

        pack = torch.load(split_path, map_location="cpu")

        # Raw DiffAE semantic embeddings are used only for cosine-to-mean ordering.
        self.raw_embeddings = pack["z_sem"].to(torch.float32)

        # Z-scored embeddings are still what the model receives and predicts.
        self.mean, self.std = load_zscore_stats(dataset_root)
        self.embeddings = (self.raw_embeddings - self.mean) / self.std

        self.cosine_order_mean = None
        if self.pair_ordering == "cosine_mean":
            self.cosine_order_mean = load_cosine_order_mean(
                dataset_root=dataset_root,
                source=cosine_mean_source,
            )

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

        # Keep these for logging/diagnostics. These are norms in z-scored space.
        norm1 = emb1.norm().item()
        norm2 = emb2.norm().item()

        cosine1 = float("nan")
        cosine2 = float("nan")

        if self.pair_ordering == "norm":
            # Current behavior:
            # dominant = embedding with largest z-scored L2 norm.
            score1 = norm1
            score2 = norm2

        elif self.pair_ordering == "cosine_mean":
            # New behavior:
            # dominant = embedding with lowest cosine similarity to the raw global mean.
            # Lower cosine similarity means more angularly distinct from the mean.
            raw1 = self.raw_embeddings[idx1]
            raw2 = self.raw_embeddings[idx2]

            mean = self.cosine_order_mean

            cosine1 = F.cosine_similarity(
                raw1.unsqueeze(0),
                mean.unsqueeze(0),
                dim=1,
                eps=1e-8,
            ).item()

            cosine2 = F.cosine_similarity(
                raw2.unsqueeze(0),
                mean.unsqueeze(0),
                dim=1,
                eps=1e-8,
            ).item()

            # Use negative cosine so that larger score = more distinct.
            score1 = -cosine1
            score2 = -cosine2

        else:
            raise ValueError(f"Unsupported pair_ordering: {self.pair_ordering}")

        if score1 >= score2:
            dominant_idx, recessive_idx = idx1, idx2
            dominant_embedding, recessive_embedding = emb1, emb2
            dominant_norm, recessive_norm = norm1, norm2
            dominant_order_score, recessive_order_score = score1, score2
            dominant_cosine_to_mean, recessive_cosine_to_mean = cosine1, cosine2
        else:
            dominant_idx, recessive_idx = idx2, idx1
            dominant_embedding, recessive_embedding = emb2, emb1
            dominant_norm, recessive_norm = norm2, norm1
            dominant_order_score, recessive_order_score = score2, score1
            dominant_cosine_to_mean, recessive_cosine_to_mean = cosine2, cosine1

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
            "dominant_order_score": dominant_order_score,
            "recessive_order_score": recessive_order_score,
            "dominant_cosine_to_mean": dominant_cosine_to_mean,
            "recessive_cosine_to_mean": recessive_cosine_to_mean,
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
        "dominant_order_score": torch.tensor([item["dominant_order_score"] for item in batch], dtype=torch.float32),
        "recessive_order_score": torch.tensor([item["recessive_order_score"] for item in batch], dtype=torch.float32),
        "dominant_cosine_to_mean": torch.tensor([item["dominant_cosine_to_mean"] for item in batch], dtype=torch.float32),
        "recessive_cosine_to_mean": torch.tensor([item["recessive_cosine_to_mean"] for item in batch], dtype=torch.float32),
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
        pair_ordering=getattr(args, "pair_ordering", "norm"),
        cosine_mean_source=getattr(args, "cosine_mean_source", "train_val"),
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
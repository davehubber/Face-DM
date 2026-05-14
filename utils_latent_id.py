import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


VAL_PAIR_SEED = 1234
_TRAINING_STATS_CACHE: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
_CELEBA_STATS_CACHE: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}


def setup_logging(run_name: str) -> str:
    base_dir = os.path.join("experiments", run_name)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "decode"), exist_ok=True)
    return base_dir


# -------------------------
# Generic readers
# -------------------------

def read_metadata_csv(metadata_path: str) -> List[Dict]:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    rows = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_columns = {
            "embedding_index",
            "filename",
            "image_path",
            "identity_id",
            "label",
            "split",
        }

        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Metadata file is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            row = dict(row)
            row["embedding_index"] = int(row["embedding_index"])
            row["label"] = int(row["label"])
            row["split"] = row["split"].strip()
            rows.append(row)

    rows = sorted(rows, key=lambda r: r["embedding_index"])
    return rows


def _ffhq_split_path(ffhq_dataset_root: str, split: str) -> str:
    return os.path.join(ffhq_dataset_root, "semantic", f"{split}_zsem.pt")


def _load_ffhq_split(ffhq_dataset_root: str, split: str) -> Dict:
    split_path = _ffhq_split_path(ffhq_dataset_root, split)

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Could not find FFHQ semantic split file: {split_path}")

    pack = torch.load(split_path, map_location="cpu")

    if "z_sem" not in pack:
        raise KeyError(f"FFHQ split file is missing key 'z_sem': {split_path}")

    z_sem = pack["z_sem"].to(torch.float32)

    if z_sem.ndim != 2:
        raise ValueError(f"Expected FFHQ z_sem to be 2D, got shape {tuple(z_sem.shape)}")

    n = z_sem.shape[0]

    sample_ids = list(pack.get("sample_ids", [f"ffhq_{split}_{i}" for i in range(n)]))
    source_paths = list(pack.get("source_paths", [""] * n))
    relative_paths = list(pack.get("relative_paths", [""] * n))

    if not (len(sample_ids) == len(source_paths) == len(relative_paths) == n):
        raise ValueError(
            f"FFHQ metadata lengths do not match embeddings in split {split}."
        )

    return {
        "z_sem": z_sem,
        "sample_ids": sample_ids,
        "source_paths": source_paths,
        "relative_paths": relative_paths,
    }


def _celeba_paths(args):
    embeddings_path = os.path.join(
        args.celeba_dataset_root,
        args.celeba_raw_embeddings_file,
    )
    metadata_path = os.path.join(
        args.celeba_dataset_root,
        args.celeba_metadata_file,
    )
    return embeddings_path, metadata_path


def _load_celeba_raw_and_metadata(args):
    embeddings_path, metadata_path = _celeba_paths(args)

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"CelebA raw embedding file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path).astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected CelebA embeddings to be 2D, got shape {embeddings.shape}")

    rows = read_metadata_csv(metadata_path)

    if len(rows) != embeddings.shape[0]:
        raise ValueError(
            f"CelebA metadata rows and embeddings do not match: "
            f"{len(rows)} rows vs {embeddings.shape[0]} embeddings."
        )

    expected_indices = list(range(len(rows)))
    actual_indices = [row["embedding_index"] for row in rows]

    if actual_indices != expected_indices:
        raise ValueError(
            "CelebA embedding_index values must be exactly 0..N-1 after sorting."
        )

    labels = sorted(set(row["label"] for row in rows))

    if labels != list(range(len(labels))):
        raise ValueError(
            "CelebA labels must be contiguous from 0 to num_classes - 1."
        )

    return torch.from_numpy(embeddings).float(), rows


# -------------------------
# Public metadata helpers
# -------------------------

def get_embedding_dim(args) -> int:
    ffhq_train = _load_ffhq_split(args.ffhq_dataset_root, "train")["z_sem"]
    celeba_raw, _ = _load_celeba_raw_and_metadata(args)

    ffhq_dim = int(ffhq_train.shape[1])
    celeba_dim = int(celeba_raw.shape[1])

    if ffhq_dim != celeba_dim:
        raise ValueError(
            f"Embedding dimension mismatch: FFHQ has {ffhq_dim}, CelebA has {celeba_dim}."
        )

    return ffhq_dim


def get_num_classes(args) -> int:
    _, rows = _load_celeba_raw_and_metadata(args)

    labels = sorted(set(row["label"] for row in rows))

    if labels != list(range(len(labels))):
        raise ValueError(
            "CelebA labels must be contiguous from 0 to num_classes - 1."
        )

    return len(labels)


def get_celeba_classifier_zscore_stats(args) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_key = (
        args.celeba_dataset_root,
        args.celeba_classifier_stats_file,
    )

    cached = _CELEBA_STATS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    stats_path = os.path.join(
        args.celeba_dataset_root,
        args.celeba_classifier_stats_file,
    )

    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"CelebA classifier z-score stats file not found: {stats_path}"
        )

    stats = np.load(stats_path)

    if "mean" not in stats or "std" not in stats:
        raise KeyError(
            f"Expected keys 'mean' and 'std' in CelebA stats file: {stats_path}"
        )

    mean = torch.from_numpy(stats["mean"].astype(np.float32))
    std = torch.from_numpy(stats["std"].astype(np.float32)).clamp_min(1e-6)

    _CELEBA_STATS_CACHE[cache_key] = (mean, std)
    return mean, std


def get_training_normalization_stats(args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the shared coordinate system used by the deaveraging model.

    The identity classifier is NOT trained in this space. During classifier loss,
    train-space embeddings are converted back to raw DiffAE z_sem and then to
    CelebA classifier z-score space.
    """
    cache_key = (
        args.ffhq_dataset_root,
        args.celeba_dataset_root,
        args.celeba_raw_embeddings_file,
        args.celeba_metadata_file,
        args.training_normalization,
    )

    cached = _TRAINING_STATS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ffhq_train = _load_ffhq_split(args.ffhq_dataset_root, "train")["z_sem"]
    celeba_raw, celeba_rows = _load_celeba_raw_and_metadata(args)

    celeba_train_indices = [
        row["embedding_index"]
        for row in celeba_rows
        if row["split"] == "train"
    ]

    if len(celeba_train_indices) == 0:
        raise ValueError("CelebA metadata contains no train split samples.")

    celeba_train = celeba_raw[celeba_train_indices]

    if ffhq_train.shape[1] != celeba_train.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: FFHQ has {ffhq_train.shape[1]}, "
            f"CelebA has {celeba_train.shape[1]}."
        )

    mode = args.training_normalization

    if mode == "combined":
        stats_source = torch.cat([ffhq_train, celeba_train], dim=0)
    elif mode == "ffhq":
        stats_source = ffhq_train
    elif mode == "celeba":
        stats_source = celeba_train
    elif mode == "none":
        dim = ffhq_train.shape[1]
        mean = torch.zeros(dim, dtype=torch.float32)
        std = torch.ones(dim, dtype=torch.float32)
        _TRAINING_STATS_CACHE[cache_key] = (mean, std)
        return mean, std
    else:
        raise ValueError(f"Unsupported training_normalization: {mode}")

    mean = stats_source.mean(dim=0).to(torch.float32)
    std = stats_source.std(dim=0, unbiased=False).clamp_min(1e-6).to(torch.float32)

    _TRAINING_STATS_CACHE[cache_key] = (mean, std)
    return mean, std


# -------------------------
# Mixed FFHQ + CelebA pair dataset
# -------------------------

class MixedSemanticPairsDataset(Dataset):
    def __init__(
        self,
        args,
        split: str,
        num_pairs: int,
        deterministic: bool = False,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")

        self.args = args
        self.split = split
        self.num_pairs = int(num_pairs)
        self.deterministic = bool(deterministic)

        self.celeba_pair_probability = float(args.celeba_pair_probability)
        self.require_different_celeba_identity = not bool(
            getattr(args, "allow_same_identity_celeba_pairs", False)
        )
        self.balanced_celeba_identity_sampling = not bool(
            getattr(args, "no_balanced_celeba_identity_sampling", False)
        )

        shared_mean, shared_std = get_training_normalization_stats(args)
        self.shared_mean = shared_mean
        self.shared_std = shared_std

        self._load_ffhq()
        self._load_celeba()

        if self.ffhq_embeddings.shape[1] != self.celeba_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch after loading: "
                f"FFHQ {self.ffhq_embeddings.shape[1]}, "
                f"CelebA {self.celeba_embeddings.shape[1]}."
            )

        print(
            f"Loaded mixed {split} split:\n"
            f"  FFHQ:   {len(self.ffhq_embeddings)} embeddings\n"
            f"  CelebA: {len(self.celeba_embeddings)} embeddings, "
            f"{len(self.celeba_unique_labels)} identities\n"
            f"  P(CelebA pair): {self.celeba_pair_probability}\n"
            f"  num_pairs: {self.num_pairs}\n"
            f"  deterministic: {self.deterministic}\n"
            f"  training_normalization: {args.training_normalization}"
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(torch.float32) - self.shared_mean) / self.shared_std

    def _load_ffhq(self):
        pack = _load_ffhq_split(self.args.ffhq_dataset_root, self.split)

        self.ffhq_embeddings = self._normalize(pack["z_sem"])
        self.ffhq_sample_ids = list(pack["sample_ids"])
        self.ffhq_source_paths = list(pack["source_paths"])
        self.ffhq_relative_paths = list(pack["relative_paths"])

        if len(self.ffhq_embeddings) < 2:
            raise ValueError(
                f"FFHQ {self.split} split must contain at least 2 embeddings."
            )

    def _load_celeba(self):
        celeba_raw, celeba_rows = _load_celeba_raw_and_metadata(self.args)

        split_rows = [
            row for row in celeba_rows
            if row["split"] == self.split
        ]

        if len(split_rows) < 2:
            raise ValueError(
                f"CelebA {self.split} split must contain at least 2 embeddings."
            )

        split_rows = sorted(split_rows, key=lambda r: r["embedding_index"])

        self.celeba_embedding_indices = [
            row["embedding_index"]
            for row in split_rows
        ]

        self.celeba_embeddings = self._normalize(
            celeba_raw[self.celeba_embedding_indices]
        )

        self.celeba_sample_ids = [row["filename"] for row in split_rows]
        self.celeba_source_paths = [row["image_path"] for row in split_rows]
        self.celeba_relative_paths = [row["filename"] for row in split_rows]
        self.celeba_identity_ids = [str(row["identity_id"]) for row in split_rows]
        self.celeba_labels = [int(row["label"]) for row in split_rows]

        self.celeba_label_to_indices: Dict[int, List[int]] = defaultdict(list)

        for local_index, label in enumerate(self.celeba_labels):
            self.celeba_label_to_indices[label].append(local_index)

        self.celeba_unique_labels = sorted(self.celeba_label_to_indices.keys())

        if self.require_different_celeba_identity and len(self.celeba_unique_labels) < 2:
            raise ValueError(
                f"CelebA {self.split} split needs at least 2 identities for "
                "different-identity pairs."
            )

    def __len__(self) -> int:
        return self.num_pairs

    def _rng_for_index(self, index: int) -> random.Random:
        if self.deterministic:
            return random.Random(VAL_PAIR_SEED + index)

        return random

    def _sample_source(self, rng: random.Random) -> str:
        if rng.random() < self.celeba_pair_probability:
            return "celeba"

        return "ffhq"

    def _sample_ffhq_pair(self, rng: random.Random) -> Tuple[int, int]:
        n = len(self.ffhq_embeddings)

        idx1 = rng.randrange(n)
        idx2 = rng.randrange(n - 1)

        if idx2 >= idx1:
            idx2 += 1

        return idx1, idx2

    def _sample_celeba_balanced_identity_pair(self, rng: random.Random) -> Tuple[int, int]:
        label_1 = rng.choice(self.celeba_unique_labels)

        if self.require_different_celeba_identity:
            label_2 = rng.choice(self.celeba_unique_labels)

            while label_2 == label_1:
                label_2 = rng.choice(self.celeba_unique_labels)
        else:
            label_2 = rng.choice(self.celeba_unique_labels)

        idx1 = rng.choice(self.celeba_label_to_indices[label_1])
        idx2 = rng.choice(self.celeba_label_to_indices[label_2])

        if not self.require_different_celeba_identity and idx1 == idx2:
            n = len(self.celeba_embeddings)
            idx2 = rng.randrange(n - 1)

            if idx2 >= idx1:
                idx2 += 1

        return idx1, idx2

    def _sample_celeba_image_level_pair(self, rng: random.Random) -> Tuple[int, int]:
        n = len(self.celeba_embeddings)

        idx1 = rng.randrange(n)

        if self.require_different_celeba_identity:
            label_1 = self.celeba_labels[idx1]

            idx2 = rng.randrange(n)
            attempts = 0

            while self.celeba_labels[idx2] == label_1:
                idx2 = rng.randrange(n)
                attempts += 1

                if attempts > 1000:
                    raise RuntimeError(
                        "Failed to sample a different-identity CelebA pair."
                    )
        else:
            idx2 = rng.randrange(n - 1)

            if idx2 >= idx1:
                idx2 += 1

        return idx1, idx2

    def _sample_celeba_pair(self, rng: random.Random) -> Tuple[int, int]:
        if self.balanced_celeba_identity_sampling:
            return self._sample_celeba_balanced_identity_pair(rng)

        return self._sample_celeba_image_level_pair(rng)

    def _make_ffhq_item(self, idx1: int, idx2: int) -> Dict:
        return {
            "pair_source": "ffhq",
            "is_celeba_pair": False,

            "clean_embedding_1": self.ffhq_embeddings[idx1],
            "clean_embedding_2": self.ffhq_embeddings[idx2],

            "clean_label_1": -1,
            "clean_label_2": -1,

            "clean_identity_id_1": "",
            "clean_identity_id_2": "",

            "clean_idx_1": idx1,
            "clean_idx_2": idx2,

            "clean_embedding_index_1": idx1,
            "clean_embedding_index_2": idx2,

            "clean_sample_id_1": self.ffhq_sample_ids[idx1],
            "clean_sample_id_2": self.ffhq_sample_ids[idx2],

            "clean_source_path_1": self.ffhq_source_paths[idx1],
            "clean_source_path_2": self.ffhq_source_paths[idx2],

            "clean_relative_path_1": self.ffhq_relative_paths[idx1],
            "clean_relative_path_2": self.ffhq_relative_paths[idx2],
        }

    def _make_celeba_item(self, idx1: int, idx2: int) -> Dict:
        return {
            "pair_source": "celeba",
            "is_celeba_pair": True,

            "clean_embedding_1": self.celeba_embeddings[idx1],
            "clean_embedding_2": self.celeba_embeddings[idx2],

            "clean_label_1": self.celeba_labels[idx1],
            "clean_label_2": self.celeba_labels[idx2],

            "clean_identity_id_1": self.celeba_identity_ids[idx1],
            "clean_identity_id_2": self.celeba_identity_ids[idx2],

            "clean_idx_1": idx1,
            "clean_idx_2": idx2,

            "clean_embedding_index_1": self.celeba_embedding_indices[idx1],
            "clean_embedding_index_2": self.celeba_embedding_indices[idx2],

            "clean_sample_id_1": self.celeba_sample_ids[idx1],
            "clean_sample_id_2": self.celeba_sample_ids[idx2],

            "clean_source_path_1": self.celeba_source_paths[idx1],
            "clean_source_path_2": self.celeba_source_paths[idx2],

            "clean_relative_path_1": self.celeba_relative_paths[idx1],
            "clean_relative_path_2": self.celeba_relative_paths[idx2],
        }

    def __getitem__(self, index: int) -> Dict:
        rng = self._rng_for_index(index)
        source = self._sample_source(rng)

        if source == "celeba":
            idx1, idx2 = self._sample_celeba_pair(rng)
            return self._make_celeba_item(idx1, idx2)

        idx1, idx2 = self._sample_ffhq_pair(rng)
        return self._make_ffhq_item(idx1, idx2)


# -------------------------
# Dataloader
# -------------------------

def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_semantic_pairs(batch: List[Dict]) -> Dict:
    clean_embedding_1 = torch.stack(
        [item["clean_embedding_1"] for item in batch],
        dim=0,
    )
    clean_embedding_2 = torch.stack(
        [item["clean_embedding_2"] for item in batch],
        dim=0,
    )

    return {
        "pair_source": [item["pair_source"] for item in batch],
        "is_celeba_pair": torch.tensor(
            [item["is_celeba_pair"] for item in batch],
            dtype=torch.bool,
        ),

        "clean_embedding_1": clean_embedding_1,
        "clean_embedding_2": clean_embedding_2,

        "clean_label_1": torch.tensor(
            [item["clean_label_1"] for item in batch],
            dtype=torch.long,
        ),
        "clean_label_2": torch.tensor(
            [item["clean_label_2"] for item in batch],
            dtype=torch.long,
        ),

        "clean_identity_id_1": [item["clean_identity_id_1"] for item in batch],
        "clean_identity_id_2": [item["clean_identity_id_2"] for item in batch],

        "clean_idx_1": torch.tensor(
            [item["clean_idx_1"] for item in batch],
            dtype=torch.long,
        ),
        "clean_idx_2": torch.tensor(
            [item["clean_idx_2"] for item in batch],
            dtype=torch.long,
        ),

        "clean_embedding_index_1": torch.tensor(
            [item["clean_embedding_index_1"] for item in batch],
            dtype=torch.long,
        ),
        "clean_embedding_index_2": torch.tensor(
            [item["clean_embedding_index_2"] for item in batch],
            dtype=torch.long,
        ),

        "clean_sample_id_1": [item["clean_sample_id_1"] for item in batch],
        "clean_sample_id_2": [item["clean_sample_id_2"] for item in batch],

        "clean_source_path_1": [item["clean_source_path_1"] for item in batch],
        "clean_source_path_2": [item["clean_source_path_2"] for item in batch],

        "clean_relative_path_1": [item["clean_relative_path_1"] for item in batch],
        "clean_relative_path_2": [item["clean_relative_path_2"] for item in batch],
    }


def get_data(args, partition: str):
    if partition not in {"train", "val"}:
        raise ValueError(f"Unsupported partition: {partition}")

    if partition == "train":
        num_pairs = args.train_samples_per_epoch
        deterministic = False
    else:
        num_pairs = args.val_samples
        deterministic = True

    dataset = MixedSemanticPairsDataset(
        args=args,
        split=partition,
        num_pairs=num_pairs,
        deterministic=deterministic,
    )

    configured_workers = getattr(args, "num_workers", None)

    if configured_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)
    else:
        num_workers = max(0, int(configured_workers))

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
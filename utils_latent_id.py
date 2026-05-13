import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


VAL_PAIR_SEED = 1234


def setup_logging(run_name: str) -> str:
    base_dir = os.path.join("experiments", run_name)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "decode"), exist_ok=True)
    return base_dir


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


def get_embedding_dim(
    dataset_root: str,
    embeddings_file: str = "celeba_diffae_zsem_zscore.npy",
) -> int:
    embeddings_path = os.path.join(dataset_root, embeddings_file)

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embedding file not found: {embeddings_path}")

    embeddings = np.load(embeddings_path, mmap_mode="r")

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    return int(embeddings.shape[1])


def get_num_classes(
    dataset_root: str,
    metadata_file: str = "celeba_diffae_zsem_metadata.csv",
) -> int:
    metadata_path = os.path.join(dataset_root, metadata_file)
    rows = read_metadata_csv(metadata_path)

    labels = sorted(set(row["label"] for row in rows))

    if not labels:
        raise ValueError("No labels found in metadata.")

    expected_labels = list(range(len(labels)))

    if labels != expected_labels:
        raise ValueError(
            "Labels must be contiguous from 0 to num_classes - 1. "
            f"Found min={min(labels)}, max={max(labels)}, num_unique={len(labels)}."
        )

    return len(labels)


class CelebADiffAESemanticPairsDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        num_pairs: int,
        embeddings_file: str = "celeba_diffae_zsem_zscore.npy",
        metadata_file: str = "celeba_diffae_zsem_metadata.csv",
        deterministic: bool = False,
        require_different_identity: bool = True,
        balanced_identity_sampling: bool = True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        if num_pairs <= 0:
            raise ValueError(f"num_pairs must be positive, got {num_pairs}")

        self.dataset_root = dataset_root
        self.split = split
        self.num_pairs = int(num_pairs)
        self.deterministic = bool(deterministic)
        self.require_different_identity = bool(require_different_identity)
        self.balanced_identity_sampling = bool(balanced_identity_sampling)

        embeddings_path = os.path.join(dataset_root, embeddings_file)
        metadata_path = os.path.join(dataset_root, metadata_file)

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embedding file not found: {embeddings_path}")

        full_embeddings = np.load(embeddings_path).astype(np.float32)
        metadata_rows = read_metadata_csv(metadata_path)

        self._validate_full_dataset(full_embeddings, metadata_rows)

        split_rows = [
            row for row in metadata_rows
            if row["split"] == split
        ]

        if len(split_rows) < 2:
            raise ValueError(
                f"Split '{split}' must contain at least 2 images; found {len(split_rows)}."
            )

        split_rows = sorted(split_rows, key=lambda r: r["embedding_index"])

        embedding_indices = [row["embedding_index"] for row in split_rows]

        self.embeddings = torch.from_numpy(full_embeddings[embedding_indices]).float()

        self.embedding_indices: List[int] = embedding_indices
        self.sample_ids: List[str] = [row["filename"] for row in split_rows]
        self.source_paths: List[str] = [row["image_path"] for row in split_rows]
        self.relative_paths: List[str] = [row.get("filename", "") for row in split_rows]
        self.identity_ids: List[str] = [str(row["identity_id"]) for row in split_rows]
        self.labels: List[int] = [int(row["label"]) for row in split_rows]

        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)

        for local_index, label in enumerate(self.labels):
            self.label_to_indices[label].append(local_index)

        self.unique_labels = sorted(self.label_to_indices.keys())

        if self.require_different_identity and len(self.unique_labels) < 2:
            raise ValueError(
                f"Split '{split}' needs at least 2 identities for different-identity pairs."
            )

        self.num_classes = len(set(row["label"] for row in metadata_rows))

        mode = "deterministic" if deterministic else "random"
        pair_type = "different-identity" if require_different_identity else "any-identity"
        sampler_type = (
            "balanced identity sampling"
            if balanced_identity_sampling
            else "image-level sampling"
        )

        print(
            f"Loaded {split} split: {len(self.embeddings)} embeddings, "
            f"{len(self.unique_labels)} identities in split, "
            f"{self.num_pairs} {mode} {pair_type} pairs, {sampler_type}."
        )

    @staticmethod
    def _validate_full_dataset(embeddings: np.ndarray, rows: List[Dict]):
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embedding array, got shape {embeddings.shape}")

        if len(rows) != embeddings.shape[0]:
            raise ValueError(
                f"Metadata rows and embeddings do not match: "
                f"{len(rows)} rows vs {embeddings.shape[0]} embeddings."
            )

        expected_indices = list(range(len(rows)))
        actual_indices = [row["embedding_index"] for row in rows]

        if actual_indices != expected_indices:
            raise ValueError(
                "embedding_index values must be exactly 0..N-1 after sorting. "
                "The deaveraging script assumes metadata directly indexes the embedding array."
            )

        labels = sorted(set(row["label"] for row in rows))

        if labels != list(range(len(labels))):
            raise ValueError(
                "Labels must be contiguous from 0 to num_classes - 1."
            )

        split_counts = defaultdict(int)

        for row in rows:
            split_counts[row["split"]] += 1

        for required_split in ["train", "val"]:
            if split_counts[required_split] == 0:
                raise ValueError(f"Metadata is missing required split: {required_split}")

    def __len__(self) -> int:
        return self.num_pairs

    def _rng_for_index(self, index: int) -> random.Random:
        if self.deterministic:
            return random.Random(VAL_PAIR_SEED + index)

        return random

    def _sample_balanced_identity_pair(self, rng: random.Random) -> Tuple[int, int]:
        label_1 = rng.choice(self.unique_labels)

        if self.require_different_identity:
            label_2 = rng.choice(self.unique_labels)

            while label_2 == label_1:
                label_2 = rng.choice(self.unique_labels)
        else:
            label_2 = rng.choice(self.unique_labels)

        idx1 = rng.choice(self.label_to_indices[label_1])
        idx2 = rng.choice(self.label_to_indices[label_2])

        if not self.require_different_identity and idx1 == idx2:
            n = len(self.embeddings)
            idx2 = rng.randrange(n - 1)

            if idx2 >= idx1:
                idx2 += 1

        return idx1, idx2

    def _sample_image_level_pair(self, rng: random.Random) -> Tuple[int, int]:
        n = len(self.embeddings)

        idx1 = rng.randrange(n)

        if self.require_different_identity:
            label_1 = self.labels[idx1]

            idx2 = rng.randrange(n)
            attempts = 0

            while self.labels[idx2] == label_1:
                idx2 = rng.randrange(n)
                attempts += 1

                if attempts > 1000:
                    raise RuntimeError(
                        "Failed to sample a different-identity pair after many attempts."
                    )
        else:
            idx2 = rng.randrange(n - 1)

            if idx2 >= idx1:
                idx2 += 1

        return idx1, idx2

    def _sample_pair_indices(self, index: int) -> Tuple[int, int]:
        rng = self._rng_for_index(index)

        if self.balanced_identity_sampling:
            return self._sample_balanced_identity_pair(rng)

        return self._sample_image_level_pair(rng)

    def _unsorted_pair(self, idx1: int, idx2: int) -> Dict:
        clean_embedding_1 = self.embeddings[idx1]
        clean_embedding_2 = self.embeddings[idx2]

        return {
            "clean_embedding_1": clean_embedding_1,
            "clean_embedding_2": clean_embedding_2,

            "clean_label_1": self.labels[idx1],
            "clean_label_2": self.labels[idx2],

            "clean_identity_id_1": self.identity_ids[idx1],
            "clean_identity_id_2": self.identity_ids[idx2],

            "clean_idx_1": idx1,
            "clean_idx_2": idx2,

            "clean_embedding_index_1": self.embedding_indices[idx1],
            "clean_embedding_index_2": self.embedding_indices[idx2],

            "clean_sample_id_1": self.sample_ids[idx1],
            "clean_sample_id_2": self.sample_ids[idx2],

            "clean_source_path_1": self.source_paths[idx1],
            "clean_source_path_2": self.source_paths[idx2],

            "clean_relative_path_1": self.relative_paths[idx1],
            "clean_relative_path_2": self.relative_paths[idx2],
        }

    def __getitem__(self, index: int) -> Dict:
        idx1, idx2 = self._sample_pair_indices(index)
        return self._unsorted_pair(idx1, idx2)


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
    if partition not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported partition: {partition}")

    if partition == "train":
        num_pairs = args.train_samples_per_epoch
        deterministic = False
    elif partition == "val":
        num_pairs = args.val_samples
        deterministic = True
    else:
        num_pairs = getattr(args, "test_samples", args.val_samples)
        deterministic = True

    dataset = CelebADiffAESemanticPairsDataset(
        dataset_root=args.dataset_root,
        split=partition,
        num_pairs=num_pairs,
        embeddings_file=args.embeddings_file,
        metadata_file=args.metadata_file,
        deterministic=deterministic,
        require_different_identity=not getattr(args, "allow_same_identity_pairs", False),
        balanced_identity_sampling=not getattr(args, "no_balanced_identity_sampling", False),
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
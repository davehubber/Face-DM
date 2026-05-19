from pathlib import Path
import argparse
import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, eps, None)


def main():
    parser = argparse.ArgumentParser(
        description="Sample 100 random unique ArcFace embedding pairs and measure cosine similarity to their averages."
    )

    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy",
        help="Path to the stored ArcFace embeddings .npy file.",
    )

    parser.add_argument(
        "--out_txt",
        type=str,
        default="/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/arcface_average_cosine_test_100.txt",
        help="Output txt file.",
    )

    parser.add_argument(
        "--num_pairs",
        type=int,
        default=100,
        help="Number of unique random pairs to sample.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    embeddings_path = Path(args.embeddings_path)
    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(embeddings_path).astype(np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings with shape [N, D], got {embeddings.shape}")

    n, d = embeddings.shape

    if n < args.num_pairs * 2:
        raise ValueError(
            f"Need at least {args.num_pairs * 2} embeddings to make "
            f"{args.num_pairs} unique non-overlapping pairs, but only found {n}."
        )

    # Re-normalize for safety, even though your encoding script already saved L2-normalized embeddings.
    embeddings = l2_normalize(embeddings)

    rng = np.random.default_rng(args.seed)

    # Sample 2*num_pairs unique indices, then pair them consecutively.
    # This guarantees:
    # - no pair is repeated
    # - no embedding is reused across pairs
    sampled_indices = rng.choice(n, size=args.num_pairs * 2, replace=False)
    pairs = sampled_indices.reshape(args.num_pairs, 2)

    lines = []
    lines.append(f"embeddings_path: {embeddings_path}")
    lines.append(f"num_embeddings: {n}")
    lines.append(f"embedding_dim: {d}")
    lines.append(f"num_pairs: {args.num_pairs}")
    lines.append(f"seed: {args.seed}")
    lines.append("")
    lines.append(
        "pair_id\tidx_1\tidx_2\tcosine_average_to_idx_1\tcosine_average_to_idx_2\taverage_l2_norm_before_normalization"
    )

    for pair_id, (idx_1, idx_2) in enumerate(pairs):
        emb_1 = embeddings[idx_1]
        emb_2 = embeddings[idx_2]

        # Arithmetic average of the two L2-normalized ArcFace embeddings.
        avg = (emb_1 + emb_2) / 2.0

        avg_norm = float(np.linalg.norm(avg))

        # Cosine similarity uses the direction of the average vector.
        avg_unit = avg / max(avg_norm, 1e-12)

        cos_to_1 = float(np.dot(avg_unit, emb_1))
        cos_to_2 = float(np.dot(avg_unit, emb_2))

        lines.append(
            f"{pair_id:03d}\t"
            f"{int(idx_1)}\t"
            f"{int(idx_2)}\t"
            f"{cos_to_1:.8f}\t"
            f"{cos_to_2:.8f}\t"
            f"{avg_norm:.8f}"
        )

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved results to: {out_txt}")


if __name__ == "__main__":
    main()

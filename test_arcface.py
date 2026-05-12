import argparse
import os
import tempfile
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from deepface import DeepFace


MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
ALIGN = True
NORMALIZATION = "ArcFace"
ENFORCE_DETECTION = True


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def choose_main_face(face_objs):
    """
    DeepFace.represent returns a list of detected faces.
    FFHQ should usually have one face, but if multiple are detected,
    we choose the largest detected face.
    """
    if len(face_objs) == 0:
        raise RuntimeError("DeepFace returned zero faces.")

    if len(face_objs) == 1:
        return face_objs[0]

    def area(obj):
        facial_area = obj.get("facial_area", {})
        return float(facial_area.get("w", 0) * facial_area.get("h", 0))

    return max(face_objs, key=area)


def average_two_images(path_a: str, path_b: str) -> Image.Image:
    img_a = Image.open(path_a).convert("RGB")
    img_b = Image.open(path_b).convert("RGB")

    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.BICUBIC)

    arr_a = np.asarray(img_a).astype(np.float32)
    arr_b = np.asarray(img_b).astype(np.float32)

    avg = ((arr_a + arr_b) / 2.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(avg, mode="RGB")


def extract_deepface_arcface_embedding(image_path: str) -> np.ndarray:
    face_objs = DeepFace.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        align=ALIGN,
        enforce_detection=ENFORCE_DETECTION,
        normalization=NORMALIZATION,
    )

    obj = choose_main_face(face_objs)
    emb = np.asarray(obj["embedding"], dtype=np.float32)

    if emb.ndim != 1:
        raise RuntimeError(f"Unexpected embedding shape: {emb.shape}")

    return l2_normalize(emb)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to your original DeepFace ArcFace .npy embeddings file.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to the matching metadata CSV with image_path column.",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        required=True,
        help="Path to save the txt report.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=100,
        help="Number of random image pairs to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--keep-averaged-images",
        action="store_true",
        help="If set, saves the averaged images next to the report.",
    )

    args = parser.parse_args()

    embeddings_path = Path(args.embeddings)
    metadata_path = Path(args.metadata)
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(embeddings_path).astype(np.float32)
    metadata = pd.read_csv(metadata_path).reset_index(drop=True)

    if "image_path" not in metadata.columns:
        raise ValueError("Metadata CSV must contain an 'image_path' column.")

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Metadata rows ({len(metadata)}) and embeddings ({len(embeddings)}) do not match."
        )

    n = len(metadata)
    if n < 2:
        raise ValueError("Need at least 2 images to form pairs.")

    rng = np.random.default_rng(args.seed)

    if n >= args.num_pairs * 2:
        selected = rng.choice(n, size=args.num_pairs * 2, replace=False)
        pairs = selected.reshape(args.num_pairs, 2)
    else:
        pairs = np.array([
            rng.choice(n, size=2, replace=False)
            for _ in range(args.num_pairs)
        ])

    if args.keep_averaged_images:
        avg_dir = out_report.parent / "averaged_images"
        avg_dir.mkdir(parents=True, exist_ok=True)
        tmp_context = None
    else:
        tmp_context = tempfile.TemporaryDirectory()
        avg_dir = Path(tmp_context.name)

    successes = []
    failures = []

    try:
        for pair_idx, (idx_a, idx_b) in enumerate(tqdm(pairs, desc="Testing averaged faces")):
            path_a = metadata.loc[idx_a, "image_path"]
            path_b = metadata.loc[idx_b, "image_path"]

            try:
                avg_img = average_two_images(path_a, path_b)

                avg_path = avg_dir / f"avg_pair_{pair_idx:04d}_idx_{idx_a}_{idx_b}.png"
                avg_img.save(avg_path)

                avg_emb = extract_deepface_arcface_embedding(str(avg_path))

                emb_a = l2_normalize(embeddings[idx_a])
                emb_b = l2_normalize(embeddings[idx_b])

                sim_to_a = cosine_similarity(avg_emb, emb_a)
                sim_to_b = cosine_similarity(avg_emb, emb_b)
                emb_pair_avg = l2_normalize((emb_a + emb_b) / 2.0)
                sim_to_embedding_average = cosine_similarity(avg_emb, emb_pair_avg)

                successes.append({
                    "pair_idx": pair_idx,
                    "idx_a": int(idx_a),
                    "idx_b": int(idx_b),
                    "image_a": path_a,
                    "image_b": path_b,
                    "avg_image": str(avg_path) if args.keep_averaged_images else "not_saved",
                    "cosine_to_a": sim_to_a,
                    "cosine_to_b": sim_to_b,
                    "cosine_to_embedding_average": sim_to_embedding_average,
                })

            except Exception as e:
                failures.append({
                    "pair_idx": pair_idx,
                    "idx_a": int(idx_a),
                    "idx_b": int(idx_b),
                    "image_a": path_a,
                    "image_b": path_b,
                    "error": repr(e),
                })

        with open(out_report, "w", encoding="utf-8") as f:
            f.write("DeepFace ArcFace on averaged FFHQ-256 image pairs\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Original embeddings file: {embeddings_path}\n")
            f.write(f"Metadata file: {metadata_path}\n")
            f.write(f"Number of requested pairs: {args.num_pairs}\n")
            f.write(f"Random seed: {args.seed}\n\n")

            f.write("DeepFace setup\n")
            f.write("-" * 70 + "\n")
            f.write(f"model_name: {MODEL_NAME}\n")
            f.write(f"detector_backend: {DETECTOR_BACKEND}\n")
            f.write(f"align: {ALIGN}\n")
            f.write(f"normalization: {NORMALIZATION}\n")
            f.write(f"enforce_detection: {ENFORCE_DETECTION}\n\n")

            f.write("Summary\n")
            f.write("-" * 70 + "\n")
            f.write(f"Succeeded: {len(successes)}\n")
            f.write(f"Failed: {len(failures)}\n\n")

            f.write("Successful pairs\n")
            f.write("-" * 70 + "\n")
            for item in successes:
                f.write(
                    f"pair {item['pair_idx']:04d} | "
                    f"idx_a={item['idx_a']} | "
                    f"idx_b={item['idx_b']} | "
                    f"cos(avg, a)={item['cosine_to_a']:.6f} | "
                    f"cos(avg, b)={item['cosine_to_b']:.6f} | "
                    f"cos(avg_img_emb, avg_emb_pair)={item['cosine_to_embedding_average']:.6f} | "
                    f"image_a={item['image_a']} | "
                    f"image_b={item['image_b']}\n"
                )

            f.write("\nFailed pairs\n")
            f.write("-" * 70 + "\n")
            if len(failures) == 0:
                f.write("None\n")
            else:
                for item in failures:
                    f.write(
                        f"pair {item['pair_idx']:04d} | "
                        f"idx_a={item['idx_a']} | "
                        f"idx_b={item['idx_b']} | "
                        f"image_a={item['image_a']} | "
                        f"image_b={item['image_b']} | "
                        f"error={item['error']}\n"
                    )

        print(f"Saved report to: {out_report}")
        print(f"Succeeded: {len(successes)}")
        print(f"Failed: {len(failures)}")

    finally:
        if tmp_context is not None:
            tmp_context.cleanup()


if __name__ == "__main__":
    main()

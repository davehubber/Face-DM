import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace


IMAGE_ROOT = Path("/nas-ctm01/datasets/public/ffhq256/")
OUT_DIR = Path("/nas-ctm01/homes/dacordeiro/arcface_embeddings/Face-DM/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIMIT = None  # set to None to process the full dataset

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"   # or "mtcnn" if RetinaFace is too slow
ALIGN = True
NORMALIZATION = "ArcFace"

image_paths = sorted(
    list(IMAGE_ROOT.rglob("*.png")) +
    list(IMAGE_ROOT.rglob("*.jpg")) +
    list(IMAGE_ROOT.rglob("*.jpeg"))
)

if LIMIT is not None:
    image_paths = image_paths[:LIMIT]

print(f"Found {len(image_paths)} images")

embeddings = []
rows = []
errors = []


def choose_main_face(face_objs):
    """
    DeepFace.represent may return multiple faces if the detector sees more than one.
    FFHQ should normally contain one main centered face.
    If multiple are returned, choose the largest detected face.
    """
    if len(face_objs) == 1:
        return face_objs[0]

    def area(obj):
        fa = obj.get("facial_area", {})
        return float(fa.get("w", 0) * fa.get("h", 0))

    return max(face_objs, key=area)


for idx, img_path in enumerate(tqdm(image_paths)):
    try:
        face_objs = DeepFace.represent(
            img_path=str(img_path),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            align=ALIGN,
            enforce_detection=True,
            normalization=NORMALIZATION,
        )

        obj = choose_main_face(face_objs)

        emb = np.asarray(obj["embedding"], dtype=np.float32)

        # Sanity check. ArcFace embeddings are normally 512-D.
        if emb.ndim != 1:
            raise ValueError(f"Unexpected embedding shape: {emb.shape}")

        # Store L2-normalized embedding for cosine-based use.
        norm = np.linalg.norm(emb)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("Invalid embedding norm")

        emb_l2 = emb / norm

        embeddings.append(emb_l2)

        rows.append({
            "index": len(embeddings) - 1,
            "image_path": str(img_path),
            "filename": img_path.name,
            "model_name": MODEL_NAME,
            "detector_backend": DETECTOR_BACKEND,
            "align": ALIGN,
            "normalization": NORMALIZATION,
            "embedding_dim": int(emb.shape[0]),
            "raw_embedding_norm": float(norm),
            "num_faces_detected": len(face_objs),
            "facial_area": json.dumps(obj.get("facial_area", {})),
            "face_confidence": obj.get("face_confidence", None),
        })

    except Exception as e:
        errors.append({
            "image_path": str(img_path),
            "filename": img_path.name,
            "error": repr(e),
        })


embeddings = np.stack(embeddings, axis=0).astype(np.float32)

np.save(OUT_DIR / "ffhq256_deepface_arcface_retinaface_l2norm.npy", embeddings)
pd.DataFrame(rows).to_csv(OUT_DIR / "ffhq256_deepface_arcface_metadata.csv", index=False)
pd.DataFrame(errors).to_csv(OUT_DIR / "ffhq256_deepface_arcface_errors.csv", index=False)

print(f"Saved embeddings: {embeddings.shape}")
print(f"Failures: {len(errors)}")

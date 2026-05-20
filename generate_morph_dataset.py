import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace

# ==========================================
# 1. Configuration
# ==========================================
METADATA_CSV = Path(
    "/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/"
    "ffhq256_deepface_arcface_metadata.csv"
)

EMBEDDINGS_NPY = Path(
    "/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/"
    "ffhq256_deepface_arcface_retinaface_l2norm.npy"
)

OUT_DIR = Path("/nas-ctm01/homes/dacordeiro/Face-DM/morphed_dataset/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PAIRS = 70000
TRAIN_RATIO = 0.9

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
ALIGN = True
enforce_detection=True
NORMALIZATION = "ArcFace"

# Safety limits so the script does not loop forever if something is wrong
MAX_ATTEMPTS = TARGET_PAIRS * 20
MAX_CONSECUTIVE_DEEPFACE_FAILURES = 50


# ==========================================
# 2. Helper Functions
# ==========================================
def choose_main_face(face_objs):
    """
    Select the largest detected face if multiple are returned.

    DeepFace.represent usually returns a list of dictionaries.
    If there is only one face, we must return face_objs[0], not face_objs.
    """
    if isinstance(face_objs, dict):
        return face_objs

    if not isinstance(face_objs, list) or len(face_objs) == 0:
        raise ValueError(f"No valid face objects returned. Got type: {type(face_objs)}")

    if len(face_objs) == 1:
        return face_objs[0]

    def area(obj):
        fa = obj.get("facial_area", {})
        return float(fa.get("w", 0) * fa.get("h", 0))

    return max(face_objs, key=area)


def l2_normalize(x, eps=1e-12):
    """
    L2-normalize one embedding vector.
    """
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)

    if norm < eps or not np.isfinite(norm):
        raise ValueError(f"Invalid embedding norm: {norm}")

    return x / norm


# ==========================================
# 3. Main Logic
# ==========================================
print("Loading original metadata and embeddings...")

df = pd.read_csv(METADATA_CSV)
original_embs = np.load(EMBEDDINGS_NPY).astype(np.float32)

num_total_images = len(df)

if len(original_embs) != num_total_images:
    raise ValueError(
        f"Metadata and embeddings length mismatch: "
        f"{num_total_images} metadata rows vs {len(original_embs)} embeddings."
    )

print(f"Loaded {num_total_images} original embeddings.")

seen_pairs = set()

dataset_z1 = []
dataset_z2 = []
dataset_c = []
metadata_rows = []

attempts = 0
consecutive_deepface_failures = 0

pbar = tqdm(total=TARGET_PAIRS, desc="Generating Morphs")

while len(dataset_c) < TARGET_PAIRS and attempts < MAX_ATTEMPTS:
    attempts += 1

    # 1. Sample two distinct random indices
    idx1, idx2 = random.sample(range(num_total_images), 2)

    # Sort indices strictly for uniqueness tracking
    pair_key = tuple(sorted((idx1, idx2)))
    if pair_key in seen_pairs:
        continue

    # 2. Load the actual source images
    path1 = str(df.iloc[idx1]["image_path"])
    path2 = str(df.iloc[idx2]["image_path"])

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        print(f"\nSkipping pair because image loading failed:")
        print(f"  idx1={idx1}, path1={path1}, loaded={img1 is not None}")
        print(f"  idx2={idx2}, path2={path2}, loaded={img2 is not None}")
        continue

    # Only mark pair as seen after the images are successfully loaded
    seen_pairs.add(pair_key)

    # Correct OpenCV resize format: size = (width, height)
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 3. Create the average image, pixel-wise morph
    avg_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    # 4. Extract ArcFace embedding from the average image
    try:
        face_objs = DeepFace.represent(
            img_path=avg_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            align=ALIGN,
            enforce_detection=False,
            normalization=NORMALIZATION,
        )

        obj = choose_main_face(face_objs)
        emb_morph = np.asarray(obj["embedding"], dtype=np.float32)

        emb_morph_l2 = l2_normalize(emb_morph)

        consecutive_deepface_failures = 0

    except Exception as e:
        consecutive_deepface_failures += 1

        print(f"\nDeepFace failed on pair idx1={idx1}, idx2={idx2}.")
        print(f"Error: {repr(e)}")

        if consecutive_deepface_failures >= MAX_CONSECUTIVE_DEEPFACE_FAILURES:
            raise RuntimeError(
                f"DeepFace failed {MAX_CONSECUTIVE_DEEPFACE_FAILURES} times in a row. "
                f"This likely means the extraction configuration is broken, not just one bad image."
            ) from e

        continue

    # 5. Fetch source embeddings
    emb1 = original_embs[idx1]
    emb2 = original_embs[idx2]

    # Make sure original embeddings are valid
    if not np.all(np.isfinite(emb1)) or not np.all(np.isfinite(emb2)):
        print(f"\nSkipping pair idx1={idx1}, idx2={idx2} because source embedding has non-finite values.")
        continue

    # 6. Compute cosine similarity
    # Since original_embs are expected to already be L2-normalized and emb_morph_l2 is normalized,
    # this dot product is cosine similarity.
    sim1 = float(np.dot(emb1, emb_morph_l2))
    sim2 = float(np.dot(emb2, emb_morph_l2))

    # 7. Sort so z1 is always the embedding closest to the morph
    if sim1 >= sim2:
        z1, z2 = emb1, emb2
        meta_z1_idx, meta_z2_idx = idx1, idx2
        meta_sim_z1, meta_sim_z2 = sim1, sim2
    else:
        z1, z2 = emb2, emb1
        meta_z1_idx, meta_z2_idx = idx2, idx1
        meta_sim_z1, meta_sim_z2 = sim2, sim1

    # Append to dataset
    dataset_z1.append(z1.astype(np.float32))
    dataset_z2.append(z2.astype(np.float32))
    dataset_c.append(emb_morph_l2.astype(np.float32))

    metadata_rows.append(
        {
            "morph_index": len(dataset_c) - 1,
            "z1_original_idx": meta_z1_idx,
            "z2_original_idx": meta_z2_idx,
            "z1_image_path": df.iloc[meta_z1_idx]["image_path"],
            "z2_image_path": df.iloc[meta_z2_idx]["image_path"],
            "z1_cosine_sim": meta_sim_z1,
            "z2_cosine_sim": meta_sim_z2,
        }
    )

    pbar.update(1)

pbar.close()

# ==========================================
# 4. Safety Check Before Saving
# ==========================================
if len(dataset_z1) == 0:
    raise RuntimeError(
        "No morph pairs were generated. This means the script failed before producing "
        "even one valid pair. Check the printed DeepFace error, image paths, and whether "
        "DeepFace.represent is returning the expected list of dictionaries."
    )

if len(dataset_c) < TARGET_PAIRS:
    print(
        f"\nWARNING: Requested {TARGET_PAIRS} pairs, but only generated {len(dataset_c)} "
        f"after {attempts} attempts."
    )

# ==========================================
# 5. Train / Val Split & Saving
# ==========================================
print("Converting to NumPy arrays...")

dataset_z1 = np.stack(dataset_z1, axis=0).astype(np.float32)
dataset_z2 = np.stack(dataset_z2, axis=0).astype(np.float32)
dataset_c = np.stack(dataset_c, axis=0).astype(np.float32)

num_generated = len(dataset_c)
split_idx = int(num_generated * TRAIN_RATIO)

train_z1, val_z1 = dataset_z1[:split_idx], dataset_z1[split_idx:]
train_z2, val_z2 = dataset_z2[:split_idx], dataset_z2[split_idx:]
train_c, val_c = dataset_c[:split_idx], dataset_c[split_idx:]

train_meta = metadata_rows[:split_idx]
val_meta = metadata_rows[split_idx:]

print(f"Saving Train Set ({len(train_z1)} pairs)...")
np.savez(
    OUT_DIR / "train_morphs.npz",
    z1=train_z1,
    z2=train_z2,
    c=train_c,
)

pd.DataFrame(train_meta).to_csv(
    OUT_DIR / "train_morphs_metadata.csv",
    index=False,
)

print(f"Saving Val Set ({len(val_z1)} pairs)...")
np.savez(
    OUT_DIR / "val_morphs.npz",
    z1=val_z1,
    z2=val_z2,
    c=val_c,
)

pd.DataFrame(val_meta).to_csv(
    OUT_DIR / "val_morphs_metadata.csv",
    index=False,
)

print("Done! Dataset is ready for training.")
print(f"Generated {num_generated} total morph pairs.")
print(f"Train pairs: {len(train_z1)}")
print(f"Val pairs:   {len(val_z1)}")

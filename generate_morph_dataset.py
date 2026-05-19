import json
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
# Input paths (adjust if necessary)
METADATA_CSV = Path("/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_metadata.csv")
EMBEDDINGS_NPY = Path("/nas-ctm01/homes/dacordeiro/Face-DM/arcface_embeddings/Face-DM/ffhq256_deepface_arcface_retinaface_l2norm.npy")

# Output directory for the new dataset
OUT_DIR = Path("/nas-ctm01/homes/dacordeiro/Face-DM/morphed_dataset/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PAIRS = 70000
TRAIN_RATIO = 0.9

# DeepFace Config (must match the original encoding to be perfectly aligned)
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
ALIGN = True
NORMALIZATION = "ArcFace"

# ==========================================
# 2. Helper Functions
# ==========================================
def choose_main_face(face_objs):
    """Select the largest detected face if multiple are found."""
    if len(face_objs) == 1:
        return face_objs

    def area(obj):
        fa = obj.get("facial_area", {})
        return float(fa.get("w", 0) * fa.get("h", 0))

    return max(face_objs, key=area)

# ==========================================
# 3. Main Logic
# ==========================================
print("Loading original metadata and embeddings...")
df = pd.read_csv(METADATA_CSV)
original_embs = np.load(EMBEDDINGS_NPY)

num_total_images = len(df)
print(f"Loaded {num_total_images} original embeddings.")

# Keep track of which pairs have already been attempted to ensure uniqueness
seen_pairs = set()

# Lists to store our final dataset elements
dataset_z1 = []   # Original embedding with HIGHEST cosine similarity to morph
dataset_z2 = []   # Original embedding with LOWEST cosine similarity to morph
dataset_c = []    # Morph embedding
metadata_rows = []

# Initialize progress bar
pbar = tqdm(total=TARGET_PAIRS, desc="Generating Morphs")

while len(dataset_c) < TARGET_PAIRS:
    # 1. Sample two distinct random indices
    idx1, idx2 = random.sample(range(num_total_images), 2)
    
    # Sort indices strictly for uniqueness tracking in the set
    pair_key = tuple(sorted((idx1, idx2)))
    if pair_key in seen_pairs:
        continue
    seen_pairs.add(pair_key)
    
    # 2. Load the actual source images
    path1 = df.iloc[idx1]['image_path']
    path2 = df.iloc[idx2]['image_path']
    
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    if img1 is None or img2 is None:
        continue # File not found or corrupt
        
    if img1.shape != img2.shape:
        # Resize img2 to img1 if sizes mismatch (shouldn't happen on ffhq256)
        img2 = cv2.resize(img2, (img1.shape, img1.shape))

    # 3. Create the average image (pixel-wise morph)
    avg_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    
    # 4. Extract ArcFace embedding from the average image
    try:
        # DeepFace can accept a numpy array directly (BGR standard from cv2)
        face_objs = DeepFace.represent(
            img_path=avg_img, 
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            align=ALIGN,
            enforce_detection=True, # Will throw exception if no face is detected
            normalization=NORMALIZATION,
        )
        
        obj = choose_main_face(face_objs)
        emb_morph = np.asarray(obj["embedding"], dtype=np.float32)
        
        # L2 Normalize
        norm = np.linalg.norm(emb_morph)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("Invalid embedding norm")
        emb_morph_l2 = emb_morph / norm
        
    except Exception:
        # Face detection failed, or extraction failed. Just skip to the next pair.
        continue
        
    # 5. Fetch source embeddings
    emb1 = original_embs[idx1]
    emb2 = original_embs[idx2]
    
    # 6. Compute Cosine Similarity (Since they are L2 normalized, it's just the dot product)
    sim1 = np.dot(emb1, emb_morph_l2)
    sim2 = np.dot(emb2, emb_morph_l2)
    
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
    dataset_z1.append(z1)
    dataset_z2.append(z2)
    dataset_c.append(emb_morph_l2)
    
    metadata_rows.append({
        "morph_index": len(dataset_c) - 1,
        "z1_original_idx": meta_z1_idx,
        "z2_original_idx": meta_z2_idx,
        "z1_cosine_sim": meta_sim_z1,
        "z2_cosine_sim": meta_sim_z2
    })
    
    pbar.update(1)

pbar.close()

# ==========================================
# 4. Train / Val Split & Saving
# ==========================================
print("Converting to Numpy arrays...")
dataset_z1 = np.stack(dataset_z1, axis=0).astype(np.float32)
dataset_z2 = np.stack(dataset_z2, axis=0).astype(np.float32)
dataset_c = np.stack(dataset_c, axis=0).astype(np.float32)

split_idx = int(TARGET_PAIRS * TRAIN_RATIO)

train_z1, val_z1 = dataset_z1[:split_idx], dataset_z1[split_idx:]
train_z2, val_z2 = dataset_z2[:split_idx], dataset_z2[split_idx:]
train_c, val_c   = dataset_c[:split_idx], dataset_c[split_idx:]

train_meta, val_meta = metadata_rows[:split_idx], metadata_rows[split_idx:]

print(f"Saving Train Set ({len(train_z1)} pairs)...")
np.savez(OUT_DIR / "train_morphs.npz", z1=train_z1, z2=train_z2, c=train_c)
pd.DataFrame(train_meta).to_csv(OUT_DIR / "train_morphs_metadata.csv", index=False)

print(f"Saving Val Set ({len(val_z1)} pairs)...")
np.savez(OUT_DIR / "val_morphs.npz", z1=val_z1, z2=val_z2, c=val_c)
pd.DataFrame(val_meta).to_csv(OUT_DIR / "val_morphs_metadata.csv", index=False)

print("Done! Dataset is ready for training.")
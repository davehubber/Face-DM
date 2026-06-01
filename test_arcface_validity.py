import re
import itertools
import numpy as np
from pathlib import Path
from tqdm import tqdm
from deepface import DeepFace

# --- Configuration ---
MORDIFF_DIR = Path("MorDIFF_crop")
FRLL_DIR = Path("FRLL")
REPORT_PATH = Path("morph_similarity_report.txt")

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
ALIGN = True
NORMALIZATION = "ArcFace"

# --- Helper Functions ---
def choose_main_face(face_objs):
    """Selects the largest face if multiple are detected."""
    if len(face_objs) == 1:
        return face_objs[0]

    def area(obj):
        fa = obj.get("facial_area", {})
        return float(fa.get("w", 0) * fa.get("h", 0))

    return max(face_objs, key=area)

def get_l2_normalized_embedding(img_path: Path) -> np.ndarray:
    """Extracts and L2-normalizes the ArcFace embedding."""
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

    if emb.ndim != 1:
        raise ValueError(f"Unexpected embedding shape: {emb.shape}")

    norm = np.linalg.norm(emb)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Invalid embedding norm")

    return emb / norm

def get_subject_id(image_id: str) -> str:
    """
    Extracts the base subject ID from the FRLL filename format.
    Assuming '036_08' -> subject is '036'. 
    This prevents comparing '036_08' to '036_03' as a bona fide inter-identity pair.
    """
    return image_id.split('_')[0]

# --- Main Pipeline ---
def main():
    print("Indexing FRLL base images...")
    base_images = {img.stem: img for img in FRLL_DIR.rglob("*.jpg")}
    print(f"Found {len(base_images)} base images.")

    print("Locating morph images...")
    morph_paths = list(MORDIFF_DIR.rglob("morphed/*.png"))
    print(f"Found {len(morph_paths)} morphs to process.")

    base_emb_cache = {}
    
    results = []
    errors = []

    filename_pattern = re.compile(r"morph_(.+)_and_(.+)")

    for morph_path in tqdm(morph_paths, desc="Evaluating Morphs"):
        match = filename_pattern.search(morph_path.stem)
        if not match:
            errors.append(f"{morph_path.name}: Filename does not match expected pattern.")
            continue
        
        id1, id2 = match.groups()

        base_path1 = base_images.get(id1)
        base_path2 = base_images.get(id2)

        if not base_path1 or not base_path2:
            errors.append(f"{morph_path.name}: Could not find base image(s) for {id1} or {id2}.")
            continue

        try:
            # 1 & 2. Get/Compute Base Embeddings
            if id1 not in base_emb_cache:
                base_emb_cache[id1] = get_l2_normalized_embedding(base_path1)
            emb1 = base_emb_cache[id1]

            if id2 not in base_emb_cache:
                base_emb_cache[id2] = get_l2_normalized_embedding(base_path2)
            emb2 = base_emb_cache[id2]

            # 3. Compute Morph Embedding
            emb_morph = get_l2_normalized_embedding(morph_path)

            # 4. Calculate NLERP Midpoint
            midpoint = emb1 + emb2
            midpoint_norm = np.linalg.norm(midpoint)
            if midpoint_norm == 0:
                raise ValueError("Base embeddings cancel each other out (perfect opposites).")
            nlerp_midpoint = midpoint / midpoint_norm

            # 5. Calculate Cosine Similarity
            cos_sim = float(np.dot(emb_morph, nlerp_midpoint))

            results.append({
                "morph": morph_path.name,
                "id1": id1,
                "id2": id2,
                "similarity": cos_sim
            })

        except Exception as e:
            errors.append(f"{morph_path.name}: Failed extraction - {repr(e)}")

    # --- Bona Fide Baseline Evaluation ---
    print("\nCalculating Bona Fide Inter-Identity Baseline...")
    bona_fide_similarities = []
    base_ids = list(base_emb_cache.keys())
    
    # Generate all unique pairs of extracted base images
    unique_pairs = list(itertools.combinations(base_ids, 2))
    
    for id1, id2 in tqdm(unique_pairs, desc="Evaluating Baseline Pairs"):
        # Ensure we are comparing entirely different subjects, not just different poses/expressions of the same subject
        if get_subject_id(id1) != get_subject_id(id2):
            emb1 = base_emb_cache[id1]
            emb2 = base_emb_cache[id2]
            sim = float(np.dot(emb1, emb2))
            bona_fide_similarities.append(sim)

    # --- Generate Report ---
    print("\nGenerating report...")
    if not results:
        print("No successful calculations to report.")
        return

    # Morph Stats
    morph_sims = [r["similarity"] for r in results]
    mean_sim = np.mean(morph_sims)
    std_sim = np.std(morph_sims)
    
    # Baseline Stats
    if bona_fide_similarities:
        bf_mean = np.mean(bona_fide_similarities)
        bf_std = np.std(bona_fide_similarities)
        bf_min = np.min(bona_fide_similarities)
        bf_max = np.max(bona_fide_similarities)
    else:
        bf_mean = bf_std = bf_min = bf_max = 0.0

    sorted_results = sorted(results, key=lambda x: x["similarity"])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("      Latent Space Morph Similarity Report          \n")
        f.write("====================================================\n\n")
        
        f.write("### BONA FIDE BASELINE (INTER-IDENTITY) ###\n")
        f.write("Expected similarity between two completely different real people.\n")
        f.write(f"Total Unique Pairs Evaluated : {len(bona_fide_similarities)}\n")
        f.write(f"Mean Cosine Similarity       : {bf_mean:.4f}\n")
        f.write(f"Std Deviation                : {bf_std:.4f}\n")
        f.write(f"Min Similarity               : {bf_min:.4f}\n")
        f.write(f"Max Similarity               : {bf_max:.4f}\n\n")

        f.write("### MORPH TO MIDPOINT OVERALL STATISTICS ###\n")
        f.write(f"Total Morphs Evaluated : {len(results)}\n")
        f.write(f"Mean Cosine Similarity : {mean_sim:.4f}\n")
        f.write(f"Std Deviation          : {std_sim:.4f}\n")
        f.write(f"Min Similarity         : {np.min(morph_sims):.4f}\n")
        f.write(f"Max Similarity         : {np.max(morph_sims):.4f}\n\n")

        f.write("### TOP 10 LOWEST SIMILARITIES (Worst Alignment) ###\n")
        for res in sorted_results[:10]:
            f.write(f"{res['morph']:<35} -> {res['similarity']:.4f}\n")
        f.write("\n")

        f.write("### TOP 10 HIGHEST SIMILARITIES (Best Alignment) ###\n")
        for res in reversed(sorted_results[-10:]):
            f.write(f"{res['morph']:<35} -> {res['similarity']:.4f}\n")
        f.write("\n")

        if errors:
            f.write("### ERRORS / FAILURES ###\n")
            f.write(f"Total Failures: {len(errors)}\n")
            for err in errors:
                f.write(f"- {err}\n")

    print(f"Done! Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()

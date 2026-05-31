import re
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

# --- Main Pipeline ---
def main():
    print("Indexing FRLL base images...")
    # Create a fast lookup dictionary for base images: { "036_08": Path(...) }
    base_images = {img.stem: img for img in FRLL_DIR.rglob("*.jpg")}
    print(f"Found {len(base_images)} base images.")

    print("Locating morph images...")
    # Find all pngs in the morphed subdirectories
    morph_paths = list(MORDIFF_DIR.rglob("morphed/*.png"))
    print(f"Found {len(morph_paths)} morphs to process.")

    # Cache for base embeddings to avoid redundant RetinaFace/ArcFace passes
    base_emb_cache = {}
    
    results = []
    errors = []

    # Regex to extract IDs (e.g., morph_036_08_and_130_08 -> 036_08, 130_08)
    filename_pattern = re.compile(r"morph_(.+)_and_(.+)")

    for morph_path in tqdm(morph_paths, desc="Evaluating Morphs"):
        match = filename_pattern.search(morph_path.stem)
        if not match:
            errors.append(f"{morph_path.name}: Filename does not match expected pattern.")
            continue
        
        id1, id2 = match.groups()

        # Locate base images
        base_path1 = base_images.get(id1)
        base_path2 = base_images.get(id2)

        if not base_path1 or not base_path2:
            errors.append(f"{morph_path.name}: Could not find base image(s) for {id1} or {id2}.")
            continue

        try:
            # 1. Get/Compute Base 1 Embedding
            if id1 not in base_emb_cache:
                base_emb_cache[id1] = get_l2_normalized_embedding(base_path1)
            emb1 = base_emb_cache[id1]

            # 2. Get/Compute Base 2 Embedding
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

            # 5. Calculate Cosine Similarity (Dot product of L2 normalized vectors)
            cos_sim = float(np.dot(emb_morph, nlerp_midpoint))

            results.append({
                "morph": morph_path.name,
                "id1": id1,
                "id2": id2,
                "similarity": cos_sim
            })

        except Exception as e:
            errors.append(f"{morph_path.name}: Failed extraction - {repr(e)}")

    # --- Generate Report ---
    print("\nGenerating report...")
    if not results:
        print("No successful calculations to report.")
        return

    similarities = [r["similarity"] for r in results]
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)

    # Sort results for top/bottom viewing
    sorted_results = sorted(results, key=lambda x: x["similarity"])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("      Latent Space Morph Similarity Report          \n")
        f.write("====================================================\n\n")
        
        f.write("### OVERALL STATISTICS ###\n")
        f.write(f"Total Morphs Evaluated : {len(results)}\n")
        f.write(f"Mean Cosine Similarity : {mean_sim:.4f}\n")
        f.write(f"Std Deviation          : {std_sim:.4f}\n")
        f.write(f"Min Similarity         : {min_sim:.4f}\n")
        f.write(f"Max Similarity         : {max_sim:.4f}\n\n")

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
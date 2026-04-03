import os
import torch
import argparse
import pandas as pd
import torch.nn.functional as F
from diffusers import StableUnCLIPImg2ImgPipeline

def get_decoder_pipeline(device):
    """Loads the UnCLIP image-to-image pipeline for decoding."""
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "sd2-community/stable-diffusion-2-1-unclip-small",
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline

def analyze_and_decode_embeddings(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")

    # 1. Load the partition map and embeddings
    print("Loading data...")
    df = pd.read_csv(args.csv_file)
    df = df[df["Partition"] == args.partition].reset_index(drop=True)
    
    embeds = torch.load(args.embed_file, map_location="cpu", weights_only=True)
    
    if len(df) != len(embeds):
        raise ValueError("Mismatch between partition.csv length and embeddings tensor length.")

    # 2. Deduplicate embeddings (since create_embeddings.py creates 1-to-k pairs)
    print("Deduplicating embeddings...")
    unique_files = []
    unique_embeds = []
    seen_files = set()

    for i, row in df.iterrows():
        img_path = row["Image1_Path"]
        if img_path not in seen_files:
            seen_files.add(img_path)
            unique_files.append(img_path)
            unique_embeds.append(embeds[i])

    unique_embeds = torch.stack(unique_embeds).to(torch.float32)
    print(f"Found {len(unique_files)} unique embeddings.")

    # 3. Calculate Average Embedding
    avg_embed = unique_embeds.mean(dim=0, keepdim=True)

    # 4. Compute Semantic Distance (Cosine Distance = 1 - Cosine Similarity)
    # Cosine distance is the preferred metric for CLIP-based latent spaces
    print("Computing semantic distances...")
    cos_sim = F.cosine_similarity(unique_embeds, avg_embed.expand_as(unique_embeds), dim=-1)
    distances = 1.0 - cos_sim

    # 5. Find Closest, Farthest, and Mean
    mean_dist = distances.mean().item()
    
    closest_idx = torch.argmin(distances).item()
    closest_dist = distances[closest_idx].item()
    closest_file = unique_files[closest_idx]

    farthest_idx = torch.argmax(distances).item()
    farthest_dist = distances[farthest_idx].item()
    farthest_file = unique_files[farthest_idx]

    # 6. Write the report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "semantic_distance_report.txt")
    
    print(f"Writing report to {report_path}...")
    with open(report_path, "w") as f:
        f.write("--- Semantic Distance Report (Dataset 1) ---\n")
        f.write(f"Average Distance:  {mean_dist:.6f}\n")
        f.write(f"Closest Distance:  {closest_dist:.6f} ({closest_file})\n")
        f.write(f"Farthest Distance: {farthest_dist:.6f} ({farthest_file})\n")
        f.write("-" * 44 + "\n\n")
        f.write("Individual Distances:\n")
        
        # Sort indices by distance for a cleaner report (closest first)
        sorted_indices = torch.argsort(distances)
        for idx in sorted_indices.tolist():
            f.write(f"{unique_files[idx]}: {distances[idx]:.6f}\n")

    # 7. Decode the average embedding back into an image
    print("Loading UnCLIP Pipeline to decode the average embedding...")
    pipeline = get_decoder_pipeline(device)
    
    print("Decoding image...")
    avg_embed = avg_embed.to(device=device, dtype=pipeline.unet.dtype)
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    with torch.no_grad():
        decoded_image = pipeline(
            prompt=[""], 
            image_embeds=avg_embed, 
            noise_level=0, 
            num_inference_steps=20,
            generator=generator
        ).images[0]

    image_path = os.path.join(args.output_dir, "decoded_average_embedding.png")
    decoded_image.save(image_path)
    print(f"Successfully saved decoded average embedding to: {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and decode average embeddings")
    parser.add_argument("--embed_file", type=str, required=True, help="Path to the dataset1 embeds .pt file")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to partition.csv")
    parser.add_argument("--partition", type=str, default="train", choices=["train", "test"], help="Partition to analyze")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save the report and image")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the diffusion decoder")
    
    args = parser.parse_args()
    analyze_and_decode_embeddings(args)

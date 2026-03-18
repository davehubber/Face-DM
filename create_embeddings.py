import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import argparse
from tqdm import tqdm
import random

def process_dataset(folder_path, processor, model, device):
    embeddings = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    with torch.no_grad():
        for f in tqdm(file_names, desc=f"Processing {folder_path}"):
            img = Image.open(os.path.join(folder_path, f)).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            embeds = model(**inputs).image_embeds
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(embeds.cpu().squeeze(0))
            
    return torch.stack(embeddings), file_names

def generate_embedding_dataset(folder1, folder2, output_dir="data_embeddings", seed=42, split_ratio=0.9):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device)
    model.eval()

    embeds1, files1 = process_dataset(folder1, processor, model, device)
    embeds2, files2 = process_dataset(folder2, processor, model, device)

    print("Building 1-to-5 pairs...")
    paired_embeds1 = []
    paired_embeds2 = []
    paired_files1 = []
    paired_files2 = []
    
    for i, e1 in enumerate(embeds1):
        num_samples = min(5, len(embeds2)) 
        sampled_indices = random.sample(range(len(embeds2)), num_samples)
        
        for j in sampled_indices:
            paired_embeds1.append(e1)
            paired_embeds2.append(embeds2[j])
            paired_files1.append(files1[i])
            paired_files2.append(files2[j])
            
    paired_embeds1 = torch.stack(paired_embeds1)
    paired_embeds2 = torch.stack(paired_embeds2)
    
    total_pairs = len(paired_embeds1)
    shuffle_idx = torch.randperm(total_pairs)
    
    paired_embeds1 = paired_embeds1[shuffle_idx]
    paired_embeds2 = paired_embeds2[shuffle_idx]
    
    paired_files1 = [paired_files1[i] for i in shuffle_idx.tolist()]
    paired_files2 = [paired_files2[i] for i in shuffle_idx.tolist()]

    split_idx = int(total_pairs * split_ratio)
    
    train_e1, test_e1 = paired_embeds1[:split_idx], paired_embeds1[split_idx:]
    train_e2, test_e2 = paired_embeds2[:split_idx], paired_embeds2[split_idx:]
    
    torch.save(train_e1, os.path.join(output_dir, "train_dataset1_embeds.pt"))
    torch.save(train_e2, os.path.join(output_dir, "train_dataset2_embeds.pt"))
    torch.save(test_e1, os.path.join(output_dir, "test_dataset1_embeds.pt"))
    torch.save(test_e2, os.path.join(output_dir, "test_dataset2_embeds.pt"))

    df = pd.DataFrame({
        "Image1_Path": paired_files1,
        "Image2_Path": paired_files2,
        "Partition": ["train"] * split_idx + ["test"] * (total_pairs - split_idx)
    })
    df.to_csv(os.path.join(output_dir, "partition.csv"), index=False)
    
    print(f"Successfully generated {total_pairs} pairs.")
    print(f"Train size: {split_idx} | Test size: {total_pairs - split_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", required=True, help="Path to Dataset 1")
    parser.add_argument("--folder2", required=True, help="Path to Dataset 2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and pairing")
    args = parser.parse_args()
    generate_embedding_dataset(args.folder1, args.folder2, seed=args.seed)
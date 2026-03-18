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


def split_embeddings_and_files(embeddings, file_names, split_ratio):
    num_samples = len(file_names)
    shuffled_indices = torch.randperm(num_samples)
    split_idx = int(num_samples * split_ratio)

    train_idx = shuffled_indices[:split_idx]
    test_idx = shuffled_indices[split_idx:]

    train_embeds = embeddings[train_idx]
    test_embeds = embeddings[test_idx]
    train_files = [file_names[i] for i in train_idx.tolist()]
    test_files = [file_names[i] for i in test_idx.tolist()]

    return train_embeds, train_files, test_embeds, test_files


def build_pairs(embeds1, files1, embeds2, files2):
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

    return paired_embeds1, paired_embeds2, paired_files1, paired_files2


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

    train_embeds1, train_files1, test_embeds1, test_files1 = split_embeddings_and_files(embeds1, files1, split_ratio)
    train_embeds2, train_files2, test_embeds2, test_files2 = split_embeddings_and_files(embeds2, files2, split_ratio)

    print("Building 1-to-5 train pairs...")
    train_e1, train_e2, train_pair_files1, train_pair_files2 = build_pairs(
        train_embeds1, train_files1, train_embeds2, train_files2
    )

    print("Building 1-to-5 test pairs...")
    test_e1, test_e2, test_pair_files1, test_pair_files2 = build_pairs(
        test_embeds1, test_files1, test_embeds2, test_files2
    )

    torch.save(train_e1, os.path.join(output_dir, "train_dataset1_embeds.pt"))
    torch.save(train_e2, os.path.join(output_dir, "train_dataset2_embeds.pt"))
    torch.save(test_e1, os.path.join(output_dir, "test_dataset1_embeds.pt"))
    torch.save(test_e2, os.path.join(output_dir, "test_dataset2_embeds.pt"))

    df = pd.DataFrame({
        "Image1_Path": train_pair_files1 + test_pair_files1,
        "Image2_Path": train_pair_files2 + test_pair_files2,
        "Partition": ["train"] * len(train_pair_files1) + ["test"] * len(test_pair_files1)
    })
    df.to_csv(os.path.join(output_dir, "partition.csv"), index=False)

    total_pairs = len(train_e1) + len(test_e1)
    print(f"Successfully generated {total_pairs} pairs.")
    print(f"Train size: {len(train_e1)} | Test size: {len(test_e1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", required=True, help="Path to Dataset 1")
    parser.add_argument("--folder2", required=True, help="Path to Dataset 2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and pairing")
    args = parser.parse_args()
    generate_embedding_dataset(args.folder1, args.folder2, seed=args.seed)

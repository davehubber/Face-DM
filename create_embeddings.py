import os
import torch
import pandas as pd
from PIL import Image
import argparse
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

UNCLIP_MODEL_ID = "sd2-community/stable-diffusion-2-1-unclip-small"

class ImageFolderDataset(Dataset):
    """Custom Dataset for fast multi-threaded image loading."""
    def __init__(self, folder_path, file_names):
        self.folder_path = folder_path
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.file_names[idx])
        # Open and ensure it is RGB
        with Image.open(file_path) as img:
            return img.convert("RGB")

def load_unclip_encoder_components(device):
    """Loads ONLY the required image encoder and processor to save VRAM and time."""
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    processor = CLIPImageProcessor.from_pretrained(
        UNCLIP_MODEL_ID, 
        subfolder="feature_extractor"
    )
    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        UNCLIP_MODEL_ID,
        subfolder="image_encoder",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    image_encoder.eval()

    return processor, image_encoder


def process_dataset(folder_path, processor, model, device, batch_size=32, num_workers=4):
    """Processes images in parallel batches rather than one by one."""
    file_names = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )

    dataset = ImageFolderDataset(folder_path, file_names)
    model_dtype = next(model.parameters()).dtype

    def collate_fn(images):
        return processor(images=images, return_tensors="pt")["pixel_values"]

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        collate_fn=collate_fn,
        pin_memory=True if str(device).startswith("cuda") else False
    )

    embeddings = []

    with torch.no_grad():
        for pixel_values in tqdm(dataloader, desc=f"Processing {folder_path}"):
            pixel_values = pixel_values.to(device=device, dtype=model_dtype)
            embeds = model(pixel_values=pixel_values).image_embeds
            embeddings.append(embeds.cpu().to(torch.float32))

    # Concat once at the end instead of appending individual tensors
    return torch.cat(embeddings, dim=0), file_names


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


def build_pairs_two_datasets(embeds1, files1, embeds2, files2, pairs_per_image=10):
    """Vectorized pairing for two datasets."""
    idx1, idx2 = [], []
    num_embeds2 = len(embeds2)

    for i in range(len(embeds1)):
        num_samples = min(pairs_per_image, num_embeds2)
        sampled_indices = random.sample(range(num_embeds2), num_samples)
        
        idx1.extend([i] * num_samples)
        idx2.extend(sampled_indices)

    # Fast vectorized indexing
    idx1_tensor = torch.tensor(idx1)
    idx2_tensor = torch.tensor(idx2)

    paired_embeds1 = embeds1[idx1_tensor]
    paired_embeds2 = embeds2[idx2_tensor]
    paired_files1 = [files1[i] for i in idx1]
    paired_files2 = [files2[i] for i in idx2]

    # Shuffle
    shuffle_idx = torch.randperm(len(idx1_tensor))
    
    return (
        paired_embeds1[shuffle_idx], 
        paired_embeds2[shuffle_idx], 
        [paired_files1[i] for i in shuffle_idx.tolist()], 
        [paired_files2[i] for i in shuffle_idx.tolist()]
    )


def build_pairs_single_dataset(embeds, files, pairs_per_image=10, avoid_self_pairs=True):
    """Vectorized pairing for a single dataset."""
    n = len(embeds)
    if n < 2 and avoid_self_pairs:
        raise ValueError("Single-dataset mode with avoid_self_pairs=True requires at least 2 images.")

    idx1, idx2 = [], []

    for i in range(n):
        if avoid_self_pairs:
            candidate_indices = [j for j in range(n) if j != i]
        else:
            candidate_indices = list(range(n))

        if not candidate_indices:
            continue

        num_samples = min(pairs_per_image, len(candidate_indices))
        sampled_indices = random.sample(candidate_indices, num_samples)

        idx1.extend([i] * num_samples)
        idx2.extend(sampled_indices)

    # Fast vectorized indexing
    idx1_tensor = torch.tensor(idx1)
    idx2_tensor = torch.tensor(idx2)

    paired_embeds1 = embeds[idx1_tensor]
    paired_embeds2 = embeds[idx2_tensor]
    paired_files1 = [files[i] for i in idx1]
    paired_files2 = [files[i] for i in idx2]

    # Shuffle
    shuffle_idx = torch.randperm(len(idx1_tensor))
    
    return (
        paired_embeds1[shuffle_idx], 
        paired_embeds2[shuffle_idx], 
        [paired_files1[i] for i in shuffle_idx.tolist()], 
        [paired_files2[i] for i in shuffle_idx.tolist()]
    )


def generate_embedding_dataset(
    folder1,
    folder2=None,
    output_dir="data_embeddings",
    seed=42,
    split_ratio=0.9,
    pairs_per_image=10,
    single_dataset=False,
    avoid_self_pairs=True,
    batch_size=32,
    num_workers=4
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(seed)
    torch.manual_seed(seed)

    processor, model = load_unclip_encoder_components(device)

    # Encode dataset 1 using Batched Processing
    embeds1, files1 = process_dataset(folder1, processor, model, device, batch_size, num_workers)

    if single_dataset:
        print("Running in single-dataset mode.")
        train_embeds, train_files, test_embeds, test_files = split_embeddings_and_files(
            embeds1, files1, split_ratio
        )

        print("Building 1-to-k train pairs from dataset 1 with itself...")
        train_e1, train_e2, train_pair_files1, train_pair_files2 = build_pairs_single_dataset(
            train_embeds,
            train_files,
            pairs_per_image=pairs_per_image,
            avoid_self_pairs=avoid_self_pairs,
        )

        print("Building 1-to-k test pairs from dataset 1 with itself...")
        test_e1, test_e2, test_pair_files1, test_pair_files2 = build_pairs_single_dataset(
            test_embeds,
            test_files,
            pairs_per_image=pairs_per_image,
            avoid_self_pairs=avoid_self_pairs,
        )

    else:
        if folder2 is None:
            raise ValueError("folder2 must be provided unless --single_dataset is used.")

        print("Running in two-dataset mode.")
        embeds2, files2 = process_dataset(folder2, processor, model, device, batch_size, num_workers)

        train_embeds1, train_files1, test_embeds1, test_files1 = split_embeddings_and_files(
            embeds1, files1, split_ratio
        )
        train_embeds2, train_files2, test_embeds2, test_files2 = split_embeddings_and_files(
            embeds2, files2, split_ratio
        )

        print("Building 1-to-k train pairs...")
        train_e1, train_e2, train_pair_files1, train_pair_files2 = build_pairs_two_datasets(
            train_embeds1, train_files1, train_embeds2, train_files2, pairs_per_image=pairs_per_image
        )

        print("Building 1-to-k test pairs...")
        test_e1, test_e2, test_pair_files1, test_pair_files2 = build_pairs_two_datasets(
            test_embeds1, test_files1, test_embeds2, test_files2, pairs_per_image=pairs_per_image
        )

    torch.save(train_e1, os.path.join(output_dir, "train_dataset1_embeds.pt"))
    torch.save(train_e2, os.path.join(output_dir, "train_dataset2_embeds.pt"))
    torch.save(test_e1, os.path.join(output_dir, "test_dataset1_embeds.pt"))
    torch.save(test_e2, os.path.join(output_dir, "test_dataset2_embeds.pt"))

    df = pd.DataFrame(
        {
            "Image1_Path": train_pair_files1 + test_pair_files1,
            "Image2_Path": train_pair_files2 + test_pair_files2,
            "Partition": ["train"] * len(train_pair_files1) + ["test"] * len(test_pair_files1),
        }
    )
    df.to_csv(os.path.join(output_dir, "partition.csv"), index=False)

    total_pairs = len(train_e1) + len(test_e1)
    print(f"Successfully generated {total_pairs} pairs.")
    print(f"Train size: {len(train_e1)} | Test size: {len(test_e1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder1", required=True, help="Path to Dataset 1")
    parser.add_argument("--folder2", default=None, help="Path to Dataset 2")
    parser.add_argument("--output_dir", default="data_embeddings", help="Where to save the embedding dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling and pairing")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/test split ratio")
    parser.add_argument("--pairs_per_image", type=int, default=10, help="How many pairs to sample per image")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU workers for dataloader")
    parser.add_argument(
        "--single_dataset",
        action="store_true",
        help="If set, encode folder1 once and build pairs within the same dataset",
    )
    parser.add_argument(
        "--allow_self_pairs",
        action="store_true",
        help="If set, allows an image to be paired with itself in single-dataset mode",
    )

    args = parser.parse_args()

    generate_embedding_dataset(
        folder1=args.folder1,
        folder2=args.folder2,
        output_dir=args.output_dir,
        seed=args.seed,
        split_ratio=args.split_ratio,
        pairs_per_image=args.pairs_per_image,
        single_dataset=args.single_dataset,
        avoid_self_pairs=not args.allow_self_pairs,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
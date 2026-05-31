import sys
import csv
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Setup paths to your repositories
PATH_TO_DIFF_MODEL = "../diffae"
sys.path.append(PATH_TO_DIFF_MODEL)

from templates import ffhq256_autoenc
from experiment import LitModel

# ==========================================
# 1. Benchmark Dataset Loader
# ==========================================
class BenchmarkDataset(Dataset):
    def __init__(self, image_paths, pre_saved_embeddings, img_size):
        self.image_paths = image_paths
        self.embeddings = pre_saved_embeddings
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load physical image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img_tensor = self.transform(img)
        # Load corresponding normalized semantic embedding vector
        emb_tensor = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        return img_tensor, emb_tensor

# ==========================================
# 2. Main Benchmarking Pipeline
# ==========================================
def run_profile_benchmarks(base_path_str: str, num_samples: int = 1000, batch_size: int = 10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running profile benchmarks on device: {device}")

    base_path = Path(base_path_str).resolve()
    parent = base_path.parent
    stem = base_path.stem.replace("_train", "").replace("_val", "").replace("_test", "")
    
    master_npy_path = parent / f"{stem}.npy"
    metadata_csv_path = parent / f"{stem}_metadata.csv"
    mean_path = parent / f"{stem}_train_mean.npy"
    std_path = parent / f"{stem}_train_std.npy"
    
    # 1. Reconstruct deterministic train split mapping
    print("Loading master embedding references and normalizing...")
    master_embeddings = np.load(master_npy_path).astype(np.float32)
    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)
    normalized_master = (master_embeddings - mean) / std
    
    total_dataset_size = len(normalized_master)
    
    # Re-run same seed shuffle to isolate training split boundaries
    np.random.seed(42)
    split_indices = np.arange(total_dataset_size)
    np.random.shuffle(split_indices)
    train_end = int(total_dataset_size * 0.8)
    train_indices = split_indices[:train_end]
    
    # Sample 1000 random indices from the isolated training split
    np.random.seed(123)  # Distinct seed to pick random subset across train space
    selected_train_indices = np.random.choice(train_indices, size=num_samples, replace=False)
    
    # Map index integers to file paths from metadata CSV
    idx_to_path = {}
    with open(metadata_csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_path[int(row['embedding_index'])] = row['image_path']
            
    selected_paths = [idx_to_path[i] for i in selected_train_indices]
    selected_embs = normalized_master[selected_train_indices]

    # 2. Initialize Model
    conf = ffhq256_autoenc()
    model = LitModel(conf)
    state = torch.load(f'{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    # 3. Create DataLoader
    dataset = BenchmarkDataset(selected_paths, selected_embs, conf.img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Timing accumulators (in milliseconds)
    time_semantic_encode = 0.0
    time_stochastic_encode = 0.0
    time_decode_random = 0.0
    time_decode_exact = 0.0

    # CUDA event instantiations for high-precision tracking
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    print(f"Beginning operational benchmarks over {num_samples} samples (Batch Size: {batch_size})...")
    
    # Warmup pass to eliminate initial initialization/lazy allocation noise
    dummy_img = torch.randn(1, 3, conf.img_size, conf.img_size, device=device)
    dummy_cond = model.encode(dummy_img)
    _ = model.encode_stochastic(dummy_img, dummy_cond, T=10)
    _ = model.render(torch.randn_like(_), dummy_cond, T=5)
    torch.cuda.synchronize()

    for batch_imgs, batch_saved_embs in tqdm(loader, desc="Benchmarking Operations"):
        batch_imgs = batch_imgs.to(device)
        batch_saved_embs = batch_saved_embs.to(device)
        
        # ----------------------------------------------------
        # Operation 1: Encode Semantic Codes
        # ----------------------------------------------------
        torch.cuda.synchronize()
        start_evt.record()
        cond = model.encode(batch_imgs)
        end_evt.record()
        torch.cuda.synchronize()
        time_semantic_encode += start_evt.elapsed_time(end_evt)

        # ----------------------------------------------------
        # Operation 2: Encode Stochastic Codes (T=250 steps)
        # ----------------------------------------------------
        torch.cuda.synchronize()
        start_evt.record()
        xT = model.encode_stochastic(batch_imgs, cond, T=250)
        end_evt.record()
        torch.cuda.synchronize()
        time_stochastic_encode += start_evt.elapsed_time(end_evt)

        # ----------------------------------------------------
        # Operation 3: Decode via Pre-Saved Embs + Random Stochastic Noise (T=20 steps)
        # ----------------------------------------------------
        # Sample standard Gaussian noise matching structural tensor grid dimensions
        xT_random = torch.randn_like(xT)
        
        torch.cuda.synchronize()
        start_evt.record()
        _ = model.render(xT_random, batch_saved_embs, T=20)
        end_evt.record()
        torch.cuda.synchronize()
        time_decode_random += start_evt.elapsed_time(end_evt)

        # ----------------------------------------------------
        # Operation 4: Decode via Pre-Saved Embs + Encoded Stochastic Codes (T=20 steps)
        # ----------------------------------------------------
        torch.cuda.synchronize()
        start_evt.record()
        _ = model.render(xT, batch_saved_embs, T=20)
        end_evt.record()
        torch.cuda.synchronize()
        time_decode_exact += start_evt.elapsed_time(end_evt)

    # Convert accumulated metrics from milliseconds to total seconds
    sec_semantic_enc = time_semantic_encode / 1000.0
    sec_stochastic_enc = time_stochastic_encode / 1000.0
    sec_dec_rand = time_decode_random / 1000.0
    sec_dec_exact = time_decode_exact / 1000.0

    # 4. Generate Performance Summary Report
    report_text = (
        f"=========================================================================\n"
        f"          DIFFUSION AUTOENCODER (DIFF-AE) BENCHMARK PROFILE REPORT\n"
        f"=========================================================================\n"
        f"Total Images Evaluated:     {num_samples}\n"
        f"Execution Batch Size:       {batch_size}\n"
        f"Hardware Target Resource:   {torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'}\n"
        f"-------------------------------------------------------------------------\n\n"
        f"1. ENCODE SEMANTIC CODES (z_sem)\n"
        f"   - Total Execution Time:  {sec_semantic_enc:.4f} seconds\n"
        f"   - Average Time per Image: {(sec_semantic_enc / num_samples) * 1000:.2f} ms\n\n"
        f"2. ENCODE STOCHASTIC CODES (x_T, T=250 steps)\n"
        f"   - Total Execution Time:  {sec_stochastic_enc:.4f} seconds\n"
        f"   - Average Time per Image: {(sec_stochastic_enc / num_samples) * 1000:.2f} ms\n\n"
        f"3. DECODE BACK TO IMAGES (Pre-Saved z_sem + Random x_T, T=20 steps)\n"
        f"   - Total Execution Time:  {sec_dec_rand:.4f} seconds\n"
        f"   - Average Time per Image: {(sec_dec_rand / num_samples) * 1000:.2f} ms\n\n"
        f"4. DECODE BACK TO IMAGES (Pre-Saved z_sem + Encoded x_T, T=20 steps)\n"
        f"   - Total Execution Time:  {sec_dec_exact:.4f} seconds\n"
        f"   - Average Time per Image: {(sec_dec_exact / num_samples) * 1000:.2f} ms\n"
        f"=========================================================================\n"
    )

    output_file = parent / "diffae_operation_benchmarks.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print("\n" + report_text)
    print(f"[SUCCESS] High-precision profile logs saved -> {output_file}")

if __name__ == "__main__":
    DATA_PATH = "/nas-ctm01/homes/dacordeiro/Face-DM/diffae_embeddings/ffhq256_diffae_zsem.npy"
    # Adjust batch_size lower if you encounter VRAM headroom issues on your GPU resource
    run_profile_benchmarks(base_path_str=DATA_PATH, num_samples=1000, batch_size=10)

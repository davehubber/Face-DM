import torch
import os

data_dir = "data_embeddings"

embeds1 = torch.load(os.path.join(data_dir, "train_dataset1_embeds.pt"))
embeds2 = torch.load(os.path.join(data_dir, "train_dataset2_embeds.pt"))

all_embeds = torch.cat([embeds1, embeds2], dim=0)

dataset_std = torch.std(all_embeds)

optimal_scale_factor = 1.0 / dataset_std.item()

print(f"Dataset 1 Std Dev: {torch.std(embeds1).item():.4f}")
print(f"Dataset 2 Std Dev: {torch.std(embeds2).item():.4f}")
print(f"Combined Std Dev: {dataset_std.item():.4f}")
print(f"Optimal Scale Factor: {optimal_scale_factor:.4f}")
import os, numpy as np, pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as TF
import random
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, obtained_images, original_images, added_images, path, **kwargs):
    input_images = (original_images * 0.5 + added_images * 0.5).type(torch.uint8)
    
    B, C, H, W = original_images.shape
    all_images = torch.stack([
        original_images, 
        added_images, 
        input_images, 
        images, 
        obtained_images
    ], dim=1).view(-1, C, H, W)
    
    grid = torchvision.utils.make_grid(all_images, nrow=5, padding=2, pad_value=255, **kwargs)
    
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy().astype(np.uint8)
    
    im = Image.fromarray(ndarr)
    im.save(path)

def save_image_sampling(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, nrow=8, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

class TheDataset(Dataset):
    def __init__(self, dataset_path, partition, csv_file, transform=None):
        self.dataset_path = dataset_path
        self.partition = partition
        
        df = pd.read_csv(csv_file, sep=",")

        if 'partition' in df.columns:
            df = df[df['partition'] == partition]

        self.image_paths_1 = np.asarray(df['Image1'].values)
        self.image_paths_2 = np.asarray(df['Image2'].values)

        self.transform = transform
        print('Size of ' + partition + ': ' + str(len(self.image_paths_1)))
        
    def __getitem__(self, index):
        x1 = Image.open(os.path.join(self.dataset_path, self.image_paths_1[index])).convert("RGB")
        x2 = Image.open(os.path.join(self.dataset_path, self.image_paths_2[index])).convert("RGB")
        
        if self.partition == 'train':
            if random.random() > 0.5:
                x1 = TF.hflip(x1)
                x2 = TF.hflip(x2)

        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
            
        return x1, x2
    
    def __len__(self):
        return len(self.image_paths_1)

def get_data(args, partition):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = TheDataset(args.dataset_path, partition, args.partition_file, transform=transforms)
    
    num_workers = min(os.cpu_count() or 1, 8) 
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(partition == 'train'), 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )
    return dataloader

def setup_logging(run_name):
    base_dir = os.path.join("experiments", run_name)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "train_fixed"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "eval"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "samples", "one_shot"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    return base_dir

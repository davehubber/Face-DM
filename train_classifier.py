import argparse
import os
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import the existing stat loader from your utils
from utils_semantic import load_zscore_stats


class RealVsAverageDataset(Dataset):
    def __init__(self, dataset_root: str, split: str, num_samples: int):
        """
        Generates a dataset where 50% of the samples are real (clean) embeddings,
        and 50% are the average of two random real embeddings.
        """
        split_path = os.path.join(dataset_root, "semantic", f"{split}_zsem.pt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Could not find: {split_path}")

        pack = torch.load(split_path, map_location="cpu")
        mean, std = load_zscore_stats(dataset_root)
        
        # Apply the exact same z-score normalization as your diffusion model
        self.embeddings = (pack["z_sem"].to(torch.float32) - mean) / std
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 50% chance of being an averaged embedding (Label 1) vs Real (Label 0)
        is_average = random.random() < 0.5
        
        if is_average:
            idx1, idx2 = random.sample(range(len(self.embeddings)), 2)
            # Create the mixed embedding (alpha = 0.5)
            emb = (self.embeddings[idx1] + self.embeddings[idx2]) / 2.0
            label = 1.0
        else:
            idx1 = random.randint(0, len(self.embeddings) - 1)
            emb = self.embeddings[idx1]
            label = 0.0
            
        return emb, torch.tensor([label], dtype=torch.float32)


class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        # A simple MLP for binary classification
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1) # Output logits
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_dataset = RealVsAverageDataset(args.dataset_root, "train", args.train_samples)
    val_dataset = RealVsAverageDataset(args.dataset_root, "val", args.val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Setup Model
    embedding_dim = train_dataset.embeddings.shape
    model = EmbeddingClassifier(input_dim=embedding_dim, hidden_dim=args.hidden_dim).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss() # Combines Sigmoid and Binary Cross Entropy

    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for embeddings, labels in pbar:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Calculate accuracy
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_acc = train_correct / train_total

        # 4. Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                logits = model(embeddings)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} Summary: "
              f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="encoded_ffhq256_semantic_split", help="Path to your dataset root")
    parser.add_argument("--train_samples", type=int, default=100000, help="Number of samples to generate per epoch")
    parser.add_argument("--val_samples", type=int, default=20000, help="Number of validation samples")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)

    args = parser.parse_args()
    train_classifier(args)
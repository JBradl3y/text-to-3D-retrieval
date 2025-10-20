import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset.pointcloud_dataset import PointCloudCLIPDataset, collate_pointcloud
from src.models.pointnet import PointNetEmbedding
from tqdm import tqdm
import argparse
from src.dataset.split_dataset import DatasetSplitter
import json

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train PointNet with Contrastive Learning")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Dimension of the embedding")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for contrastive loss")
    parser.add_argument("--csv_path", type=str, default="data/captions.tablechair.emb.csv", help="Path to the CSV file")
    parser.add_argument("--pcd_root", type=str, default="data/text2shape_2048_pointcloud", help="Root directory for point cloud data")
    return parser.parse_args()

# --- Contrastive Loss ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        dist = torch.norm(emb1 - emb2, p=2, dim=1)
        loss = torch.mean(label * dist**2 + (1 - label) * torch.clamp(self.margin - dist, min=0.0)**2)
        return loss

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# --- Training Function ---
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        points = batch["points"].to(device)  # [B, C, N]
        clip_emb = batch["clip_emb"].to(device)  # [B, EMB_DIM]

        optimizer.zero_grad()

        # Forward pass
        point_emb = model(points)  # [B, EMB_DIM]

        # Compute contrastive loss
        labels = torch.eye(points.size(0), device=device)  # Identity matrix as labels
        loss = 0
        for i in range(points.size(0)):
            for j in range(points.size(0)):
                label = labels[i, j]
                loss += criterion(point_emb[i].unsqueeze(0), clip_emb[j].unsqueeze(0), label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# --- Main Training Script ---
if __name__ == "__main__":
    # --- Hyperparameters ---
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = f"results/point_cloud_contrastive/lr_{args.learning_rate}/margins_{args.margin}/"
    os.makedirs(results_dir, exist_ok=True)

    # Save training parameters to a JSON file
    params = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "embedding_dim": args.embedding_dim,
        "margin": args.margin,
        "csv_path": args.csv_path,
        "pcd_root": args.pcd_root,
        "device": str(device)
    }
    with open(os.path.join(results_dir, "training_params.json"), "w") as f:
        json.dump(params, f, indent=4)
    print("Training parameters saved to training_params.json")

    # Split dataset into train, val, and test
    splitter = DatasetSplitter(args.csv_path)
    train_df, val_df, _ = splitter.split_by_category()

    # Load training dataset and dataloader
    train_dataset = PointCloudCLIPDataset(train_df, args.pcd_root, use_color=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_pointcloud,
    )

    # Load validation dataset and dataloader
    val_dataset = PointCloudCLIPDataset(val_df, args.pcd_root, use_color=True)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_pointcloud,
    )

    # Initialize models
    pointnet = PointNetEmbedding(input_dim=6, emb_dim=args.embedding_dim).to(device)

    # Loss and optimizer
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.Adam(pointnet.parameters(), lr=args.learning_rate)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5)
    

    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        epoch_loss = train_one_epoch(pointnet, train_dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1} Training Loss: {epoch_loss:.4f}")

        # Validate the model
        val_loss = train_one_epoch(pointnet, val_dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

        # Save the best model checkpoint
        if val_loss < early_stopping.best_loss:
            torch.save(pointnet.state_dict(), os.path.join(results_dir, "best_pointnet_contrastive.pth"))
            print("Best model checkpoint saved.")

            # Save information about the best model
            best_model_info = {
                "epoch": epoch + 1,
                "validation_loss": val_loss
            }
            with open(os.path.join(results_dir, "best_model_info.json"), "w") as f:
                json.dump(best_model_info, f, indent=4)

    
    # Save the trained model
    torch.save(pointnet.state_dict(), os.path.join(results_dir, "pointnet_contrastive.pth"))
    print("Model saved as pointnet_contrastive.pth")


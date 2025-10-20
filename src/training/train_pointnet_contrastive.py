import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import argparse
import json

from src.dataset.pointcloud_dataset import PointCloudCLIPDataset, collate_pointcloud
from src.dataset.split_dataset import DatasetSplitter
from src.models.pointnet import PointNetEmbedding


# ------------------ Argument Parser ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train PointNet with Contrastive Learning (InfoNCE)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Dimension of the embedding")
    parser.add_argument("--csv_path", type=str, default="data/captions.tablechair.emb.csv", help="Path to the CSV file")
    parser.add_argument("--pcd_root", type=str, default="data/text2shape_2048_pointcloud", help="Root directory for point cloud data")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for InfoNCE loss")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    return parser.parse_args()


# ------------------ Contrastive InfoNCE Loss ------------------
def contrastive_loss(point_emb, clip_emb, temperature=0.07):
    # Normalize embeddings
    point_emb = F.normalize(point_emb, dim=1)
    clip_emb = F.normalize(clip_emb, dim=1)

    # Similarity matrix [B, B]
    logits = point_emb @ clip_emb.t() / temperature
    labels = torch.arange(point_emb.size(0), device=point_emb.device)

    # Symmetric InfoNCE loss
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


# ------------------ Early Stopping ------------------
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4):
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


# ------------------ Training Function ------------------
def train_one_epoch(model, dataloader, optimizer, device, temperature):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        points = batch["points"].to(device)
        clip_emb = batch["clip_emb"].to(device)

        optimizer.zero_grad()
        point_emb = model(points)

        loss = contrastive_loss(point_emb, clip_emb, temperature)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


# ------------------ Evaluation Function ------------------
def evaluate_one_epoch(model, dataloader, device, temperature):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            points = batch["points"].to(device)
            clip_emb = batch["clip_emb"].to(device)

            point_emb = model(points)
            loss = contrastive_loss(point_emb, clip_emb, temperature)
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ------------------ Main Training Script ------------------
if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = f"results/point_cloud_contrastive/batch_size_{args.batch_size}/lr_{args.learning_rate}/temperature_{args.temperature}/"
    os.makedirs(results_dir, exist_ok=True)

    # Save training parameters
    params = vars(args)
    params["device"] = str(device)
    with open(os.path.join(results_dir, "training_params.json"), "w") as f:
        json.dump(params, f, indent=4)
    print("Training parameters saved to training_params.json")

    # Split dataset
    splitter = DatasetSplitter(args.csv_path)
    train_df, val_df, test_df = splitter.split_by_category()

    # Dataloaders
    train_dataset = PointCloudCLIPDataset(train_df, args.pcd_root, use_color=True)
    val_dataset = PointCloudCLIPDataset(val_df, args.pcd_root, use_color=True)
    test_dataset = PointCloudCLIPDataset(test_df, args.pcd_root, use_color=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_pointcloud, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_pointcloud)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=collate_pointcloud)

    # Model + optimizer + scheduler
    model = PointNetEmbedding(input_dim=6, emb_dim=args.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)

    # ------------------ Training Loop ------------------
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.temperature)
        print(f"Training Loss: {train_loss:.4f}")

        val_loss = evaluate_one_epoch(model, val_loader, device, args.temperature)
        print(f"Validation Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < early_stopping.best_loss:
            torch.save(model.state_dict(), os.path.join(results_dir, "best_pointnet_contrastive.pth"))
            print("✅ Best model checkpoint saved.")
            with open(os.path.join(results_dir, "best_model_info.json"), "w") as f:
                json.dump({"epoch": epoch + 1, "validation_loss": val_loss}, f, indent=4)

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model checkpoint
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_pointnet_contrastive.pth")))
    print("Loaded best model checkpoint.")

    # Evaluate on test set
    test_loss = evaluate_one_epoch(model, test_loader, device, args.temperature)
    print(f"Test Loss: {test_loss:.4f}")

    # Save test loss to a JSON file
    with open(os.path.join(results_dir, "test_results.json"), "w") as f:
        json.dump({"test_loss": test_loss}, f, indent=4)
    print("Test results saved to test_results.json.")

    print("✅ Training complete! Model saved as best_pointnet_contrastive.pth")

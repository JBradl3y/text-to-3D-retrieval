import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset.pointcloud_dataset import PointCloudCLIPDataset, collate_pointcloud
from src.models.pointnet import PointNetEmbedding
from src.dataset.split_dataset import DatasetSplitter
import json
import argparse

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Test PointNet with Top-K Accuracy")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate used during training")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for InfoNCE loss")
    parser.add_argument("--k", type=int, default=5, help="Top-K value for evaluation")
    return parser.parse_args()

def fraction_correct_top_k(model, dataloader, device, k=5):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing Top-K"):
            points = batch["points"].to(device)
            clip_emb = batch["clip_emb"].to(device)

            # Get embeddings
            point_emb = model(points)

            # Compute similarity
            similarity = torch.mm(point_emb, clip_emb.t())  # [B, B]
            top_k_indices = torch.topk(similarity, k=k, dim=1).indices  # [B, K]

            # Check if ground truth is in top-K
            labels = torch.arange(points.size(0), device=device)
            for i in range(points.size(0)):
                if labels[i] in top_k_indices[i]:
                    correct += 1
                total += 1

    return correct / total

if __name__ == "__main__":
    args = parse_args()

    results_dir = f"results/point_cloud_contrastive/batch_size_{args.batch_size}/lr_{args.learning_rate}/temperature_{args.temperature}/"
    model_path = os.path.join(results_dir, "best_pointnet_contrastive.pth")
    csv_path = "data/captions.tablechair.emb.csv"
    pcd_root = "data/text2shape_2048_pointcloud"
    batch_size = args.batch_size
    k = args.k

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    splitter = DatasetSplitter(csv_path)
    _, _, test_df = splitter.split_by_category()
    test_dataset = PointCloudCLIPDataset(test_df, pcd_root, use_color=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=collate_pointcloud)

    # Load model
    model = PointNetEmbedding(input_dim=6, emb_dim=512).to(device)
    model.load_state_dict(torch.load(model_path))
    print("Loaded model from:", model_path)

    # Evaluate Top-K accuracy
    top_k_accuracy = fraction_correct_top_k(model, test_loader, device, k=k)
    print(f"Fraction of Correct Items in Top-{k}: {top_k_accuracy:.4f}")

    # Save results
    with open(os.path.join(results_dir, "top_k_results.json"), "w") as f:
        json.dump({"top_k_accuracy": top_k_accuracy, "k": k}, f, indent=4)
    print("Top-K results saved to top_k_results.json.")
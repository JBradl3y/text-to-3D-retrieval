import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset.pointcloud_dataset import PointCloudCLIPDataset, collate_pointcloud
from src.models.pointnet import PointNetEmbedding
from src.dataset.split_dataset import DatasetSplitter
import argparse
from torch.nn import functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss_type", type=str, default="InfoNCE_duplicated_positive", help="Type of loss function to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    loss_type = args.loss_type
    results_dir = f"results/point_cloud_contrastive/batch_size_{args.batch_size}/lr_{args.learning_rate}/temperature_{args.temperature}/"
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f"best_pointnet_contrastive_{loss_type}.pth")
    ids_path = os.path.join(results_dir, f"test_pointcloud_ids_{loss_type}.npy")
    embeddings_path = os.path.join(results_dir, f"test_pointcloud_embeddings_{loss_type}.npy")

    csv_path = "data/captions.tablechair.emb.csv"
    pcd_root = "data/text2shape_2048_pointcloud"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splitter = DatasetSplitter(csv_path)
    _, _, test_df = splitter.split_by_category()
    test_dataset = PointCloudCLIPDataset(test_df, pcd_root, use_color=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, collate_fn=collate_pointcloud)

    model = PointNetEmbedding(input_dim=6, emb_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ids_list = []
    embs_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Embedding Point Clouds"):
            points = batch["points"].to(device)
            model_ids = batch["model_id"]  # Use model_id instead of id
            emb = model(points)                 # [B, D]
            emb = F.normalize(emb, dim=1)       # normalize
            emb = emb.cpu().numpy()

            for i, model_id in enumerate(model_ids):
                ids_list.append(model_id)
                embs_list.append(emb[i])

    combined_data = {model_id: emb for model_id, emb in zip(ids_list, embs_list)}
    np.save(embeddings_path, combined_data)
    print(f"Saved combined model IDs and embeddings -> {embeddings_path}")

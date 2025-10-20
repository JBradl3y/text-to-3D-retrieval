# src/dataset/pointcloud_dataset.py
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import open3d as o3d
import numpy as np

class PointCloudCLIPDataset(Dataset):
    """
    Dataset for colored point clouds paired with CLIP embeddings.
    Designed for PointNet: all point clouds sampled to self.num_points.
    """

    def __init__(self, dataframe: pd.DataFrame, root_dir: str, num_points: int = 2048, use_color: bool = True):
        """
        Args:
            dataframe: Pandas DataFrame with columns ['modelId', 'category', 'clip_embedding']
            root_dir: root directory containing point clouds organized by category
            num_points: number of points to sample per cloud
            use_color: whether to return colors
        """
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.num_points = num_points
        self.use_color = use_color

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        model_id = row["modelId"]
        category = row["category"].lower()

        ply_path = os.path.join(self.root_dir, category, f"{model_id}.ply")
        if not os.path.exists(ply_path):
            print(f"[Warning] Missing PLY for modelId {model_id}")
            return None

        # --- Load point cloud ---
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if self.use_color else None
        # --- Merge points + colors if use_color ---
        if self.use_color and colors is not None:
            points_colors = np.concatenate([points, colors], axis=1)  # [num_points, 6]
        else:
            points_colors = points  # [num_points, 3]

        # --- Sample fixed number of points ---
        if points_colors.shape[0] >= self.num_points:
            idx_sample = np.random.choice(points_colors.shape[0], self.num_points, replace=False)
        else:
            # If we have fewer points than desired, take all points first
            idx_sample = np.arange(points_colors.shape[0])
            # Then randomly sample additional points with replacement to reach num_points
            extra = np.random.choice(points_colors.shape[0], self.num_points - points_colors.shape[0], replace=True)
            idx_sample = np.concatenate([idx_sample, extra])
            # idx_sample = np.random.choice(points.shape[0], self.num_points, replace=True)
        points_colors = points_colors[idx_sample]

        # --- Load CLIP embedding from CSV ---
        clip_emb = torch.tensor(
            np.fromstring(row["clip_embedding"][1:-1], sep=","),
            dtype=torch.float32
        )

        return {
            "points": torch.from_numpy(points_colors).float(),  # [num_points, 6] or [num_points, 3]
            "clip_emb": clip_emb,                                   # [512,]
            "model_id": model_id,
            "category": category,
        }

def collate_pointcloud(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Stack points: [B, N, C] → transpose → [B, C, N]
    points = torch.stack([b["points"] for b in batch], dim=0).transpose(1,2)
    clip_emb = torch.stack([b["clip_emb"] for b in batch])

    batch_dict = {
        "points": points,       # [B, C, N]
        "clip_emb": clip_emb,   # [B, 512]
        "model_id": [b["model_id"] for b in batch],
        "category": [b["category"] for b in batch],
    }
    return batch_dict

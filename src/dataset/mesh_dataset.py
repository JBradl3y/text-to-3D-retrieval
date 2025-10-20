# mesh_dataset.py
import torch
from torch.utils.data import Dataset
import trimesh
import numpy as np
import os

class MeshCLIPDataset(Dataset):
    def __init__(self, df, mesh_root, use_color=True):
        self.df = df
        self.mesh_root = mesh_root
        self.use_color = use_color

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mesh_path = os.path.join(self.mesh_root, row['category'].lower(), f"{row['modelId']}.ply")
        mesh = trimesh.load(mesh_path, process=False)

        # vertices, faces
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)

        # colors
        colors = None
        if self.use_color:
            if hasattr(mesh.visual, 'vertex_colors'):
                colors = mesh.visual.vertex_colors[:, :3]
                colors = torch.tensor(colors, dtype=torch.float32) / 255.0
            else:
                colors = torch.zeros_like(vertices)

        # clip embedding
        clip_emb = torch.tensor(np.fromstring(row['clip_embedding'][1:-1], sep=','), dtype=torch.float32)

        return {
            'vertices': vertices,
            'faces': faces,
            'colors': colors,
            'clip_emb': clip_emb,
            'model_id': row['modelId'],
            'category': row['category']
        }

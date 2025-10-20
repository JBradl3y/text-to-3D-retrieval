# mesh_collate.py
import torch

def collate_mesh(batch):
    """
    batch: list of dict, mỗi dict có keys: vertices, faces, colors, clip_emb, model_id, category
    Trả về dict với lists hoặc batch tensors offset (cho PyG / MeshCNN)
    """

    # # Remove None samples (e.g., missing mesh files)
    # batch = [b for b in batch if b is not None]

    # if len(batch) == 0:
    #     return None  # return None if batch is empty

    # Prepare batched dictionary
    batch_dict = {
        'vertices': [b['vertices'] for b in batch],   # list of vertex tensors
        'faces': [b['faces'] for b in batch],         # list of face tensors
        'colors': [b['colors'] for b in batch],       # list of color tensors
        'clip_emb': torch.stack([b['clip_emb'] for b in batch]),  # [batch_size, embedding_dim]
        'model_id': [b['model_id'] for b in batch],   # list of model IDs
        'category': [b['category'] for b in batch]    # list of categories
    }

    return batch_dict

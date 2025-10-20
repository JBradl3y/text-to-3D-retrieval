# test_dataloader.py
from split_dataset import DatasetSplitter
from mesh_dataset import MeshCLIPDataset
from mesh_collate import collate_mesh  # import collate function
from torch.utils.data import DataLoader

CSV_PATH = 'data/temp.csv'
MESH_ROOT = 'data/text2shape_mesh'

if __name__ == '__main__':
    # 1️⃣ Split dataset
    splitter = DatasetSplitter(CSV_PATH)
    train_df, val_df, test_df = splitter.split_by_category()
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2️⃣ Create dataset & dataloader
    train_dataset = MeshCLIPDataset(train_df, MESH_ROOT, use_color=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,        
        shuffle=True,
        collate_fn=collate_mesh 
    )

    # 3️⃣ Get 1 batch for testing
    batch = next(iter(train_loader))

    print("Keys:", batch.keys())
    print("Number of meshes in batch:", len(batch['vertices']))
    print()
    print("Vertices in first mesh:", batch['vertices'][0].shape)
    print("Faces in first mesh:", batch['faces'][0].shape)
    print("Colors in first mesh:", batch['colors'][0].shape if batch['colors'][0] is not None else None)
    print("CLIP embedding shape:", batch['clip_emb'].shape)  # [batch_size, 512]
    print("Model IDs:", batch['model_id'])
    print("Categories:", batch['category'])
    print()
    print("Vertices in first mesh:", batch['vertices'][1].shape)
    print("Faces in first mesh:", batch['faces'][1].shape)
    print("Colors in first mesh:", batch['colors'][1].shape if batch['colors'][1] is not None else None)
    print("CLIP embedding shape:", batch['clip_emb'].shape)  # [batch_size, 512]
    print("Model IDs:", batch['model_id'])
    print("Categories:", batch['category'])

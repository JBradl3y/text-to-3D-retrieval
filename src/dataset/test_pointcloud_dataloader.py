# src/dataset/test_pointcloud_dataloader.py
from split_dataset import DatasetSplitter
from pointcloud_dataset import PointCloudCLIPDataset, collate_pointcloud
from torch.utils.data import DataLoader

CSV_PATH = "data/temp.csv"
PCD_ROOT = "data/text2shape_pointcloud"

if __name__ == "__main__":
    # 1️⃣ Split dataset
    splitter = DatasetSplitter(CSV_PATH)
    train_df, val_df, test_df = splitter.split_by_category()
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2️⃣ Create dataset and dataloader
    train_dataset = PointCloudCLIPDataset(train_df, PCD_ROOT, use_color=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_pointcloud,
    )

    # 3️⃣ Fetch one batch
    batch = next(iter(train_loader))
    print("Keys:", batch.keys())
    print("Number of point clouds in batch:", len(batch["points"]))
    print("Points shape of first:", batch["points"][0].shape)
    # print("Colors shape of first:", batch["colors"][0].shape)
    print("CLIP embedding shape:", batch["clip_emb"].shape)
    print("Model IDs:", batch["model_id"])
    print("Categories:", batch["category"])

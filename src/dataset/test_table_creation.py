import pandas as pd
from src.dataset.pointcloud_dataset import PointCloudCLIPDataset, collate_pointcloud
from src.dataset.split_dataset import DatasetSplitter
from torch.utils.data import DataLoader

# -----------------------
# LOAD TEST DATASET
# -----------------------
csv_path = "data/captions.tablechair.emb.csv"
pcd_root = "data/text2shape_2048_pointcloud"
batch_size = 32

# Use your existing splitter
splitter = DatasetSplitter(csv_path)
_, _, test_df = splitter.split_by_category()

# Create test dataset and loader
test_dataset = PointCloudCLIPDataset(test_df, pcd_root, use_color=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True, collate_fn=collate_pointcloud)

# -----------------------
# GET ALL IDS IN TEST LOADER
# -----------------------
test_ids = []
for batch in test_loader:
    # Assuming your dataset returns 'id' for each sample
    if "id" in batch:
        test_ids.extend(batch["id"])
    else:
        # If not, use indices (row indices in test_df)
        test_ids.extend(batch["index"].tolist())

print(f"Number of test samples: {len(test_ids)}")

# -----------------------
# READ FULL CSV AND FILTER
# -----------------------
full_df = pd.read_csv("data/captions.tablechair.csv")

# Filter only rows with IDs in test_ids
# Assuming the CSV has a column 'id' matching the dataset ids
filtered_df = full_df[full_df["id"].isin(test_ids)]

# -----------------------
# SAVE TO NEW CSV
# -----------------------
filtered_df.to_csv("data/captions.tablechair_test.csv", index=False)
print(f"Saved test CSV with {len(filtered_df)} samples to data/captions.tablechair_test.csv")

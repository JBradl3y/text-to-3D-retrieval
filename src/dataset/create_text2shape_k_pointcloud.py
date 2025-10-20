import os
import pandas as pd
import nrrd
import numpy as np
import open3d as o3d
from tqdm import tqdm

# --- Paths ---
captions_file = "data/captions.tablechair.csv"
voxel_source_dir = "data/nrrd_256_filter_div_32/nrrd_256_filter_div_32"
output_dir = "data/text2shape_2048_pointcloud"

# --- Step 1: Load captions ---
df = pd.read_csv(captions_file)

# Convert category to lowercase
df['category'] = df['category'].str.lower()

# Keep only unique modelId
df_unique = df.drop_duplicates(subset=['modelId'])

# --- Limit to first 10 point clouds ---
# df_unique = df_unique.head(10)

# --- Step 2: Create output folders ---
for category in df_unique['category'].unique():
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)

# --- Step 3: Convert voxel to point cloud and save ---
for _, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Processing point clouds"):
    model_id = row['modelId']
    category = row['category']

    nrrd_path = os.path.join(voxel_source_dir, model_id, f"{model_id}.nrrd")
    ply_path = os.path.join(output_dir, category, f"{model_id}.ply")

    if not os.path.exists(nrrd_path):
        print(f"[Warning] Voxel file not found for modelId {model_id}")
        import sys; sys.exit(1)

    # --- Load NRRD data ---
    data, header = nrrd.read(nrrd_path)

    # Separate RGBA channels
    red = data[0]
    green = data[1]
    blue = data[2]
    alpha = data[3]

    # Mask solid voxels
    solid_mask = alpha > 0

    # Get coordinates of solid voxels
    z, y, x = np.where(solid_mask)
    points = np.stack((x, y, z), axis=-1).astype(float)

    # Get RGB colors of solid voxels
    colors = np.stack((
        red[solid_mask],
        green[solid_mask],
        blue[solid_mask]
    ), axis=-1) / 255.0  # normalize

    # --- Create Open3D point cloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- Apply Farthest Point Sampling if points exceed 2048 ---
    if len(points) > 2048:
        pcd = pcd.farthest_point_down_sample(2048)

    # --- Save as PLY ---
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
    # print(f"[Saved] {ply_path}")

print("✅ All done! Point clouds organized by category in:", output_dir)

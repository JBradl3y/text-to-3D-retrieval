import os
import pandas as pd
import nrrd
import numpy as np
from skimage import measure
import trimesh
from tqdm import tqdm

# --- Paths ---
captions_file = "data/captions.tablechair.csv" 
voxel_source_dir = "data/nrrd_256_filter_div_32/nrrd_256_filter_div_32"
output_dir = "data/text2shape_mesh"

# --- Step 1: Load captions ---
df = pd.read_csv(captions_file)

# Convert category to lowercase
df['category'] = df['category'].str.lower()

# Keep only unique modelId
df_unique = df.drop_duplicates(subset=['modelId'])

# --- Step 2: Create output folders ---
for category in df_unique['category'].unique():
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)

# --- Step 3: Convert voxel to mesh and save ---
# count = 0
for _, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="Processing models"):
    # count += 1
    # if count > 29:
    #     break
    model_id = row['modelId']
    category = row['category']

    nrrd_path = os.path.join(voxel_source_dir, model_id, f"{model_id}.nrrd")
    ply_path = os.path.join(output_dir, category, f"{model_id}.ply")

    if not os.path.exists(nrrd_path):
        print(f"[Warning] Voxel file not found for modelId {model_id}")
        continue

    # --- Load NRRD ---
    data, header = nrrd.read(nrrd_path)

    # Separate RGBA channels
    red = data[0, :, :, :]
    green = data[1, :, :, :]
    blue = data[2, :, :, :]
    alpha = data[3, :, :, :]

    # Identify solid voxels
    solid_voxels = alpha > 0

    # Mask RGB by alpha
    red = red * solid_voxels
    green = green * solid_voxels
    blue = blue * solid_voxels

    # --- Marching cubes ---
    volume = solid_voxels.astype(float)
    vertices, faces, normals, _ = measure.marching_cubes(volume, level=0.5)

    # Interpolate vertex colors
    verts_idx = np.round(vertices).astype(int)
    verts_idx = np.clip(verts_idx, 0, np.array(volume.shape) - 1)

    vertex_colors = np.zeros((vertices.shape[0], 3))
    vertex_colors[:,0] = red[verts_idx[:,0], verts_idx[:,1], verts_idx[:,2]]
    vertex_colors[:,1] = green[verts_idx[:,0], verts_idx[:,1], verts_idx[:,2]]
    vertex_colors[:,2] = blue[verts_idx[:,0], verts_idx[:,1], verts_idx[:,2]]

    vertex_colors /= 255.0  # normalize to 0-1

    # --- Save mesh ---
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
    mesh.export(ply_path)
    # print(f"[Saved] {ply_path}")

print("All done! Meshes organized by category in:", output_dir)

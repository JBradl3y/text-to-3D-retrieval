"""
Convert ShapeNetCore to colored 32x32x32 voxel grid and save as NPY file.
Then visualize it with matplotlib.
"""
import sys
import os
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Configuration
MODEL_DIR = "data/shapenetcore_c5/03046257/1a157c6c3b71bbd6b4792411407bb04c/models"
OUTPUT_NPY = "shapenet_colored_32x32x32.npy"
OUTPUT_PNG = "shapenet_colored_visualization.png"
TARGET_SIZE = 32


def parse_mtl(mtl_path):
    """Parse MTL file and return material name -> RGB color dict."""
    materials = {}
    current_material = None
    
    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('newmtl '):
                current_material = line.split()[1]
                materials[current_material] = [128, 128, 128]  # Default grey
            elif line.startswith('Kd ') and current_material:
                parts = line.split()
                r, g, b = float(parts[1]), float(parts[2]), float(parts[3])
                materials[current_material] = [int(r * 255), int(g * 255), int(b * 255)]
    
    return materials


def parse_obj_with_materials(obj_path, materials):
    """Parse OBJ file and return vertices and faces with their material colors."""
    vertices = []
    faces = []
    current_material = None
    current_color = [128, 128, 128]  # Default grey
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                # Vertex
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('usemtl '):
                # Material assignment
                current_material = line.split()[1]
                current_color = materials.get(current_material, [128, 128, 128])
            elif line.startswith('f '):
                # Face with current material color
                parts = line.split()[1:]
                face_verts = []
                for p in parts:
                    v_idx = int(p.split('/')[0]) - 1
                    face_verts.append(v_idx)
                faces.append({'vertices': face_verts, 'color': current_color})
    
    return np.array(vertices), faces


def voxelize_colored_mesh(vertices, faces, grid_size=32):
    """
    Voxelize mesh with per-voxel colors from face materials.
    Returns: voxel_grid of shape (grid_size, grid_size, grid_size, 4) with RGBA
    """
    # Normalize vertices to [0, grid_size-1]
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    v_range = v_max - v_min
    v_range[v_range == 0] = 1
    
    vertices_norm = ((vertices - v_min) / v_range) * (grid_size - 1)
    
    # Create voxel grid: (grid_size, grid_size, grid_size, 4) for RGBA
    voxel_grid = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.uint8)
    color_accum = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
    count_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.int32)
    
    # Rasterize each face
    for face in faces:
        face_color = face['color']
        face_verts = vertices_norm[face['vertices']]
        
        # Get bounding box of face
        min_coord = np.floor(face_verts.min(axis=0)).astype(int)
        max_coord = np.ceil(face_verts.max(axis=0)).astype(int)
        
        # Clip to grid bounds
        min_coord = np.clip(min_coord, 0, grid_size - 1)
        max_coord = np.clip(max_coord, 0, grid_size - 1)
        
        # Fill voxels in face bounding box
        for x in range(min_coord[0], max_coord[0] + 1):
            for y in range(min_coord[1], max_coord[1] + 1):
                for z in range(min_coord[2], max_coord[2] + 1):
                    point = np.array([x, y, z])
                    distances = np.linalg.norm(face_verts - point, axis=1)
                    if distances.min() < 2.0:  # Within distance threshold
                        color_accum[x, y, z] += face_color
                        count_grid[x, y, z] += 1
    
    # Average colors and set alpha
    mask = count_grid > 0
    voxel_grid[mask, 0:3] = (color_accum[mask] / count_grid[mask, np.newaxis]).astype(np.uint8)
    voxel_grid[mask, 3] = 255  # Alpha channel
    
    return voxel_grid


def visualize_colored_voxels(voxel_grid, output_path):
    """Visualize colored voxel grid using matplotlib."""
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    
    # Extract occupancy and colors
    occupancy = voxel_grid[:, :, :, 3] > 0
    colors = voxel_grid[:, :, :, 0:3] / 255.0  # Normalize to 0-1
    
    print(f"Filled voxels: {np.sum(occupancy)}")
    print(f"Color range: R[{voxel_grid[occupancy, 0].min()}-{voxel_grid[occupancy, 0].max()}]")
    print(f"             G[{voxel_grid[occupancy, 1].min()}-{voxel_grid[occupancy, 1].max()}]")
    print(f"             B[{voxel_grid[occupancy, 2].min()}-{voxel_grid[occupancy, 2].max()}]")
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    
    # Plot colored voxels
    ax.voxels(occupancy, facecolors=colors, edgecolor='none')
    
    ax.set_title("ShapeNetCore Colored Voxel Visualization", fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    plt.close()


# Main execution
print("="*60)
print("Converting ShapeNetCore to Colored Voxel Grid")
print("="*60)

print(f"\n[1/4] Parsing materials from MTL...")
mtl_path = os.path.join(MODEL_DIR, "model_normalized.mtl")
materials = parse_mtl(mtl_path)
print(f"Found {len(materials)} materials:")
for name, color in materials.items():
    print(f"  {name}: RGB{tuple(color)}")

print(f"\n[2/4] Parsing OBJ with material assignments...")
obj_path = os.path.join(MODEL_DIR, "model_normalized.obj")
vertices, faces = parse_obj_with_materials(obj_path, materials)
print(f"Loaded {len(vertices)} vertices, {len(faces)} faces")

print(f"\n[3/4] Voxelizing mesh with colors to {TARGET_SIZE}³...")
voxel_grid = voxelize_colored_mesh(vertices, faces, TARGET_SIZE)

print(f"\n[4/4] Saving results...")
# Save as NPY file (easy to load in Python)
np.save(OUTPUT_NPY, voxel_grid)
print(f"Saved voxel grid: {OUTPUT_NPY}")
print(f"  Shape: {voxel_grid.shape} (32x32x32x4 RGBA)")
print(f"  Size: {voxel_grid.nbytes / 1024:.1f} KB")

# Visualize
visualize_colored_voxels(voxel_grid, OUTPUT_PNG)

print("\n" + "="*60)
print("Done!")
print("="*60)
print(f"\nTo load the voxel grid in Python:")
print(f"  voxels = np.load('{OUTPUT_NPY}')")
print(f"  # Shape: (32, 32, 32, 4) where last dim is RGBA")

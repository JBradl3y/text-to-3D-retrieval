#!/usr/bin/env python3
"""
Convert a ShapeNet model to a 32x32x32 colored voxel grid and save to a single file.
"""

import numpy as np
import trimesh
from pathlib import Path
import pickle


def parse_mtl(mtl_path):
    """Parse MTL file to extract material colors."""
    materials = {}
    current_material = None
    
    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('newmtl'):
                current_material = line.split()[1]
                materials[current_material] = {}
            elif line.startswith('Kd') and current_material:
                rgb = [float(x) for x in line.split()[1:4]]
                materials[current_material]['color'] = rgb
    
    return materials


def load_obj_with_colors(obj_path, mtl_path):
    """Load OBJ file and assign colors from MTL."""
    mesh = trimesh.load(obj_path, force='mesh', process=False)
    materials = parse_mtl(mtl_path)
    
    # Get default color from materials
    default_color = [0.7, 0.7, 0.7]
    for mat_name, mat_props in materials.items():
        if 'color' in mat_props:
            default_color = mat_props['color']
            break
    
    return mesh, default_color


def voxelize_colored_mesh(mesh, default_color, grid_size=32):
    """Convert mesh to colored voxel grid."""
    # Center and scale mesh to fit in unit cube
    mesh.apply_translation(-mesh.centroid)
    scale = 0.9 / mesh.extents.max()
    mesh.apply_scale(scale)
    
    # Voxelize
    voxel_grid = mesh.voxelized(pitch=1.0 / grid_size)
    voxel_matrix = voxel_grid.matrix
    
    # Ensure it's exactly grid_size x grid_size x grid_size
    if voxel_matrix.shape != (grid_size, grid_size, grid_size):
        # Resize to exact dimensions
        new_matrix = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(voxel_matrix.shape, new_matrix.shape))
        new_matrix[:min_shape[0], :min_shape[1], :min_shape[2]] = voxel_matrix[:min_shape[0], :min_shape[1], :min_shape[2]]
        voxel_matrix = new_matrix
    
    # Create color grid - assign default color to all occupied voxels
    color_grid = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
    color_grid[voxel_matrix] = default_color
    
    return voxel_matrix, color_grid


def save_colored_voxels(voxel_matrix, color_grid, output_path):
    """Save colored voxel grid to a single file."""
    data = {
        'voxels': voxel_matrix.astype(np.bool_),
        'colors': color_grid.astype(np.float32),
        'grid_size': voxel_matrix.shape
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved to: {output_path}")
    print(f"Grid: {voxel_matrix.shape}")
    print(f"Occupied: {voxel_matrix.sum()} voxels")


def main():
    # Input paths
    model_dir = Path("data/shapenetcore_c5/03046257/1a157c6c3b71bbd6b4792411407bb04c/models")
    obj_path = model_dir / "model_normalized.obj"
    mtl_path = model_dir / "model_normalized.mtl"
    output_path = "colored_voxels_32x32x32.pkl"
    
    print(f"Loading: {obj_path}")
    mesh, default_color = load_obj_with_colors(obj_path, mtl_path)
    print(f"Color: {default_color}")
    
    print("Voxelizing to 32x32x32...")
    voxel_matrix, color_grid = voxelize_colored_mesh(mesh, default_color, grid_size=32)
    
    save_colored_voxels(voxel_matrix, color_grid, output_path)
    print("Done!")


if __name__ == "__main__":
    main()

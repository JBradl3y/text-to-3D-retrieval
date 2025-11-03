"""Render Text2Shape meshes to multi-view PNGs using Open3D."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm

from .config import ZeroShotPaths

LIGHT_INTENSITY = 75000


def parse_args():
    parser = argparse.ArgumentParser(description="Render Text2Shape meshes with Open3D.")
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--mesh-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def evenly_spaced_angles(num):
    return [(360.0 * i) / num for i in range(num)]


def camera_positions(azimuths, elevation_deg, radius=2.5):
    elevation_rad = math.radians(elevation_deg)
    positions = []
    for az_deg in azimuths:
        az_rad = math.radians(az_deg)
        cam_x = radius * math.cos(elevation_rad) * math.cos(az_rad)
        cam_y = radius * math.cos(elevation_rad) * math.sin(az_rad)
        cam_z = radius * math.sin(elevation_rad)
        positions.append(np.array([cam_x, cam_y, cam_z], dtype=np.float32))
    return positions


def setup_renderer(width, height):
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 0.0])
    renderer.scene.scene.set_sun_light(np.array([0.5, -0.5, -1.0]), np.array([1.0, 1.0, 1.0]), LIGHT_INTENSITY, True)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.show_axes(False)
    renderer.scene.show_skybox(False)
    return renderer


def load_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    mesh = mesh.translate(-mesh.get_center())
    bbox = mesh.get_axis_aligned_bounding_box()
    scale = 1.0 / max(bbox.extent) if max(bbox.extent) > 0 else 1.0
    return mesh.scale(scale, center=np.zeros(3))


def render_views(renderer, mesh, positions, width, height):
    renderer.scene.clear_geometry()
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, material)

    camera = renderer.scene.camera
    camera.set_projection(60.0, width / float(height), 0.01, 100.0)

    images = []
    for eye in positions:
        camera.look_at(np.zeros(3), eye, np.array([0.0, 0.0, 1.0]))
        images.append(np.asarray(renderer.render_to_image()))
    return images


def main():
    args = parse_args()
    paths = ZeroShotPaths.from_env()
    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else paths.mesh_dir
    output_dir = Path(args.output_dir) if args.output_dir else paths.render_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    renderer = setup_renderer(args.width, args.height)
    positions = camera_positions(evenly_spaced_angles(args.views), args.elevation)

    df = pd.read_csv(args.index_csv)
    df["category"] = df["category"].str.lower()

    metadata = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Rendering meshes"):
        model_id = str(row["modelId"])
        category = row["category"]
        mesh_path = Path(mesh_dir) / category / f"{model_id}.ply"
        render_path = Path(output_dir) / category / model_id
        render_path.mkdir(parents=True, exist_ok=True)

        expected = [render_path / f"view_{i:02d}.png" for i in range(len(positions))]
        if not args.overwrite and all(p.exists() for p in expected):
            metadata.append({"modelId": model_id, "category": category, "status": "already_rendered", "render_dir": str(render_path), "num_views": len(expected)})
            continue

        if not mesh_path.exists():
            metadata.append({"modelId": model_id, "category": category, "status": "missing_mesh"})
            continue

        mesh = load_mesh(mesh_path)
        images = render_views(renderer, mesh, positions, args.width, args.height)
        for idx, img in enumerate(images):
            o3d.io.write_image(str(render_path / f"view_{idx:02d}.png"), o3d.geometry.Image(img))

        metadata.append({"modelId": model_id, "category": category, "status": "rendered", "render_dir": str(render_path), "num_views": len(images)})

    meta_file = Path(output_dir) / "render_metadata.jsonl"
    with meta_file.open("w") as fh:
        for record in metadata:
            fh.write(json.dumps(record) + "\n")

    print(f"Wrote metadata to {meta_file}")


if __name__ == "__main__":
    main()

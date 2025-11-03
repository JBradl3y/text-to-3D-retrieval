import os
import csv
import torch
import numpy as np
import nrrd
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')  # for headless environments (no GUI)
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# -------------------------
# CONFIGURATION
# -------------------------
CSV_PATH = "data/captions.tablechair_test.csv"
NRRD_DIR = "./data/nrrd_256_filter_div_32/nrrd_256_filter_div_32"
OUTPUT_DIR = "./data/2D_render_clip_embeddings/tablechair_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Eight rendering views (elevation, azimuth)
VIEWS = {
    "0": (30, -90),
    "45": (30, -45),
    "90": (30, 0),
    "135": (30, 45),
    "180": (30, 90),
    "225": (30, 135),
    "270": (30, 180),
    "315": (30, -135),
}

# -------------------------
# LOAD CLIP MODEL
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# -------------------------
# VOXEL RENDER FUNCTION
# -------------------------
def render_voxel_view(solid_voxels, colors, elev, azim, size=224):
    """Render a single 3D voxel grid view into an RGB image."""
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(solid_voxels, facecolors=colors, edgecolor='none')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    plt.tight_layout()
    
    # Convert the rendered matplotlib figure to a PIL image
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = Image.fromarray(buf[..., :3]).convert("RGB")
    plt.close(fig)
    return img

# -------------------------
# LOAD MODEL IDS
# -------------------------
model_ids = []
with open(CSV_PATH, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if "modelId" in row:
            model_ids.append(row["modelId"])
model_ids = sorted(set(model_ids))
print(f"Found {len(model_ids)} models in test split.")

# -------------------------
# MAIN LOOP: RENDER + EMBED
# -------------------------
count = 0
for model_id in tqdm(model_ids, desc="Embedding 3D models"):
    # if count > 999:
    #     break
    count += 1
    if count < 1001:
        continue
    nrrd_path = os.path.join(NRRD_DIR, model_id, f"{model_id}.nrrd")
    if not os.path.exists(nrrd_path):
        tqdm.write(f"⚠️ Missing file: {nrrd_path}")
        continue

    try:
        data, _ = nrrd.read(nrrd_path)
    except Exception as e:
        tqdm.write(f"❌ Error reading {model_id}: {e}")
        continue

    # Split RGBA channels
    r, g, b, a = data
    solid_voxels = a > 0
    colors = np.zeros(solid_voxels.shape + (3,))
    colors[..., 0] = r
    colors[..., 1] = g
    colors[..., 2] = b
    colors_normalized = colors / 255.0

    # Transpose axes to align the orientation correctly
    solid_voxels = np.transpose(solid_voxels, (2, 1, 0))       # X,Z,Y
    colors_normalized = np.transpose(colors_normalized, (2, 1, 0, 3))

    # Render and compute CLIP embeddings
    clip_embeds = []
    for _, (elev, azim) in VIEWS.items():
        img = render_voxel_view(solid_voxels, colors_normalized, elev, azim)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            embed = model.get_image_features(**inputs)
        clip_embeds.append(embed.cpu().numpy())

    clip_embeds = np.concatenate(clip_embeds, axis=0)  # shape: (8, D)
    
    # Save per-view embeddings
    os.makedirs(os.path.join(OUTPUT_DIR, model_id), exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{model_id}/clip_embed.npy")
    np.save(save_path, clip_embeds)

    # # Save mean embedding
    # avg_embed = clip_embeds.mean(axis=0)
    # np.save(os.path.join(OUTPUT_DIR, f"{model_id}_clip_avg.npy"), avg_embed)

print("✅ Done! All CLIP embeddings have been saved.")

# Instructions for Using the Dataset and Libraries

## Dataset Setup

Currently, we can use the Text2Shape dataset and ShapeNetCore. Note that you need to ask for author permission to use ShapeNetCore, as the public version on Kaggle is corrupted.

1. Create a folder named `data`.
2. Download the Text2Shape dataset.
3. Ensure the `data` folder contains the following:
   - `captions.tablechair.csv`
   - `nrrd_256_filter_div_32/nrrd_256_filter_div_32/` (all voxel files from the dataset).

## Encoding Text with CLIP Model

Run the following command to encode text (captions of shapes) using the CLIP model:
```bash
python src/dataset/caption_encoding.py
```

## Creating Mesh Data

Run the following command to create mesh data:
```bash
python src/dataset/create_text2shape_mesh.py
```

## Creating Point Cloud Data

Run the following command to create point cloud data:
```bash
python src/dataset/create_text2shape_pointcloud.py
```

## Testing Dataset Loading

Run the following command to test loading the dataset from Text2Shape:
```bash
python src/dataset/test_dataloader.py
```

## Required Libraries

Install the following libraries to set up the environment:

### Jupyter Notebook
```bash
pip install notebook
pip install notebook ipykernel
```

### NRRD File Reading
```bash
pip install pynrrd
```

### Numerical Computations
```bash
pip install numpy
```

### Marching Cubes / 3D Image Processing
```bash
pip install scikit-image
```

### Open3D for Mesh Creation, Simplification, and Visualization
```bash
pip install open3d
```

### Optional: Matplotlib for Plotting or Quick Visualization in Notebooks
```bash
pip install matplotlib
```

### Additional Libraries
```bash
pip install trimesh
pip3 install torch torchvision
pip install transformers
```
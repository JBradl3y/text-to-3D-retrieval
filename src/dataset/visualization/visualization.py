import nrrd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read the NRRD file
file_path = "data/nrrd_256_filter_div_32/nrrd_256_filter_div_32/1a00aa6b75362cc5b324368d54a7416f/1a00aa6b75362cc5b324368d54a7416f.nrrd"
data, header = nrrd.read(file_path)

print(f"Data shape: {data.shape}")

# Step 2: Separate the 4D data.
# The `pynrrd` library reads the data in the order of the axes defined in the NRRD header.
# A common convention is (channels, z, y, x).
# `data[3]` would be the alpha channel, but based on your header:
# `space directions`: array([[ nan, nan, nan], [0., 0.0024, 0.], ...])
# This suggests that the axes are ordered differently, often (X, Y, Z, Channels).
# In that case, the alpha channel is the *last* dimension. Let's assume the first dimension is RGBA for now.

# Extract the volumetric data (e.g., the red channel) and the alpha channel
# The first dimension (index 0) is a single slice for a given channel.
# Let's assume the first dimension is the channel (RGBA). The alpha channel is at index 3.
# The rest of the dimensions (1, 2, 3) are the 3D space (Z, Y, X).
alpha_channel = data[3, :, :, :]

# Step 3: Create a boolean array to identify solid voxels
# We define a "solid" voxel as one with a non-zero alpha value.
solid_voxels = alpha_channel > 0

# Step 4: Visualize the 3D shape with matplotlib
# `matplotlib`'s `voxels()` function is perfect for this task.
# It takes a boolean array to define the shape and optional colors.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(solid_voxels)

# Set the title and show the plot
ax.set_title("3D Shape Visualization")
plt.show()

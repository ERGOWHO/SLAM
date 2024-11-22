import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load camera positions from images.txt
camera_positions = []

with open('/home/yuhu/gaussian-splatting/data/IDU117_droid2colmap/sparse/0/images.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith('#') or line.strip() == "":
            continue
        parts = line.split()
        tx, ty, tz = map(float, parts[5:8])  # Extract TX, TY, TZ
        camera_positions.append([tx, ty, tz])

camera_positions = np.array(camera_positions)

# Visualize camera positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='blue', label='Camera Positions')

ax.set_title('Camera Positions Visualization')
ax.set_xlabel('X (TX)')
ax.set_ylabel('Y (TY)')
ax.set_zlabel('Z (TZ)')
ax.legend()
plt.show()

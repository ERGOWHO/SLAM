import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the matrix data from the file
file_path = '/home/yuhu/Downloads/full_trajectory_today.txt'
data = np.loadtxt(file_path)

# Extract translation components
translations = data[:, [3, 7, 11]]

def save_ply(filename, vertices, edges):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")

# Create a function to plot the camera frustum and collect vertices and edges for the PLY file
def plot_camera(ax, position, rotation, size=0.02, height=0.06):  # Added height parameter for taller frustums
    # Define the pyramid (frustum) corners relative to the camera center
    pyramid = np.array([
        [0, 0, 0],     # Apex of the pyramid (camera position)
        [size, size, height],  # Base corners of the pyramid
        [-size, size, height],
        [-size, -size, height],
        [size, -size, height]
    ])
    
    # Rotate and translate the pyramid according to the camera pose
    pyramid = pyramid @ rotation.T + position

    faces = [
        [pyramid[0], pyramid[1], pyramid[2]],  # Front face
        [pyramid[0], pyramid[2], pyramid[3]],  # Left face
        [pyramid[0], pyramid[3], pyramid[4]],  # Back face
        [pyramid[0], pyramid[4], pyramid[1]],  # Right face
        [pyramid[1], pyramid[2], pyramid[3], pyramid[4]]  # Base face
    ]

    # Plot the faces of the pyramid
    ax.add_collection3d(Poly3DCollection(faces, color='cyan', linewidths=0.5, edgecolors='r', alpha=.25))


    # Define the edges of the pyramid for the PLY file
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # Edges from apex to base corners
        (1, 2), (2, 3), (3, 4), (4, 1)  # Edges of the base
    ]

    # Plot the pyramid outline
    for start, end in edges:
        ax.plot([pyramid[start, 0], pyramid[end, 0]], 
                [pyramid[start, 1], pyramid[end, 1]], 
                [pyramid[start, 2], pyramid[end, 2]], 'r-')
    
    return pyramid, edges

# Prepare the plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

vertices = []
edges = []

# Plot every 5th camera position and orientation
for i in range(0, translations.shape[0], 20):
    position = translations[i]
    # Extract the rotation matrix part
    rotation = data[i].reshape(4, 4)[:3, :3]
    pyramid, pyramid_edges = plot_camera(ax, position, rotation)
    offset = len(vertices)
    vertices.extend(pyramid)
    edges.extend([(start + offset, end + offset) for start, end in pyramid_edges])

# Set plot limits
ax.set_xlim(np.min(translations[:, 0]), np.max(translations[:, 0]))
ax.set_ylim(np.min(translations[:, 1]), np.max(translations[:, 1]))
ax.set_zlim(np.min(translations[:, 2]), np.max(translations[:, 2]))

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.title('Camera Trajectory')
plt.show()

# Save to PLY file
ply_filename = '/home/yuhu/Downloads/camera_trajectory.ply'
save_ply(ply_filename, vertices, edges)
print(f"PLY file saved as {ply_filename}")

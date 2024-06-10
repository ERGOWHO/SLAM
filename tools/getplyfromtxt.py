import numpy as np

def read_matrices_from_txt(file_path):
    """
    Reads a text file containing 4x4 matrices and returns them as a numpy array.
    """
    # Read the file into a DataFrame
    data = np.loadtxt(file_path)
    
    # Convert the DataFrame to a numpy array of shape (-1, 4, 4)
    matrices = data.reshape(-1, 4, 4)
    
    return matrices

def save_matrices_to_ply(matrices, output_file):
    """
    Saves the translation components of 4x4 matrices to a PLY file.
    """
    header = '''ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
end_header
'''
    vertex_count = matrices.shape[0]
    vertices = matrices[:, :3, 3]

    with open(output_file, 'w') as f:
        f.write(header.format(vertex_count=vertex_count))
        for vertex in vertices:
            f.write(f'{vertex[0]} {vertex[1]} {vertex[2]}\n')

# File paths
input_file_path = 'ATE_compare/full_trajectory.txt'  # Replace with your input file path
output_file_path = 'ATE_compare/full_trajectory.ply'  # Replace with your desired output file path

# Read the matrices from the txt file
matrices = read_matrices_from_txt(input_file_path)

# Save the matrices to a PLY file
save_matrices_to_ply(matrices, output_file_path)

print(f"PLY file has been saved to {output_file_path}")

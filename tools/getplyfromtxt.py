import numpy as np
import struct

def read_matrices_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    matrices = []
    current_matrix = []
    for line in lines:
        if line.strip():
            current_matrix.append(list(map(float, line.split())))
            if len(current_matrix) == 4:
                matrices.append(np.array(current_matrix))
                current_matrix = []
    
    return matrices

def write_ply(file_path, points):
    ply_header = f'''ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
'''
    with open(file_path, 'wb') as f:
        f.write(ply_header.encode('utf-8'))
        for point in points:
            f.write(struct.pack('fff', *point))

def main():
    input_txt_path = 'reconstructions/savereconstruction/full_trajectory2.txt'
    output_ply_path = 'reconstructions/savereconstruction/full_trajectory2.ply'
    
    matrices = read_matrices_from_txt(input_txt_path)
    points = [matrix[:3, 3] for matrix in matrices]
    write_ply(output_ply_path, points)
    
if __name__ == "__main__":
    main()

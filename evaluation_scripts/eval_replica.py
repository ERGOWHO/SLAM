import numpy as np
import struct
from evo.core.trajectory import PosePath3D
from evo.core.metrics import APE, PoseRelation
from evo.tools.file_interface import read_tum_trajectory_file

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

def matrices_to_posepath(matrices):
    timestamps = np.arange(len(matrices)) * 0.1  # 假设时间间隔为0.1秒
    positions = [matrix[:3, 3] for matrix in matrices]
    orientations = [matrix[:3, :3] for matrix in matrices]
    poses = [np.hstack((orientation, position.reshape(-1, 1))) for orientation, position in zip(orientations, positions)]
    poses = [np.vstack((pose, [0, 0, 0, 1])) for pose in poses]
    return PosePath3D(timestamps, poses)

def main():
    input_txt_path_A = 'ATE_compare/full_trajectory.txt'
    input_txt_path_B = 'ATE_compare/camera_trajectory_gt_replica_room0.txt'
    output_ply_path_A = 'ATE_compare/full_trajectory1.ply'
    output_ply_path_B = 'ATE_compare/camera_trajectory_gt_replica_room0.ply'
    
    matrices_A = read_matrices_from_txt(input_txt_path_A)
    matrices_B = read_matrices_from_txt(input_txt_path_B)
    
    points_A = [matrix[:3, 3] for matrix in matrices_A]
    points_B = [matrix[:3, 3] for matrix in matrices_B]
    
    write_ply(output_ply_path_A, points_A)
    write_ply(output_ply_path_B, points_B)
    
    traj_A = matrices_to_posepath(matrices_A)
    traj_B = matrices_to_posepath(matrices_B)
    
    # Align and calculate ATE
    traj_B_aligned = traj_B.align(traj_A, correct_scale=True, correct_only_scale=False)
    
    ape_metric = APE(PoseRelation.translation_part)
    ape_metric.process_data((traj_A, traj_B_aligned))
    
    ape_result = ape_metric.get_result()
    print(ape_result.pretty_str())

if __name__ == "__main__":
    main()

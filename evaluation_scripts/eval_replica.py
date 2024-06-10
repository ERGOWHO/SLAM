import numpy as np
import argparse
from evo.core.sync import associate_trajectories
from evo.core.metrics import PoseRelation, APE
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape

def read_matrices_from_txt(file_path):
    """
    Reads a text file containing 4x4 matrices and returns them as a numpy array.
    """
    data = np.loadtxt(file_path)
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

def compute_ate(traj_ref, traj_est):
    """
    Computes the Absolute Trajectory Error (ATE) between two trajectories.
    """
    traj_ref, traj_est = associate_trajectories(traj_ref, traj_est)
    # ape_metric = APE(PoseRelation.translation_part)
    # ape_metric.process_data((traj_ref, traj_est))
    # result = ape_metric.get_all_statistics()

    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    return result

def main(args):
    # Read the matrices from the txt files
    matrices_A = read_matrices_from_txt(args.input_txt_path_A)
    matrices_B = read_matrices_from_txt(args.input_txt_path_B)

    # Save the matrices to PLY files
    save_matrices_to_ply(matrices_A, args.output_ply_path_A)
    save_matrices_to_ply(matrices_B, args.output_ply_path_B)

    # Generate timestamps for the trajectories
    timestamps_A = np.arange(matrices_A.shape[0], dtype=float)
    timestamps_B = np.arange(matrices_B.shape[0], dtype=float)

    # Convert matrices to PoseTrajectory3D for ATE computation
    traj_ref = PoseTrajectory3D(timestamps=timestamps_A, poses_se3=matrices_A)
    traj_est = PoseTrajectory3D(timestamps=timestamps_B, poses_se3=matrices_B)

    # Compute ATE
    ate_result = compute_ate(traj_ref, traj_est)
    print(f"ATE Result: {ate_result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ATE between two trajectories")
    parser.add_argument("--input_txt_path_A", default='ATE_compare/full_trajectory2.txt')
    parser.add_argument("--input_txt_path_B", default='ATE_compare/camera_trajectory_gt_replica_room0.txt')
    parser.add_argument("--output_ply_path_A", default='ATE_compare/full_trajectory2.ply')
    parser.add_argument("--output_ply_path_B", default='ATE_compare/camera_trajectory_gt_replica_room0.ply')
    args = parser.parse_args()

    main(args)

import numpy as np
import argparse
from evo.core.sync import associate_trajectories
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
import evo.main_ape as main_ape
from scipy.spatial.transform import Rotation as R

def read_matrices_from_txt(file_path):
    """
    Reads a text file containing 4x4 matrices and returns them as a numpy array.
    """
    data = np.loadtxt(file_path)
    matrices = data.reshape(-1, 4, 4)
    return matrices

def save_matrices_and_frustums_to_ply(matrices, output_file, fov_h=60, fov_v=45, scale=0.05):
    """
    Saves the translation components of 4x4 matrices to a PLY file
    and also saves the frustum vertices, generating frustums every fourth matrix.
    """
    header = '''ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
element face {face_count}
property list uchar int vertex_index
end_header
'''
    vertices = []
    faces = []
    vertex_index = 0

    fov_h_rad = np.deg2rad(fov_h)
    fov_v_rad = np.deg2rad(fov_v)

    for i in range(0, len(matrices), 10):  # Iterate every fourth matrix
        matrix = matrices[i]
        pos = matrix[:3, 3]
        rot_matrix = matrix[:3, :3]

        # Camera direction
        dir_camera = np.dot(rot_matrix, np.array([0, 0, 1]))
        dir_camera /= np.linalg.norm(dir_camera)

        # Camera up vector
        up_vector = np.dot(rot_matrix, np.array([0, -1, 0]))
        up_vector /= np.linalg.norm(up_vector)

        # Camera right vector
        right_vector = np.cross(dir_camera, up_vector)
        right_vector /= np.linalg.norm(right_vector)

        # Points on image plane
        d = scale  # Distance from camera to image plane
        h = d * np.tan(fov_v_rad / 2)  # Half height of image plane
        w = d * np.tan(fov_h_rad / 2)  # Half width of image plane

        center = pos + d * dir_camera
        corners = [
            center + h * up_vector + w * right_vector,
            center + h * up_vector - w * right_vector,
            center - h * up_vector - w * right_vector,
            center - h * up_vector + w * right_vector
        ]

        # Add vertices for the frustum
        vertices.append(pos)
        vertices.extend(corners)

        # Define faces for the frustum
        frustum_faces = [
            (0, 1, 2),
            (0, 2, 3),
            (0, 3, 4),
            (0, 4, 1),
            (1, 2, 3, 4)
        ]
        
        # Add faces with offset
        for face in frustum_faces:
            faces.append(tuple(vertex_index + i for i in face))

        vertex_index += 5  # Update index offset

    vertex_count = len(vertices)
    face_count = len(faces)

    # Write to PLY file
    with open(output_file, 'w') as f:
        f.write(header.format(vertex_count=vertex_count, face_count=face_count))
        for vertex in vertices:
            f.write(f'{vertex[0]} {vertex[1]} {vertex[2]}\n')
        for face in faces:
            f.write(f'{len(face)} {" ".join(map(str, face))}\n')

def compute_ate(traj_ref, traj_est):
    """
    Computes the Absolute Trajectory Error (ATE) between two trajectories.
    """
    traj_ref, traj_est = associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)
    return result

def main(args):
    # Read the matrices from the txt files
    matrices_A = read_matrices_from_txt(args.input_txt_path_A)
    matrices_B = read_matrices_from_txt(args.input_txt_path_B)

    # Save the matrices and frustums to PLY files
    save_matrices_and_frustums_to_ply(matrices_A, args.output_ply_path_A)
    save_matrices_and_frustums_to_ply(matrices_B, args.output_ply_path_B)

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
    parser.add_argument("--input_txt_path_A", default='reconstructions/savereconstruction/full_trajectory2.txt')
    parser.add_argument("--input_txt_path_B", default='ATE_compare/camera_trajectory_gt_replica_room0.txt')
    parser.add_argument("--output_ply_path_A", default='reconstructions/savereconstruction/full_trajectory2.ply')
    parser.add_argument("--output_ply_path_B", default='ATE_compare/camera_trajectory_gt_replica_room0.ply')
    args = parser.parse_args()

    main(args)

import numpy as np

# Load the poses.npy file
poses_path = "reconstructions/savereconstruction/poses.npy"
poses = np.load(poses_path)

# Convert poses to 4x4 matrices
def convert_poses_to_4x4(poses):
    traj = []
    for pose in poses:
        tx, ty, tz, qx, qy, qz, qw = pose
        rotation_matrix = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw), tx],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw), ty],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2), tz],
            [0, 0, 0, 1]
        ])
        traj.append(rotation_matrix)
    return np.array(traj)

# Convert and save the trajectory
est_traj = convert_poses_to_4x4(poses)
output_path = "reconstructions/savereconstruction/est_traj.npy"
np.save(output_path, est_traj)

output_path

import numpy as np
import torch

# 定义align函数
def align(model, data):
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.svd(W.transpose())
    S = np.identity(3)
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U @ S @ Vh
    trans = data.mean(1).reshape((3,-1)) - rot @ model.mean(1).reshape((3,-1))

    model_aligned = rot @ model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0))

    return rot, trans, trans_error

# 计算ATE的函数
def evaluate_ate(gt_traj, est_traj):
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  = torch.tensor(gt_traj_pts).T
    est_traj_pts = torch.tensor(est_traj_pts).T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean().item()

    return avg_trans_error

# 加载和转换ground truth轨迹
def load_gt_traj(filepath):
    gt_poses = []
    with open(filepath, 'r') as f:
        for line in f:
            pose = list(map(float, line.strip().split()))
            gt_poses.append(np.array(pose).reshape(4, 4))
    return np.array(gt_poses)

gt_traj = load_gt_traj('reconstructions/savereconstruction/traj.txt')

# 加载估计的轨迹
est_traj = np.load("reconstructions/savereconstruction/est_traj.npy")

# 确保估计的轨迹也是4x4的矩阵
if est_traj.shape[1] != 4:
    est_traj = est_traj.reshape(-1, 4, 4)

# 计算ATE
avg_trans_error = evaluate_ate(gt_traj, est_traj)
print("Average Translational Error (ATE):", avg_trans_error)

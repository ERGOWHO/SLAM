import numpy as np

# 读取 full_trajectory.txt 文件
traj_data = np.loadtxt('full_trajectory.txt')

# 创建 PLY 文件头
ply_header = '''ply
format ascii 1.0
element vertex {0}
property float x
property float y
property float z
property float qw
property float qx
property float qy
property float qz
end_header
'''.format(traj_data.shape[0])

# 将数据保存为 PLY 文件
ply_data = np.hstack((traj_data[:, :3], traj_data[:, 3:]))
ply_filename = 'full_trajectory.ply'

with open(ply_filename, 'w') as f:
    f.write(ply_header)
    np.savetxt(f, ply_data, fmt='%f %f %f %f %f %f %f')

print(f"PLY file saved to {ply_filename}")

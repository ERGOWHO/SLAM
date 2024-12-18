import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import numpy as np
from torch.multiprocessing import Process
from droid import Droid
from scipy.spatial.transform import Rotation as R
from lietorch import SE3
import torch.nn.functional as F

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
def convert_to_4x4_matrix(traj):
    matrices = []
    for row in traj:
        x, y, z, tx, ty, tz, tw = row
        translation = np.array([x, y, z])
        rotation = R.from_quat([tx, ty, tz, tw])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = translation
        matrices.append(matrix)
    return matrices

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string
    init_data = np.array([0., 0., 0., 0., 0., 0., 1.])
    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)

def getPosefromSlam(pose):
    # format x, y, z, tx, ty, tz, tw = pose
    pose = torch.tensor(pose, dtype=torch.float32)
    pose = SE3(pose).inv().matrix()

    R = pose[:3, :3]
    R = R.t()

    T = pose[:3, 3]
    
    T = torch.matmul(-R, T.t())
 
    # R = R.t()
    T = T.t()
    T = T*10

    return R,T


def save_images_and_camera_info(droid, output_dir,traj):
    # Save images.txt
    # poses_matrices = np.array(poses_matrices)
    # traj = torch.tensor(np.array(traj), dtype=torch.float32)
    # world2cam = invert_matrix(poses_matrices)
    with open(os.path.join(output_dir, "images.txt"), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        # for idx ,pose in enumerate(traj, 1):
        for idx, pose in enumerate(droid.video.poses[:droid.video.counter.value]):
            # rotation_matrix = world2cam[idx, :3, :3]
            # qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            # tx, ty, tz = world2cam[idx, :3, 3]

            R_matrix, T_vector = getPosefromSlam(pose)
            
            R_matrix = R_matrix.cpu().numpy()
            T_vector = T_vector.cpu().numpy()
            
            qw, qx, qy, qz = rotmat2qvec(R_matrix)
            tx, ty, tz = T_vector
            image_name = f"gt_{idx}.png"  
            f.write(f"{idx} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {image_name}\n")
            f.write(f"\n")


            
    # Save camera.txt
    with open(os.path.join(output_dir, "cameras.txt"), 'w') as f:
        # Assuming the camera model is SIMPLE_PINHOLE and the parameters are as given
        f.write("1 PINHOLE 1599 895 1554.6 1546.7 799.5 447.5\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=300)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=0, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')
    # save traj
    output_dir = "reconstructions/{}/".format(args.reconstruction_path)
    os.makedirs(output_dir, exist_ok=True)

    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)
        print(droid.video.counter.value)
    
    traj_est_full = droid.get_full_est_traj(image_stream(args.imagedir, args.calib, args.stride))
    matrices = convert_to_4x4_matrix(traj_est_full)

    with open(os.path.join(output_dir, "full_trajectory_today.txt"), 'w') as f:
        for matrix in matrices:     
            matrix_flat = matrix.flatten()  
            matrix_str = ' '.join(map(str, matrix_flat)) 
            f.write(matrix_str + '\n')
    print("Full trajectory saved to full_trajectory.txt")  

    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)
        save_images_and_camera_info(droid, output_dir,traj_est_full)

    # traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
    # if args.reconstruction_path is not None:
    #     save_reconstruction(droid, args.reconstruction_path)
    #     save_images_and_camera_info(droid, output_dir,traj_est_full)

    # traj_est_full2 = droid.get_full_est_traj(image_stream(args.imagedir, args.calib, args.stride))
    # matrices2 = convert_to_4x4_matrix(traj_est_full2)
    # with open(os.path.join(output_dir, "full_trajectory2.txt"), 'w') as f2:
    #     for matrix2 in matrices2:     
    #         matrix_flat2 = matrix2.flatten()  
    #         matrix_str2 = ' '.join(map(str, matrix_flat2)) 
    #         f2.write(matrix_str2 + '\n')
    # print("Full trajectory saved to full_trajectory2.txt")  
    
    
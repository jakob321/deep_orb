import os
import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math


def load_velodyne_points(bin_path: str) -> np.ndarray:
    # Each row in the .bin file is [x, y, z, reflectance]
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def load_data():
    all_data = np.load("saved_data.npz", allow_pickle=True)
    return all_data["pose"], all_data["depth"], all_data["orig"], all_data["points"]


def load_calib_velo_to_cam(filename):
    """
    Load the Velodyne-to-camera calibration from the provided file.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find and parse the R and T lines.
    R_line = [line for line in lines if line.startswith("R:")][0]
    T_line = [line for line in lines if line.startswith("T:")][0]

    # Extract the numbers after the prefix "R:" and "T:"
    R_values = list(map(float, R_line.strip().split()[1:]))
    T_values = list(map(float, T_line.strip().split()[1:]))

    # Convert to numpy arrays and reshape
    R = np.array(R_values).reshape(3, 3)
    T = np.array(T_values).reshape(3, 1)

    # Create the 4x4 transformation matrix
    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[:3, :3] = R
    Tr_velo_to_cam[:3, 3] = T.flatten()

    return Tr_velo_to_cam

from scipy.spatial import cKDTree

def compute_chamfer_distance(cloudA: np.ndarray, cloudB: np.ndarray) -> float:
    """
    Computes a simple Chamfer distance between two point clouds.
    cloudA, cloudB: arrays of shape (N,3) and (M,3) respectively.
    Returns the sum of the average nearest neighbor distances in both directions.
    """
    # Build KD-Trees for fast nearest-neighbor search.
    treeA = cKDTree(cloudA)
    treeB = cKDTree(cloudB)
    
    # For each point in A, find the distance to its nearest neighbor in B.
    distances_A_to_B, _ = treeB.query(cloudA)
    # For each point in B, find the distance to its nearest neighbor in A.
    distances_B_to_A, _ = treeA.query(cloudB)
    
    chamfer = np.mean(distances_A_to_B) + np.mean(distances_B_to_A)
    return chamfer

def compute_sparse_to_dense_alignment_metric(sparse_cloud: np.ndarray, dense_cloud: np.ndarray) -> float:
    """
    Computes the average nearest neighbor distance from each point in the sparse_cloud
    (e.g., ORB SLAM points) to the nearest point in the dense_cloud (e.g., Velodyne points).
    This gives a measure of how well the sparse points align with the dense cloud.
    """
    dense_tree = cKDTree(dense_cloud)
    distances, _ = dense_tree.query(sparse_cloud)
    return np.mean(distances)

# Folder containing KITTI .bin files
bin_folder = "../datasets/kitti/2011_09_26_drive_0093_sync/velodyne_points/data"
# Get all .bin files, sorted
bin_files = sorted(glob.glob(os.path.join(bin_folder, "*.bin")))
print(len(bin_files))

# Camera intrinsics
fx = 721.5377
fy = 721.5377
cx = 609.5593
cy = 172.8540

if __name__ == "__main__":
    # Define the scaling factor to apply ONLY to ORB SLAM points.
    orb_scale_factor = 1.2  # Adjust as needed for the ORB SLAM point cloud

    # Velodyne and RGBD are assumed to be at the correct absolute scale, so we do not apply an extra scale here.
    
    # Load calibration and print transformation matrix.
    calib_file = 'calib_velo_to_cam.txt'
    T_velo_to_cam = load_calib_velo_to_cam(calib_file)
    print("Velodyne-to-Cam Transformation Matrix:\n", T_velo_to_cam)

    # Create a visualizer window.
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=500, height=500)

    # Load data.
    all_pose, all_depth, all_orig, points_list = load_data()
    WIDTH = all_orig.shape[2]
    HEIGHT = all_orig.shape[1]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)

    for i, bin_file in enumerate(bin_files):
        # if i < 2 or i > 400: continue
        # if i % 20 > 0:continue
        if i not in [100]: continue

        orb_pose = all_pose[i, :, :].copy()
        orb_pose[:3, 3] *= orb_scale_factor

        # ========================================================
        # ======= Velodyne Point Cloud Creation =======
        # ========================================================
        # Load and slice out only x, y, z (ignoring reflectance)
        points = load_velodyne_points(bin_file)[:, :3]
        N = points.shape[0]
        points_hom = np.hstack((points, np.ones((N, 1))))  # (N, 4)
        points_cam_hom = (T_velo_to_cam @ points_hom.T).T
        # Apply the conversion factor (0.1) only.
        points_cam = points_cam_hom[:, :3] * 0.1

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_cam)
        points_array = np.asarray(pcd.points)

        # Color the point cloud based on the z-axis values.
        axis_values = points_array[:, 2]
        min_val = axis_values.min()
        max_val = axis_values.max()
        norm_values = (axis_values - min_val) / (max_val - min_val + 1e-8)
        cmap = plt.get_cmap('viridis')
        colors = cmap(norm_values)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Apply the pose transformation without additional scaling.
        pcd.transform(orb_pose)
        visualizer.add_geometry(pcd)

        # Add camera visualization lines.
        
        standardCameraParametersObj = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=800,
            view_height_px=400,
            intrinsic=standardCameraParametersObj.intrinsic.intrinsic_matrix,
            extrinsic=np.linalg.inv(orb_pose),
            scale=0.1)
        visualizer.add_geometry(cameraLines)

        # # ========================================================
        # # ======= RGBD Point Cloud Creation =======
        # # ========================================================
        # color_raw = o3d.geometry.Image(all_orig[i, :, :, :])
        # depth_raw = o3d.geometry.Image(all_depth[i, :, :] * 100)
        # position = np.linalg.inv(orb_pose)
        # # No extra scaling is applied here.
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     color_raw, depth_raw, convert_rgb_to_intensity=False)
        # pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        # pcd2.transform(np.linalg.inv(position))
        # visualizer.add_geometry(pcd2)

        # ========================================================
        # ======= ORB SLAM Point Cloud Creation =======
        # ========================================================
        combined_coords = points_list[i]
        frame_points = combined_coords.T
        # Apply the orb_scale_factor ONLY to the ORB points.
        frame_points = orb_scale_factor * frame_points
        orb_points = o3d.geometry.PointCloud()
        orb_points.points = o3d.utility.Vector3dVector(frame_points)
        orb_points.paint_uniform_color([0.0, 1.0, 0.0])
        visualizer.add_geometry(orb_points)

        # ========================================================
        # ======= ORB SLAM Point Cloud Creation (Scaled & Filtered) =======
        # ========================================================
        combined_coords = points_list[i]
        frame_points = combined_coords.T
        # Apply scaling only to ORB SLAM points.
        frame_points = orb_scale_factor * frame_points

        # --- Filtering based on vertical angle ---
        # Compute the ORB camera pose (with scaled translation) for the camera center.
        orb_pose = all_pose[i, :, :].copy()
        orb_pose[:3, 3] *= orb_scale_factor
        camera_center = orb_pose[:3, 3].reshape(1, 3)

        # Compute vectors from the camera center to each ORB point.
        diff = frame_points - camera_center
        # Horizontal distance (using x and y components).
        horizontal_distance = np.linalg.norm(diff[:, :2], axis=1)
        #print(horizontal_distance)
        # Vertical angle (angle between diff vector and the horizontal plane).
        vertical_angles = np.arctan2(diff[:, 1], horizontal_distance)#-math.pi/2

        #print(vertical_angles)
        # Define maximum allowed vertical angle (e.g., 10 degrees).
        max_vertical_angle = np.deg2rad(10)
        #print("---------------------", max_vertical_angle)
        # Create a mask to keep only points within the allowed vertical field.
        valid_mask = -vertical_angles <= max_vertical_angle
        filtered_frame_points = frame_points[valid_mask]
        #print("valid points", filtered_frame_points.shape)
        # -------------------------------------------------

        orb_points = o3d.geometry.PointCloud()
        orb_points.points = o3d.utility.Vector3dVector(filtered_frame_points)
        orb_points.paint_uniform_color([1.0, 0.0, 0.0])
        visualizer.add_geometry(orb_points)

        chamfer = compute_chamfer_distance(filtered_frame_points, np.asarray(pcd.points))
        alignment_metric = compute_sparse_to_dense_alignment_metric(filtered_frame_points, np.asarray(pcd.points))
        print("Chamfer distance between clouds:", chamfer)
        print("Average NN distance from ORB to Velodyne:", alignment_metric)

    visualizer.run()

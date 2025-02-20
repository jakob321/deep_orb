# import os
# import numpy as np
# import open3d as o3d

# def load_velodyne_points(bin_path: str) -> np.ndarray:
#     scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
#     return scan

# def visualize_velodyne_points(points: np.ndarray):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points[:, :3])
#     o3d.visualization.draw_geometries([pcd])

# if __name__ == "__main__":
#     # Example path: adjust to your KITTI folder structure
#     bin_file = "../datasets/kitti/2011_09_26_drive_0093_sync/velodyne_points/data/0000000000.bin"
#     if not os.path.exists(bin_file):
#         raise FileNotFoundError(f"File not found: {bin_file}")
    
#     # Load the Velodyne scan
#     points = load_velodyne_points(bin_file)
#     print(f"Loaded {points.shape[0]} points from {bin_file}")
    
#     # Visualize
#     visualize_velodyne_points(points)
import os
import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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

# Folder containing KITTI .bin files
bin_folder = "../datasets/kitti/2011_09_26_drive_0093_sync/velodyne_points/data"

# Get all .bin files, sorted
bin_files = sorted(glob.glob(os.path.join(bin_folder, "*.bin")))

# If you want every other file, slice with [::10]
bin_files = bin_files[::10]
print(len(bin_files))
bin_files = bin_files[:40]

if __name__ == "__main__":

    # Example usage:
    calib_file = 'calib_velo_to_cam.txt'
    T_velo_to_cam = load_calib_velo_to_cam(calib_file)
    print("Velodyne-to-Cam Transformation Matrix:\n", T_velo_to_cam)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=500, height=500)

    all_pose, all_depth, all_orig, points_list = load_data()

    for i, bin_file in enumerate(bin_files):
        if i%2>0:continue
        # Load and slice out only x,y,z (ignoring reflectance)
        points = load_velodyne_points(bin_file)[:, :3]
        N = points.shape[0]
        points_hom = np.hstack((points, np.ones((N, 1))))  # (N, 4)
        points_cam_hom = (T_velo_to_cam @ points_hom.T).T
        points_cam = points_cam_hom[:, :3]*0.1

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_cam)
        points_array = np.asarray(pcd.points)

        # For example, if you want the gradient to be based on the x-axis:
        axis_values = points_array[:, 2]  # Change this to 1 for y-axis, or 2 for z-axis

        # Normalize the axis values between 0 and 1.
        min_val = axis_values.min()
        max_val = axis_values.max()
        norm_values = (axis_values - min_val) / (max_val - min_val + 1e-8)

        # Choose a colormap (e.g., 'viridis', 'plasma', etc.)
        cmap = plt.get_cmap('viridis')

        # Map normalized values to RGB colors.
        colors = cmap(norm_values)[:, :3]  # Extract only RGB channels, ignoring alpha

        # Assign these colors to the point cloud.
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Apply the pose transformation (inverse of camera pose)
        pcd.transform(all_pose[i, :, :])
        visualizer.add_geometry(pcd)

        standardCameraParametersObj = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=800, 
            view_height_px=400, 
            intrinsic=standardCameraParametersObj.intrinsic.intrinsic_matrix, 
            extrinsic=np.linalg.inv(np.array(all_pose[i,:,:])),
            scale=1)
        visualizer.add_geometry(cameraLines)

    visualizer.run()


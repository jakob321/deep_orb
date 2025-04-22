import threading
import orbslam3
import time
import numpy as np
from . import generic_helper
from helper import vkitti

# ====== ORB-SLAM Functions ======
import threading
import time
import orbslam3

# Define global variables that will be accessible from outside
final_result = None
current_pose_list = None
current_points_list = None
current_points_2d = None
is_slam_running = False

def run_orb_slam(settings_file, path_dataset):
    voc_file = "../ORB_SLAM3/Vocabulary/ORBvoc.txt"
    global current_pose_list, current_points_list, current_points_2d, is_slam_running, final_result
    
    def run_slam():
        global final_result
        final_result = orbslam3.run_orb_slam3(voc_file, settings_file, path_dataset, fps=120)
    
    # Start the SLAM thread
    slam_thread = threading.Thread(target=run_slam)
    slam_thread.start()
    is_slam_running = True
    
    # Start a separate thread for continuously updating the global variables
    def update_data():
        global current_pose_list, current_points_list, current_points_2d, is_slam_running
        while slam_thread.is_alive():
            poses, points = orbslam3.get_all_data_np()
            points_2d = orbslam3.get_2d_points()
            
            # Update the global variables
            current_pose_list = poses.transpose(2, 0, 1) if poses is not None else None
            current_points_list = points
            current_points_2d = points_2d[0] if points_2d is not None else None
            
            time.sleep(0.01)
        is_slam_running = False
    
    # Start the update thread
    update_thread = threading.Thread(target=update_data)
    update_thread.daemon = True  # This thread will exit when the main program exits
    update_thread.start()
    
    # Wait for SLAM to finish
    slam_thread.join()
    
    # Return the final results
    return current_pose_list, current_points_list, current_points_2d

def started():
    global current_pose_list
    return len(current_pose_list)!=0

# Getter functions to access the current data
def get_current_pose_list():
    global current_pose_list
    return current_pose_list

def get_current_points_list():
    global current_points_list
    return current_points_list

def get_current_points_2d():
    global current_points_2d
    return current_points_2d

def is_slam_thread_running():
    global is_slam_running
    return is_slam_running

def predict_orb_scale(orb_points, deep_points):
    u_coords = np.round(orb_points[0, :]).astype(int)
    v_coords = np.round(orb_points[1, :]).astype(int)
    orb_depths = orb_points[2, :]

    # For each orb point, get the corresponding depth value from the deep network depth map.
    deep_depths = deep_points[v_coords, u_coords]

    # Compute the ratio for each point
    scales = deep_depths / orb_depths

    # The scaling factor is the median of these ratios
    # Initial robust estimate using the median
    med = np.median(scales)
    # Compute Median Absolute Deviation (MAD)
    mad = np.median(np.abs(scales - med))
    
    # Define inliers as points within threshold_multiplier * MAD of the median
    threshold_multiplier=1.25
    inliers = np.abs(scales - med) < threshold_multiplier * mad
    if np.sum(inliers) == 0:
        # Fallback if no inliers are found
        return med
    # Recompute the scale using the median of inliers
    scaling_factor = np.median(scales[inliers])
    
    return scaling_factor

def compute_keypoint_errors(true_depth, points_uvd):
    """
    Compute the absolute and percentage error for keypoints.
    points_uvd: (3, N) array where rows are [u, v, predicted_depth].
    """
    u_coords = np.round(points_uvd[0, :]).astype(int)
    v_coords = np.round(points_uvd[1, :]).astype(int)
    pred_depths = points_uvd[2, :]

    # Get the true depth at each keypoint location
    true_depths = true_depth[v_coords, u_coords]

    with np.errstate(divide='ignore', invalid='ignore'):
        errors = np.abs(pred_depths - true_depths)
        percentage = np.where(true_depths != 0, (errors / true_depths) * 100, 0)
    return errors, percentage

def run_if_no_saved_values(dataset, override_run=False):
    settings_file = dataset.settings_file
    path_dataset = dataset.get_rgb_folder_path()
    result = generic_helper.load_data(path_dataset)
    if result == False or override_run:
        new_result = run_orb_slam(settings_file, path_dataset)
        generic_helper.save_data(path_dataset, *new_result)
        return new_result
    return result

def predict_orb_scale(scales):
    med = np.median(scales)
    # Compute Median Absolute Deviation (MAD)
    mad = np.median(np.abs(scales - med))
    
    # Define inliers as points within threshold_multiplier * MAD of the median
    threshold_multiplier=2
    inliers = np.abs(scales - med) < threshold_multiplier * mad
    if np.sum(inliers) == 0:
        # Fallback if no inliers are found
        return med
    # Recompute the scale using the median of inliers
    scaling_factor = np.median(scales[inliers])
    
    return scaling_factor

def all_scales(orb_points_list, deep_points_list):
    """
    Computes the ratio between deep network depth and ORB-SLAM depth for multiple frames.
    Parameters:
    orb_points_list (list of np.array): Each array is (3, N), representing ORB depth points per frame.
    deep_points_list (list of np.array): Each array represents the dense depth map for the corresponding frame.
    Returns:
    np.array of shape (N, 1): A flattened array containing all computed scale ratios across all frames.
    """
    all_ratios = []
    for i in range(len(orb_points_list)):
        orb_points = orb_points_list[i]
        deep_points = deep_points_list[i]
        
        # Get height and width of the depth image
        height, width = deep_points.shape[:2]
        
        u_coords = np.round(orb_points[0, :]).astype(int)
        v_coords = np.round(orb_points[1, :]).astype(int)
        orb_depths = orb_points[2, :]
        
        # Check which coordinates are within the image boundaries
        valid_indices = (u_coords >= 0) & (u_coords < width) & (v_coords >= 0) & (v_coords < height)
        
        # Filter coordinates and depths based on validity
        valid_u_coords = u_coords[valid_indices]
        valid_v_coords = v_coords[valid_indices]
        valid_orb_depths = orb_depths[valid_indices]
        
        # Get the corresponding dense depth values at valid ORB point locations
        deep_depths = deep_points[valid_v_coords, valid_u_coords]
        
        # Compute the scale ratio for each valid point
        ratios = deep_depths / valid_orb_depths
        
        # Append all values into a single list
        all_ratios.extend(ratios)
    
    # Convert to a NumPy array with shape (N, 1)
    return np.array(all_ratios).reshape(-1, 1)



def compute_true_scale(slam_matrices: np.ndarray, gt_positions: np.ndarray):
    """
    Computes the true scale for mono-SLAM extrinsics based on ground truth positions.
    
    Args:
        slam_matrices (np.ndarray): Array of shape (n, 4, 4) representing the mono-SLAM extrinsic matrices.
        gt_positions (np.ndarray): Array of shape (n, 3) representing the ground truth camera positions.
    
    Returns:
        scale_factor (float): The computed scale factor.
        scaled_slam_matrices (np.ndarray): The SLAM extrinsic matrices with scaled translations.
    """
    # Extract translation components from SLAM matrices (shape: (n, 3))
    slam_positions = slam_matrices[:, :3, 3]
    
    # Compute distances between consecutive frames for ground truth and SLAM estimated positions
    gt_dists = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)
    slam_dists = np.linalg.norm(np.diff(slam_positions, axis=0), axis=1)
    
    # Sum the distances to get the total path length
    gt_path_length = np.sum(gt_dists)
    slam_path_length = np.sum(slam_dists)
    
    # Compute the scale factor
    if slam_path_length == 0:
        return 1, slam_matrices
        # raise ValueError("The SLAM path length is zero. Check your input data.")
    
    scale_factor = gt_path_length / slam_path_length
    
    # Apply the scale factor to the translation components of the SLAM matrices
    scaled_slam_matrices = slam_matrices.copy()
    scaled_slam_matrices[:, :3, 3] *= scale_factor
    
    return scale_factor, scaled_slam_matrices

# def sim_run_if_no_saved_values(dataset, override_run=False):
#     settings_file = dataset.settings_file
#     path_dataset = dataset.get_rgb_folder_path()
#     result = generic_helper.load_data(path_dataset)
#     if not result: # If there is no saved data
#         new_result = run_orb_slam(settings_file, path_dataset)
#         generic_helper.save_data(path_dataset, *new_result)
#         return new_result
#     return result

def sim_run_orb_slam(dataset, path_dataset, sim_delay=0.1):
    settings_file = dataset.settings_file
    path_dataset = dataset.get_rgb_folder_path()
    result = generic_helper.load_data(path_dataset)
    if not result: # If there is no saved data
        new_result = run_orb_slam(settings_file, path_dataset)
        generic_helper.save_data(path_dataset, *new_result)
        return new_result

    # If there is saved data we simulate that it is updated live
    pose_list, points_list, points_2d = result
    global current_pose_list, current_points_list, current_points_2d, is_slam_running, final_result
    for current_index in range(len(pose_list)-1500):
        is_slam_running = True
        time.sleep(sim_delay)
        current_points_2d = points_2d[:current_index]
        current_points_list = points_list[:current_index]
        current_pose_list = pose_list[:current_index]
    is_slam_running=False
    final_result=result
    return current_pose_list, current_points_list, current_points_2d
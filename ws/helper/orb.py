import threading
import orbslam3
import time
import numpy as np
from . import generic_helper
from helper import vkitti

# ====== ORB-SLAM Functions ======
def run_orb_slam(voc_file, settings_file, path_dataset):
    def run_slam():
        global final_result
        final_result = orbslam3.run_orb_slam3(voc_file, settings_file, path_dataset, fps=100)

    slam_thread = threading.Thread(target=run_slam)
    slam_thread.start()

    pose_list, points_list, points_2d = (0, 0, 0)
    time.sleep(0)
    while slam_thread.is_alive():
        pose_list, points_list = orbslam3.get_all_data_np()
        points_2d = orbslam3.get_2d_points()
        time.sleep(0.01)
    return pose_list.transpose(2, 0, 1), points_list, points_2d[0]

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
    voc_file = "../ORB_SLAM3/Vocabulary/ORBvoc.txt"
    settings_file = dataset.settings_file
    path_dataset = dataset.get_rgb_folder_path()
    result = generic_helper.load_data(path_dataset)
    if result == False or override_run:
        new_result = run_orb_slam(voc_file, settings_file, path_dataset)
        generic_helper.save_data(path_dataset, *new_result)
        return new_result
    return result
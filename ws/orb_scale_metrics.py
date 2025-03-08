from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np

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

        u_coords = np.round(orb_points[0, :]).astype(int)
        v_coords = np.round(orb_points[1, :]).astype(int)
        orb_depths = orb_points[2, :]

        # Get the corresponding dense depth values at ORB point locations.
        deep_depths = deep_points[v_coords, u_coords]

        # Compute the scale ratio for each point.
        ratios = deep_depths / orb_depths

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
        raise ValueError("The SLAM path length is zero. Check your input data.")
    
    scale_factor = gt_path_length / slam_path_length
    
    # Apply the scale factor to the translation components of the SLAM matrices
    scaled_slam_matrices = slam_matrices.copy()
    scaled_slam_matrices[:, :3, 3] *= scale_factor
    
    return scale_factor, scaled_slam_matrices

def main():
    # This file creates metrics for how well the scaling of ORB works using deep points
    
    # We use the dataset vkitti and midair for testing. The sequences have been choosen for ORB to not lose tracking
    dataset = vkitti.dataset("midair")
    vkitti_seq=[1]

    dataset = vkitti.dataset("vkitti")
    vkitti_seq=[0]
    # a=dataset.get_all_seq()
    # print(a)
    # return
    for seq in vkitti_seq:
        dataset.set_sequence(seq) 
        pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
        number_of_deep_frames = 10
        deep_frames_index = np.linspace(1, len(points_2d)-1, number_of_deep_frames, dtype=int).tolist()
        rgb_path = [dataset.get_rgb_frame_path()[i] for i in deep_frames_index]
        pred_depth, rgb_img = deep.run_if_no_saved_values(rgb_path, override_run=False)
        selected_orb_frames = [points_2d[i] for i in deep_frames_index]
        scales = all_scales(selected_orb_frames, pred_depth)
        scale_factor, scaled_slam_matrices = compute_true_scale(pose_list, dataset.load_extrinsic_matrices())
        predicted_scale = predict_orb_scale(scales)

        print("predicted scale: ",predicted_scale)
        print("true scale: ", scale_factor)
        plots.plot_error_histograms(scales, 
            num_bins=100, 
            log_scale_x=True, 
            label1="Single Error",
            ground_truth_line=scale_factor,
            pred_line=predicted_scale)


        # scale_list=[]
        # for i, index in enumerate(deep_frames_index):
        #     print(i,index)

        #     scales = predict_orb_scale(points_2d[index], pred_depth[i])
        #     scale_list.append(prediced_scale)
        #     print(prediced_scale)

        # avg_scale=np.average(np.array(scale_list))
        # print("avg_scale")
        # print(avg_scale)

        # print("true scale")
        # scale_factor, scaled_slam_matrices = compute_true_scale(pose_list, dataset.load_extrinsic_matrices())
        # print(scale_factor)


if __name__ == "__main__":
    main()

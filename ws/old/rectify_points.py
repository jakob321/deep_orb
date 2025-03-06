import numpy as np
from PIL import Image
import depth_pro
import orbslam3
import time
import threading
import torch
import torch.cuda.amp as amp
import os
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
# import ctypes; ctypes.cdll.LoadLibrary('/usr/local/lib/libopencv_core.so')
import pickle

# ====== Global configuration & paths ======
path_depth = "../datasets/vkitti/vkitti_1.3.1_depthgt/0002/clone"
pathDataset = "../datasets/vkitti/vkitti_1.3.1_rgb/0002/clone"
voc_file = "../ORB_SLAM3/Vocabulary/ORBvoc.txt"
settings_file = "../ORB_SLAM3/Examples/Monocular/KITTI03.yaml"
timestamps = "../ORB_SLAM3/Examples/Monocular/EuRoC_TimeStamps/MH01.txt"

fx = 721.5377
fy = 721.5377
cx = 609.5593
cy = 172.8540

# ====== Inference Functions ======
def run_inference_mixed(model, images, f_px, label=""):
    with torch.no_grad(), amp.autocast():
        start_time = time.time()
        prediction = model.infer(images, f_px=f_px)
        torch.cuda.synchronize()  # Wait for all GPU ops to complete.
        elapsed = time.time() - start_time
        print(f"Inference time {label} (mixed precision): {elapsed:.3f} seconds")
    return prediction

def run_inference_half(model, images, f_px, label=""):
    # Convert model and images to half precision.
    model.half()
    images = images.half()
    with torch.no_grad():
        start_time = time.time()
        prediction = model.infer(images, f_px=f_px)
        torch.cuda.synchronize()  # Wait for all GPU ops to complete.
        elapsed = time.time() - start_time
        print(f"Inference time {label} (all FP16): {elapsed:.3f} seconds")
    return prediction

# ====== ORB-SLAM Functions ======
def run_orb_slam():
    def run_slam():
        global final_result
        final_result = orbslam3.run_orb_slam3(voc_file, settings_file, pathDataset, timestamps, fps=20)

    slam_thread = threading.Thread(target=run_slam)
    slam_thread.start()

    pose_list, points_list, points_2d = (0, 0, 0)
    time.sleep(0)
    while slam_thread.is_alive():
        pose_list, points_list = orbslam3.get_all_data_np()
        points_2d = orbslam3.get_2d_points()
        time.sleep(0.01)
    return pose_list, points_list, points_2d[0]

# ====== Depth Model Functions ======
def load_depth_pro():
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device)
    model.eval()
    return model, device

def run_depth_pro(paths):
    # Model initialization
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device)
    model.eval()

    all_depth = []
    all_orig = []
    for path in paths:
        original_image = np.asarray(Image.open(path))
        all_orig.append(original_image)
        
        image, _, f_px = depth_pro.load_rgb(path)
        image = transform(image).to(device)
        prediction = run_inference_half(model, image, f_px, label="img")
        depth = prediction["depth"].cpu().numpy()  # Convert depth tensor to numpy array
        all_depth.append(depth)

    return all_depth, all_orig

def load_data():
    all_data = np.load("saved_data.npz", allow_pickle=True)
    return all_data["pose"], all_data["depth"], all_data["orig"], all_data["points"]

# ====== Error Computation & Plotting Functions ======
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

def compute_full_depth_errors(true_depth, predicted_depth, clip_value=655.35):
    """
    Compute absolute error for the full depth map.
    """
    p_depth = predicted_depth.copy()
    p_depth[p_depth > clip_value] = clip_value
    full_errors = np.abs(true_depth - p_depth)
    return full_errors

def prepare_histogram_bins(errors1, errors2, num_bins=50, epsilon=1e-8):
    """
    Prepare common logarithmic bins for histogram plotting.
    Returns bins and flattened arrays for both error types.
    """
    errors1_flat = errors1.ravel()
    errors2_flat = errors2.ravel()
    errors1_flat = errors1_flat[errors1_flat > 0]
    errors2_flat = errors2_flat[errors2_flat > 0]
    min_val = np.minimum(errors1_flat.min(), errors2_flat.min())
    max_val = np.maximum(errors1_flat.max(), errors2_flat.max())
    bins = np.logspace(np.log10(min_val + epsilon), np.log10(max_val + epsilon), num_bins)
    return bins, errors1_flat, errors2_flat

def plot_error_histograms(errors1, label1, errors2, label2, bins):
    """
    Plot a combined histogram comparing keypoint and full depth errors.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(errors1, bins=bins, density=True, alpha=0.5,
             label=label1, color='blue', edgecolor='black')
    plt.hist(errors2, bins=bins, density=True, alpha=0.5,
             label=label2, color='red', edgecolor='black')
    plt.xscale('log')
    plt.xlabel("Absolute Error (meters, log scale)")
    plt.ylabel("Normalized Frequency")
    plt.title("Comparison of Depth Absolute Errors")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

def predict_orb_scale(orb_points, deep_points):
    u_coords = np.round(orb_points[0, :]).astype(int)
    v_coords = np.round(orb_points[1, :]).astype(int)
    orb_depths = orb_points[2, :]

    # For each orb point, get the corresponding depth value from the deep network depth map.
    deep_depths = deep_points[v_coords, u_coords]

    # Compute the ratio for each point
    scales = deep_depths / orb_depths

    # Initial robust estimate using the median
    med = np.median(scales)
    # Compute Median Absolute Deviation (MAD)
    mad = np.median(np.abs(scales - med))
    
    # Define inliers as points within threshold_multiplier * MAD of the median
    threshold_multiplier = 1.25
    inliers = np.abs(scales - med) < threshold_multiplier * mad
    
    # Extract the inlier orb points (same shape as orb_points, but fewer points)
    inlier_orb_points = orb_points[:, inliers]
    
    if np.sum(inliers) == 0:
        # Fallback if no inliers are found: return median and an empty array of inlier orb points.
        return med, inlier_orb_points

    # Recompute the scale using the median of inlier ratios.
    scaling_factor = np.median(scales[inliers])
    
    return scaling_factor, inlier_orb_points


def guided_filter(I, p, r, eps):
    """
    Perform guided filtering.
    
    Parameters:
      I   : Guidance image (np.float32, shape: HxW)
      p   : Input image to be filtered (np.float32, shape: HxW)
      r   : Radius of the box filter (window size)
      eps : Regularization parameter
      
    Returns:
      q   : Filtered image (np.float32, shape: HxW)
    """
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))
    
    q = mean_a * I + mean_b
    return q

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

def plot_correction_mesh_and_depth_image(sparse_points, dense_depth):
    # Extract coordinates and sensor depths from the sparse_points array
    u_coords = np.round(sparse_points[0, :]).astype(int)  # x coordinates (columns)
    v_coords = np.round(sparse_points[1, :]).astype(int)  # y coordinates (rows)
    sensor_depths = sparse_points[2, :]
    
    # Get the dense depth values at the sparse point locations
    dense_at_points = dense_depth[v_coords, u_coords]
    
    # Compute corrections (difference between sensor and dense depth)
    corrections = sensor_depths - dense_at_points

    # Create a triangulation over the (u,v) points
    triang = Triangulation(u_coords, v_coords)

    # Create a figure with two subplots: one for the 3D mesh and one for the depth image
    fig = plt.figure(figsize=(12, 6))
    
    # --- Plot 3D Triangular Mesh ---
    ax1 = fig.add_subplot(121, projection='3d')
    # Plot a trisurf using the sparse point coordinates and their correction values
    ax1.plot_trisurf(u_coords, v_coords, corrections,
                     triangles=triang.triangles, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('u (pixel)')
    ax1.set_ylabel('v (pixel)')
    ax1.set_zlabel('Correction (m)')
    ax1.set_title('Triangular Mesh of Correction')

    # --- Interpolate onto a Grid to Create a Depth Image ---
    H, W = dense_depth.shape
    grid_u, grid_v = np.meshgrid(np.arange(W), np.arange(H))
    # Interpolate the corrections onto the grid using linear interpolation.
    grid_corrections = griddata((u_coords, v_coords), corrections,
                                (grid_u, grid_v), method='linear')
    # Some grid points may fall outside the convex hull of the data; fill these with nearest-neighbor interpolation.
    nan_mask = np.isnan(grid_corrections)
    if np.any(nan_mask):
        grid_corrections[nan_mask] = griddata((u_coords, v_coords), corrections,
                                              (grid_u, grid_v), method='nearest')[nan_mask]

    # --- Plot the Depth Image ---
    ax2 = fig.add_subplot(122)
    im = ax2.imshow((grid_corrections), cmap='viridis', origin='upper')
    ax2.set_title('Interpolated Correction Depth Image')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    #plt.show()
    return grid_corrections

import numpy as np

def compare_depth_images(depth1, depth2):
    """
    Compare two depth images and compute difference metrics.

    Parameters:
      depth1: np.array of shape (H, W)
              Depth image in meters.
      depth2: np.array of shape (H, W)
              Depth image in meters.

    Returns:
      A dictionary containing:
        - 'median_diff_m': Median of the absolute difference (meters).
        - 'std_diff_m': Standard deviation of the absolute difference (meters).
        - 'median_percent_diff': Median percent difference, computed as
             100 * |depth1 - depth2| / ((depth1 + depth2) / 2)
    """
    # Check that the images have the same shape.
    if depth1.shape != depth2.shape:
        raise ValueError("Depth images must have the same shape")
    
    # Compute the absolute difference.
    clip_value=600
    depth1[depth1 > clip_value] = clip_value
    depth2[depth2 > clip_value] = clip_value
    diff = depth1 - depth2
    abs_diff = np.abs(diff)
    
    # Calculate median and standard deviation of the differences in meters.
    median_diff_m = np.median(abs_diff)
    std_diff_m = np.std(abs_diff)
    
    # Compute percent difference relative to the average depth of the two images.
    avg_depth = (depth1 + depth2) / 2.0
    # Avoid division by zero using a small epsilon.
    epsilon = 1e-6
    avg_depth_safe = np.maximum(avg_depth, epsilon)
    percent_diff = 100 * abs_diff / avg_depth_safe
    median_percent_diff = np.median(percent_diff)
    
    return {
        'median_diff_m': median_diff_m,
        'std_diff_m': std_diff_m,
        'median_percent_diff': median_percent_diff
    }


# ====== Main Process ======
def main():
    # ====== Configuration ======
    frame_num = 200                    # Update as needed
    frame_str = f"{frame_num:05d}"       # e.g., 100 -> "00100"
    use_saved_values = False             # Toggle to run model inference or load saved data

    # ====== Run ORB-SLAM to get keypoint data ======
    temp_save_file = "temp_save2.npz"


    if not use_saved_values:
        pose_list, points_list, points_2d = run_orb_slam()
        rgb_path = os.path.join(pathDataset, "data", f"{frame_str}.png")
        pred_depth, rgb_img = run_depth_pro([rgb_path])
        
        with open("orb_slam_data.pkl", "wb") as f:
            pickle.dump({"p_depths": pred_depth, 
                        "rgb_imgs": rgb_img, 
                        "pose_list": pose_list, 
                        "points_list": points_list, 
                        "points_2d": points_2d}, f)
    else:
        with open("orb_slam_data.pkl", "rb") as f:
            data = pickle.load(f)
        
        pred_depth = data["p_depths"]
        rgb_img = data["rgb_imgs"]
        pose_list = data["pose_list"]
        points_list = data["points_list"]
        points_2d = data["points_2d"]

    
    print("go all data")


    # ====== Load the true depth image ======
    true_depth_path = os.path.join(path_depth, f"{frame_str}.png")
    t_depth = np.array(Image.open(true_depth_path).getdata()).reshape(pred_depth[0].shape) / 100

    # ====== Compute Keypoint Percentage Error ======
    # points_2d[frame_num] is (3, N): rows [u, v, predicted_depth]
    scaling_factor, inlier_orb_points = predict_orb_scale(points_2d[frame_num], pred_depth[0])
    points_uvd = points_2d[frame_num] #*50#* scaling_factor
    points_uvd[2,:] = points_uvd[2, :] * scaling_factor
    inlier_orb_points[2, :] = inlier_orb_points[2, :] * scaling_factor
    print("scaling_factor: ", scaling_factor)
    
    #=====rectify!!=====
    rec_pred_depth=plot_correction_mesh_and_depth_image(inlier_orb_points, pred_depth[0])
    rec_pred_depth=rec_pred_depth+pred_depth[0]

    # test
    # u_coords = np.round(inlier_orb_points[0, :]).astype(int)
    # v_coords = np.round(inlier_orb_points[1, :]).astype(int)
    # orb_depth = inlier_orb_points[2, :]
    # deep_matching_orb = rec_pred_depth[v_coords, u_coords]
    # #t_matching_orb = t_depth[v_coords, u_coords]
    # kp_errors = np.abs(deep_matching_orb-orb_depth)
    # print(kp_errors)

    #kp_errors, kp_percentage = compute_keypoint_errors(t_depth, deep_matching_orb)
    # full_errors = compute_full_depth_errors(t_depth, pred_depth[0])
    # rec_full_errors = compute_full_depth_errors(t_depth, rec_pred_depth)
    # bins, kp_errors_flat, full_errors_flat = prepare_histogram_bins(kp_errors, full_errors)
    # plot_error_histograms(kp_errors_flat, "1", full_errors_flat, "2", bins)


    u_coords = np.round(inlier_orb_points[0, :]).astype(int)
    v_coords = np.round(inlier_orb_points[1, :]).astype(int)

    rec_full_errors = compute_full_depth_errors(t_depth, rec_pred_depth)
    full_errors = compute_full_depth_errors(t_depth, pred_depth[0])
    bins, rec_errors_flat, full_errors_flat = prepare_histogram_bins(rec_full_errors, full_errors)
    plot_error_histograms(rec_errors_flat, "rec", full_errors_flat, "default", bins)

    dist_to_correct_default = np.abs(t_depth-pred_depth[0])
    dist_to_correct_rectified = np.abs(t_depth-rec_pred_depth)
    # Compute improvement per pixel:
    #   improvement > 0: rectified error is lower (better)
    #   improvement < 0: rectified error is higher (worse)
    improvement = dist_to_correct_default - dist_to_correct_rectified

    # Set up the plot
    plt.figure(figsize=(8, 6))
    # Use a diverging colormap (blue-white-red) so that zero is centered.
    # The vmin and vmax are set symmetrically around zero.
    max_abs_val = np.max(np.abs(improvement))
    im = plt.imshow(np.log(improvement), cmap='bwr')#, vmin=-max_abs_val, vmax=max_abs_val)
    plt.colorbar(im, label='Improvement (m)\n(positive: rectified is better)')
    plt.title('Per-pixel Improvement: Rectified vs Default Depth Map')
    plt.xlabel('Pixel Column')
    plt.ylabel('Pixel Row')
    plt.scatter(u_coords, v_coords, c='red', s=30, marker='o', label='Inlier ORB Points')

    plt.show()

    print("error of unchanged depth prediction")
    print(compare_depth_images(t_depth, pred_depth[0]))

    print("error of rectified depth")
    print(compare_depth_images(t_depth, rec_pred_depth))

    # Example usage:
# depth_img1 = np.array([[1.0, 2.0], [3.0, 4.0]])
# depth_img2 = np.array([[1.1, 1.9], [3.1, 3.8]])
# result = compare_depth_images(depth_img1, depth_img2)
# print(result)

    # # Segment the depth image into regions
    # depth = pred_depth[0]
    # labels = segment_depth_image(np.log(depth), n_clusters=6)
    # #labels = segment_depth_image(depth, n_clusters=6)
    
    # # Colorize the segmentation for visualization
    # colored_segments = colorize_labels(labels)
    
    # # Plot the original depth image and the segmented color map
    # plt.figure(figsize=(12, 5))
    
    # plt.subplot(1, 2, 1)
    # plt.title("Depth Image")
    # plt.imshow(np.squeeze(rgb_img), cmap='gray')
    # plt.colorbar()
    
    # plt.subplot(1, 2, 2)
    # plt.title("Segmented Surfaces")
    # plt.imshow(colored_segments)
    
    # plt.tight_layout()
    # plt.show()

    # # ====== Compute Keypoint Percentage Error ======
    # # points_2d[frame_num] is (3, N): rows [u, v, predicted_depth]
    # scaling_factor = predict_orb_scale(points_2d[frame_num], pred_depth[0])
    # points_uvd = points_2d[frame_num] #*50#* scaling_factor
    # points_uvd[2,:] = points_uvd[2, :] * scaling_factor
    # print("scaling_factor: ", scaling_factor)
    # kp_errors, kp_percentage = compute_keypoint_errors(t_depth, points_uvd)

    # # ====== Compute Full Depth Map Absolute Error ======
    # full_errors = compute_full_depth_errors(t_depth, pred_depth[0])
    # #rec_full_errors = compute_full_depth_errors(t_depth, rec_pred_depth)

    # # ====== Prepare Histogram Data & Plot Histograms ======
    # bins, kp_errors_flat, full_errors_flat = prepare_histogram_bins(kp_errors, full_errors)
    # plot_error_histograms(kp_errors_flat, full_errors_flat, bins)

    # # bins, rf_errors_flat, full_errors_flat = prepare_histogram_bins(rec_full_errors, full_errors)
    # # plot_error_histograms(rf_errors_flat, full_errors_flat, bins)

    print("done")

if __name__ == "__main__":
    main()

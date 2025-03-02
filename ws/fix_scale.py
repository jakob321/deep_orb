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
        final_result = orbslam3.run_orb_slam3(voc_file, settings_file, pathDataset, timestamps, fps=100)

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

def plot_error_histograms(errors1, errors2, bins):
    """
    Plot a combined histogram comparing keypoint and full depth errors.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(errors1, bins=bins, density=True, alpha=0.5,
             label="Keypoint Absolute Error", color='blue', edgecolor='black')
    plt.hist(errors2, bins=bins, density=True, alpha=0.5,
             label="Depth Map Absolute Error", color='red', edgecolor='black')
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

    # The scaling factor is the median of these ratios
    scaling_factor = np.median(scales)

    return scaling_factor

# ====== Main Process ======
def main():
    # ====== Configuration ======
    frame_num = 50                     # Update as needed
    frame_str = f"{frame_num:05d}"       # e.g., 100 -> "00100"
    use_saved_values = True             # Toggle to run model inference or load saved data

    # ====== Run ORB-SLAM to get keypoint data ======
    pose_list, points_list, points_2d = run_orb_slam()

    # ====== Load or compute predicted depth and corresponding RGB image ======
    rgb_path = os.path.join(pathDataset, "data", f"{frame_str}.png")
    print("RGB image path:", rgb_path)
    temp_save_file = "temp_save2.npz"
    if not use_saved_values:
        pred_depth, rgb_img = run_depth_pro([rgb_path])
        np.savez(temp_save_file, p_depths=pred_depth, rgb_imgs=rgb_img)
    else:
        all_data = np.load(temp_save_file, allow_pickle=True)
        pred_depth, rgb_img = all_data["p_depths"], all_data["rgb_imgs"]

    # ====== Load the true depth image ======
    true_depth_path = os.path.join(path_depth, f"{frame_str}.png")
    t_depth = np.array(Image.open(true_depth_path).getdata()).reshape(pred_depth[0].shape) / 100

    # ====== Compute Keypoint Percentage Error ======
    # points_2d[frame_num] is (3, N): rows [u, v, predicted_depth]
    scaling_factor = predict_orb_scale(points_2d[frame_num], pred_depth[0])
    points_uvd = points_2d[frame_num] #*50#* scaling_factor
    points_uvd[2,:] = points_uvd[2, :] * scaling_factor
    print("scaling_factor: ", scaling_factor)
    kp_errors, kp_percentage = compute_keypoint_errors(t_depth, points_uvd)

    # ====== Compute Full Depth Map Absolute Error ======
    full_errors = compute_full_depth_errors(t_depth, pred_depth[0])

    # ====== Prepare Histogram Data & Plot Histograms ======
    bins, kp_errors_flat, full_errors_flat = prepare_histogram_bins(kp_errors, full_errors)
    plot_error_histograms(kp_errors_flat, full_errors_flat, bins)

    print("done")

if __name__ == "__main__":
    main()

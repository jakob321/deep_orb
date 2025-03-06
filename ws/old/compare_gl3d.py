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

path_depth="../datasets/gl3d/gl3d_rendered_depth/data/000000000000000000000006/rendered_depths"
pathDataset = "../datasets/gl3d/gl3d_img/data/000000000000000000000007/undist_images"
#pathDataset = "../datasets/kitti/2011_09_26_drive_0093_sync/image_02"
voc_file = "../ORB_SLAM3/Vocabulary/ORBvoc.txt"
settings_file = "../ORB_SLAM3/Examples/Monocular/KITTI03.yaml"
timestamps = "../ORB_SLAM3/Examples/Monocular/EuRoC_TimeStamps/MH01.txt"

fx = 721.5377
fy = 721.5377
cx = 609.5593
cy = 172.8540


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


def run_orb_slam():
    def run_slam():
        global final_result
        final_result = orbslam3.run_orb_slam3(voc_file, settings_file, pathDataset, timestamps, fps=100)

    slam_thread = threading.Thread(target=run_slam)
    slam_thread.start()

    pose_list, points_list, points_2d = (0,0, 0)
    time.sleep(0)
    while slam_thread.is_alive():
        pose_list, points_list = orbslam3.get_all_data_np()
        points_2d = orbslam3.get_2d_points()
        time.sleep(0.01)
    return pose_list, points_list, points_2d[0]

def load_depth_pro():
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device)
    model.eval()
    return model, device

def run_depth_pro(paths):
    # Model init
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device)
    model.eval()

    all_depth=[]
    all_orig=[]
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
    all_data=np.load("saved_data.npz", allow_pickle=True)
    return all_data["pose"], all_data["depth"], all_data["orig"], all_data["points"]


if __name__ == "__main__":

    # ====== Configuration (update these once) ======
    frame_num = 100                     # Change this number to update all paths (will be 5-digit formatted)
    frame_str = f"{frame_num:05d}"       # e.g. 200 becomes "00200"
    # path_depth = "path/to/depth"         # Folder where ground truth depth images are stored
    # pathDataset = "path/to/dataset"      # Folder where RGB/depth images are stored

    # ====== Data Loading ======
    use_saved_values = True

    # Run ORB-SLAM (or load saved data) to get points_2d
    # (points_2d is assumed to be a list-like object where each element is a (3, N) array)
    pose_list, points_list, points_2d = run_orb_slam()
def abc():

    # Load predicted depth and RGB image from a saved file
    rgb_path=pathDataset+"/data/"+frame_str+".png"
    #rgb_path=pathDataset+"/data/00200.png"
    print(rgb_path)

    if not use_saved_values:
        pred_depth, rgb_img = run_depth_pro([rgb_path])
        np.savez("temp_save2.npz", p_depths=pred_depth, rgb_imgs=rgb_img)
    all_data = np.load("temp_save2.npz", allow_pickle=True)
    pred_depth, rgb_img = all_data["p_depths"], all_data["rgb_imgs"]

    # Load the true depth image for the current frame (assumes 5-digit filenames)
    true_depth_path = f"{path_depth}/{frame_str}.png"
    t_depth = np.array(Image.open(true_depth_path).getdata()).reshape(pred_depth[0].shape) / 100

    # ====== Compute Keypoint Percentage Error ======
    # Get keypoint data for the selected frame (points_2d[frame_num] is (3, N))
    points_uvd = points_2d[frame_num]  # rows: 0 = u, 1 = v, 2 = predicted depth (from ORB-SLAM)
    u_coords = np.round(points_uvd[0, :]).astype(int)
    v_coords = np.round(points_uvd[1, :]).astype(int)
    # Apply a scaling factor as in your code (adjust as needed)
    pred_depths = points_uvd[2, :] * 50

    # Ensure keypoint coordinates fall within the image bounds
    valid_idx = (u_coords >= 0) & (u_coords < t_depth.shape[1]) & (v_coords >= 0) & (v_coords < t_depth.shape[0])
    u_coords = u_coords[valid_idx]
    v_coords = v_coords[valid_idx]
    pred_depths = pred_depths[valid_idx]

    # Get the true depth at each keypoint location
    true_depths = t_depth[v_coords, u_coords]

    # Compute absolute error and then percentage error: (|pred - true| / true)*100.
    # Avoid division by zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        keypoint_errors = np.abs(pred_depths - true_depths)
        keypoint_percentage = np.where(true_depths != 0, (keypoint_errors / true_depths) * 100, 0)

    # ====== Compute Full Depth Map Absolute Error ======
    p_depth = pred_depth[0].copy()
    # Optionally clip the predicted depth map if needed
    p_depth[p_depth > 655.35] = 655.35
    full_errors = np.abs(t_depth - p_depth)
    plt.imshow(full_errors, cmap='hot')

    # ====== Prepare Histogram Data ======
    # Flatten errors; remove zero (or non-positive) values for log scaling
    keypoint_errors_flat = keypoint_errors.ravel()
    full_errors_flat = full_errors.ravel()
    keypoint_errors_flat = keypoint_errors_flat[keypoint_errors_flat > 0]
    full_errors_flat = full_errors_flat[full_errors_flat > 0]

    # Determine common logarithmic bins for both histograms
    epsilon = 1e-8
    min_val = np.minimum(keypoint_errors_flat.min(), full_errors_flat.min())
    max_val = np.maximum(keypoint_errors_flat.max(), full_errors_flat.max())
    log_bins = np.logspace(np.log10(min_val + epsilon), np.log10(max_val + epsilon), 50)

    # ====== Combined Histogram Plot ======
    plt.figure(figsize=(8, 6))
    plt.hist(keypoint_errors_flat, bins=log_bins, density=True, alpha=0.5,
            label="Keypoint Absolute Error", color='blue', edgecolor='black')
    plt.hist(full_errors_flat, bins=log_bins, density=True, alpha=0.5,
            label="Depth Map Absolute Error", color='red', edgecolor='black')

    plt.xscale('log')
    plt.xlabel("Absolute Error (meters, log scale)")
    plt.ylabel("Normalized Frequency")
    plt.title("Comparison of Depth Absolute Errors")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

    print("done")

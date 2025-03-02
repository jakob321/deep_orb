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

path_depth="../datasets/vkitti/vkitti_1.3.1_depthgt/0002/clone"
pathDataset = "../datasets/vkitti/vkitti_1.3.1_rgb/0002/clone"
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

#import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np



import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    use_saved_values = True
    pose_list, points_list = (0, 0)

    # If not using saved values, run ORB-SLAM to get data
    #if not use_saved_values:
    pose_list, points_list, points_2d = run_orb_slam()

    print("2d_point shape: ", points_2d[0].shape)  # (3, N), where N is number of keypoints

    true_depth = [
        path_depth + "/00200.png",
    ]
    frame_idx = 200

    paths=[
        pathDataset+"/data/00200.png",
    ]


    if not use_saved_values:
        pred_depth, rgb_img = run_depth_pro(paths)
        np.savez("temp_save2.npz", p_depths=pred_depth, rgb_imgs=rgb_img)


    all_data = np.load("temp_save2.npz", allow_pickle=True)
    pred_depth, rgb_img = all_data["p_depths"], all_data["rgb_imgs"]
    t_depth = np.array(Image.open(path_depth + "/00200.png").getdata()).reshape(pred_depth[0].shape) / 100

    # Extract u, v, depth from points_2d
     # Select the first frame (change as needed)
    points_uvd = points_2d[frame_idx]  # Shape (3, N)
    u_coords = np.round(points_uvd[0, :]).astype(int)  # Round u to integer pixel coordinates
    v_coords = np.round(points_uvd[1, :]).astype(int)  # Round v to integer pixel coordinates
    pred_depths = points_uvd[2, :]*50  # Predicted depth values from ORB-SLAM3

    # Ensure points are within image bounds
    valid_idx = (u_coords >= 0) & (u_coords < t_depth.shape[1]) & (v_coords >= 0) & (v_coords < t_depth.shape[0])
    u_coords = u_coords[valid_idx]
    v_coords = v_coords[valid_idx]
    pred_depths = pred_depths[valid_idx]

    # Get ground truth depth from t_depth at the corresponding (u, v) locations
    true_depths = t_depth[v_coords, u_coords]

    # Compute depth differences (errors)
    depth_errors = np.abs(pred_depths - true_depths)

    # Plot histogram of depth differences
    plt.figure(figsize=(8, 6))
    plt.hist(depth_errors, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel("Depth Difference (Predicted - Ground Truth)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Depth Differences")
    plt.grid(True)
    print(points_2d[2])

    points_uv = points_2d[frame_idx][:2, :]  # Only take u, v (ignore depth)

    # Display depth image
    plt.figure(figsize=(10, 6))
    plt.imshow(t_depth, cmap="jet")  # Use 'jet' colormap for better contrast

    # Overlay keypoints
    plt.scatter(points_uv[0, :], points_uv[1, :], c="red", s=10, label="Keypoints")

    for i in range(len(u_coords)-10):
        plt.text(
            u_coords[i], v_coords[i] - 5,  # Offset text slightly above the point
            f"Pred: {pred_depths[i]:.2f}\nTrue: {true_depths[i]:.2f}",
            fontsize=8, color="white", ha="center", va="bottom", bbox=dict(facecolor="black", alpha=0.5, edgecolor="none")
        )

    # Show the plot
    plt.legend()
    plt.colorbar(label="Depth Value")
    plt.title("Depth Image with Keypoints Overlay")
    #plt.show()

    # Show the histogram
    #######################################################################

    print(t_depth.shape)
    p_depth = pred_depth[0]
    p_depth[p_depth > 655.35] = 655.35

    # Compute absolute difference
    diff = np.abs(t_depth - p_depth)
    print(np.max(t_depth))
    print(np.max(p_depth))
    print(diff)

    # Create a figure with 4 subplots
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(t_depth, cmap='viridis')
    plt.title('True Depth')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(p_depth, cmap='viridis')
    plt.title('Predicted Depth')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(np.log10(diff), cmap='hot')
    plt.title('Difference')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    epsilon = 1e-8
    log_bins = np.logspace(np.log10(np.maximum(diff[diff>0].min(), epsilon)), np.log10(diff.max()), 50)
    plt.hist(diff.ravel(), bins=log_bins)
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('Difference (distance)')
    plt.ylabel('Number of pixels')
    plt.title('Histogram of differences')
    
    plt.tight_layout()


    epsilon = 1e-8
    # Compute the minimum (positive) value across both datasets
    min_val = np.maximum(
        np.min(depth_errors[depth_errors > 0]) if np.any(depth_errors > 0) else epsilon,
        np.min(diff[diff > 0]) if np.any(diff > 0) else epsilon
    )
    max_val = np.max([np.max(depth_errors), np.max(diff)])

    # Create logarithmically spaced bins from min_val to max_val
    log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)

    plt.figure(figsize=(8, 6))
    # Overlay histogram of keypoint depth errors
    plt.hist(depth_errors, bins=log_bins, density=True, alpha=0.5,
            label="Keypoint Depth Error", color='blue', edgecolor='black')
    # Overlay histogram of full depth map differences
    plt.hist(diff.ravel(), bins=log_bins, density=True, alpha=0.5,
            label="Depth Map Error", color='red', edgecolor='black')

    plt.xscale('log')
    plt.xlabel("Difference (distance)")
    plt.ylabel("Normalized Frequency")
    plt.title("Comparison of Depth Differences")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    plt.show()




def nothing():

    paths=[
        pathDataset+"/data/00000.png",
        pathDataset+"/data/00020.png",
        pathDataset+"/data/00040.png",
        pathDataset+"/data/00060.png"
    ]

    true_depth=[
        path_depth+"/00000.png",
        path_depth+"/00020.png",
        path_depth+"/00040.png",
        path_depth+"/00060.png",
    ]


    if not use_saved_values:
        pred_depth, rgb_img = run_depth_pro(paths)
        np.savez("temp_save.npz", p_depths=pred_depth, rgb_imgs=rgb_img)
    else:
        all_data=np.load("temp_save.npz", allow_pickle=True)
        pred_depth, rgb_img = all_data["p_depths"], all_data["rgb_imgs"]

    print(pred_depth[0].shape)

    for i in range(len(paths)):
        # Load true depth and predicted depth images
        t_depth = np.array(Image.open(true_depth[i]).getdata()).reshape(pred_depth[0].shape)/100
        print(t_depth.shape)
        p_depth = pred_depth[i]
        p_depth[p_depth > 655.35] = 655.35

        # Compute absolute difference
        diff = np.abs(t_depth - p_depth)
        print(np.max(t_depth))
        print(np.max(p_depth))
        print(diff)

        # Create a figure with 4 subplots
        plt.figure(figsize=(16, 4))
        
        plt.subplot(1, 4, 1)
        plt.imshow(t_depth, cmap='viridis')
        plt.title('True Depth')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(p_depth, cmap='viridis')
        plt.title('Predicted Depth')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(np.log10(diff), cmap='hot')
        plt.imshow(t_depth, cmap='hot')
        plt.imshow(p_depth, cmap='hot')
        plt.title('Difference')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        epsilon = 1e-8
        log_bins = np.logspace(np.log10(np.maximum(diff[diff>0].min(), epsilon)), np.log10(diff.max()), 50)
        plt.hist(diff.ravel(), bins=log_bins)
        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel('Difference (distance)')
        plt.ylabel('Number of pixels')
        plt.title('Histogram of differences')
        
        plt.tight_layout()
        plt.show()
        #break


    print("done")

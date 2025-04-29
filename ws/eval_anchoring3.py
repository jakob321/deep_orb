from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
from helper import generic_helper
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter

# def create_correction_map(p_depth, orb_points_2d, rgb_image):
#     # Extract coordinates and depths
#     orb_u = orb_points_2d[0].astype(int)
#     orb_v = orb_points_2d[1].astype(int)
#     orb_depths = orb_points_2d[2]
#     height, width = p_depth.shape
    
#     # Get eroded depth to make measurements more robust
#     p_depth_eroded = cv2.erode(p_depth, np.ones((5, 5), np.uint8))
#     depth_at_orb_points = p_depth_eroded[orb_v, orb_u]
    
#     # Calculate depth differences at ORB points
#     depth_diff = orb_depths - depth_at_orb_points
    
#     # Get sky mask
#     sky_mask = generic_helper.get_sky_mask(rgb_image)
    
#     # Filter out ORB points that are in the sky or have large corrections
#     valid_u = []
#     valid_v = []
#     valid_diffs = []
    
#     for u, v, diff in zip(orb_u, orb_v, depth_diff):
#         if v < height and u < width and not sky_mask[v, u] and abs(diff) <= 10.0:
#             valid_u.append(u)
#             valid_v.append(v)
#             valid_diffs.append(diff)
    
#     # If no valid points, return zeros
#     if len(valid_u) == 0:
#         return np.zeros((height, width)), np.zeros((height, width))
    

    
#     return correction_map, confidence_map

import numpy as np
import cv2
from scipy.interpolate import Rbf

def create_correction_map1(p_depth, orb_points_2d, rgb_image):
    # Extract coordinates and depths
    orb_u = orb_points_2d[0].astype(int)
    orb_v = orb_points_2d[1].astype(int)
    orb_depths = orb_points_2d[2]
    height, width = p_depth.shape

    # Get eroded depth to make measurements more robust
    p_depth_eroded = cv2.erode(p_depth, np.ones((5, 5), np.uint8))
    depth_at_orb_points = p_depth_eroded[orb_v, orb_u]

    # Calculate depth differences at ORB points
    depth_diff = orb_depths - depth_at_orb_points

    # Get sky mask
    sky_mask = generic_helper.get_sky_mask(rgb_image)

    # Filter out ORB points that are in the sky or have large corrections
    valid_u = []
    valid_v = []
    valid_diffs = []

    for u, v, diff in zip(orb_u, orb_v, depth_diff):
        if v < height and u < width and not sky_mask[v, u] and abs(diff) <= 10.0:
            valid_u.append(u)
            valid_v.append(v)
            valid_diffs.append(diff)

    # If no valid points, return zeros
    if len(valid_u) == 0:
        return np.zeros((height, width)), np.zeros((height, width))

    # MVP: Use RBF interpolation to smoothly spread the correction over the image
    valid_u = np.array(valid_u)
    valid_v = np.array(valid_v)
    valid_diffs = np.array(valid_diffs)

    # Create RBF interpolator
    rbf = Rbf(valid_u, valid_v, valid_diffs, function='linear', smooth=1.0)

    # Create a grid of coordinates
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Evaluate correction map on the full image grid
    correction_map = rbf(grid_x, grid_y)

    # Compute corrected map
    corrected_map = p_depth + correction_map

    return correction_map, corrected_map

import numpy as np
import cv2
from scipy.interpolate import Rbf

def create_correction_map2(p_depth, orb_points_2d, rgb_image):
    # Extract coordinates and depths
    orb_u = orb_points_2d[0].astype(int)
    orb_v = orb_points_2d[1].astype(int)
    orb_depths = orb_points_2d[2]
    height, width = p_depth.shape

    # Get eroded depth to make measurements more robust
    p_depth_eroded = cv2.erode(p_depth, np.ones((5, 5), np.uint8))
    depth_at_orb_points = p_depth_eroded[orb_v, orb_u]

    # Calculate depth differences at ORB points
    depth_diff = orb_depths - depth_at_orb_points

    # Get sky mask
    sky_mask = generic_helper.get_sky_mask(rgb_image)

    # Filter out ORB points that are in the sky or have large corrections
    valid_u, valid_v, valid_diffs = [], [], []
    for u, v, diff in zip(orb_u, orb_v, depth_diff):
        if (0 <= v < height and 0 <= u < width
            and not sky_mask[v, u]
            and abs(diff) <= 10.0):
            valid_u.append(u)
            valid_v.append(v)
            valid_diffs.append(diff)

    # If no valid points, return zeros
    if len(valid_u) == 0:
        return np.zeros((height, width)), np.zeros((height, width))

    # Convert to arrays
    valid_u = np.array(valid_u)
    valid_v = np.array(valid_v)
    valid_diffs = np.array(valid_diffs)

    # --- LIMIT INFLUENCE: Gaussian RBF ---
    # Define how far (in pixels) each point should noticeably influence
    influence_radius = 100.0
    epsilon = 1.0 / influence_radius

    # Build RBF interpolator with Gaussian kernel
    rbf = Rbf(
        valid_u, valid_v, valid_diffs,
        function='gaussian',
        # epsilon=epsilon,
        smooth=1.0
    )

    # rbf = Rbf(valid_u, valid_v, valid_diffs, function='linear', smooth=1.0)

    # Create full-image grid
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute correction map
    correction_map = rbf(grid_x, grid_y)

    # Apply to original depth
    corrected_map = p_depth + correction_map

    return correction_map, corrected_map


import numpy as np
import cv2
from scipy.interpolate import Rbf

import numpy as np
import cv2
from scipy.interpolate import Rbf

def create_correction_map(p_depth, orb_points_2d, rgb_image):
    # Extract ORB coords & depths
    orb_u = orb_points_2d[0].astype(int)
    orb_v = orb_points_2d[1].astype(int)
    orb_depths = orb_points_2d[2]
    height, width = p_depth.shape

    # Robustify sensor depth
    p_depth_eroded = cv2.erode(p_depth, np.ones((5, 5), np.uint8))
    depth_at_orb = p_depth_eroded[orb_v, orb_u]

    # Compute residuals
    depth_diff = orb_depths - depth_at_orb

    # Sky mask & filter
    sky_mask = generic_helper.get_sky_mask(rgb_image)
    valid_u, valid_v, valid_diffs = [], [], []
    for u, v, d in zip(orb_u, orb_v, depth_diff):
        if (0 <= v < height and 0 <= u < width
            and not sky_mask[v, u]
            and abs(d) <= 10.0):
            valid_u.append(u)
            valid_v.append(v)
            valid_diffs.append(d)

    if not valid_u:
        return np.zeros_like(p_depth), np.zeros_like(p_depth)

    valid_u = np.array(valid_u)
    valid_v = np.array(valid_v)
    valid_diffs = np.array(valid_diffs)

    # 1) Linear RBF on a coarse grid, then upsample to full resolution
    rbf = Rbf(valid_u, valid_v, valid_diffs, function='linear', smooth=1.0)

    # Downscale factor
    downscale = 4
    coarse_h = max(2, height // downscale)
    coarse_w = max(2, width  // downscale)

    # Create coarse-grid coordinates in original pixel space
    coarse_x = np.linspace(0, width - 1, coarse_w)
    coarse_y = np.linspace(0, height - 1, coarse_h)
    cx, cy   = np.meshgrid(coarse_x, coarse_y)

    # Evaluate RBF on the small grid
    coarse_corr = rbf(cx, cy)

    # Upsample back to full resolution
    # Note: cv2.resize takes (width, height)
    global_correction = cv2.resize(
        coarse_corr.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    # 2) Build a confidence mask of Gaussians via union (local-windowed)
    influence_radius = 70.0
    sigma = influence_radius
    radius = int(3 * sigma)

    # Precompute Gaussian kernel patch
    ax = np.arange(-radius, radius+1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))

    mask = np.zeros_like(global_correction, dtype=float)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    for u, v in zip(valid_u, valid_v):
        # Window bounds
        x0 = max(u - radius, 0)
        x1 = min(u + radius + 1, width)
        y0 = max(v - radius, 0)
        y1 = min(v + radius + 1, height)

        # Kernel slice
        kx0 = x0 - (u - radius)
        kx1 = kx0 + (x1 - x0)
        ky0 = y0 - (v - radius)
        ky1 = ky0 + (y1 - y0)

        sub = mask[y0:y1, x0:x1]
        gsub = kernel[ky0:ky1, kx0:kx1]
        mask[y0:y1, x0:x1] = 1 - (1 - sub) * (1 - gsub)

    # 3) Localize the correction
    weighted_correction = global_correction * mask

    # 4) Apply it
    corrected_map = p_depth + weighted_correction

    return weighted_correction, corrected_map



def main():
    dataset = vkitti.dataset("midair", environment="spring")
    seq = 1
    dataset.set_sequence(seq)

    depth = deep.DepthModelWrapper(model_name="depth_pro", load_weights=False)
    pose_list, points_list, points_2d = orb.run_if_no_saved_values(
        dataset, override_run=False
    )
    orb_true_scale, pose_list = orb.compute_true_scale(
        pose_list, dataset.load_extrinsic_matrices()
    )
    print(orb_true_scale)

    number_of_deep_frames = 10
    deep_frames_index = np.linspace(
        1, len(points_2d) - 1, number_of_deep_frames, dtype=int
    ).tolist()
    rgb_path = [dataset.get_rgb_frame_path()[i] for i in deep_frames_index]
    pred_depth, rgb_img = depth.process_images(rgb_path, caching=True)

    for index, frame in enumerate(deep_frames_index):
        t_depth = dataset.get_depth_frame_np(frame)
        p_depth = pred_depth[index]
        color_img = rgb_img[index]

        # Get corresponding ORB points for this frame
        orb_points_3d = points_list[frame]  # 3D points (3, n)
        orb_points_2d = points_2d[frame]  # 2D points with depth (u, v, depth)
        orb_points_2d[2] = orb_points_2d[2] * orb_true_scale

        if len(orb_points_2d[0]) == 0:
            print(f"No valid ORB points for frame {frame}")
            continue

        p_depth = generic_helper.scale_predicted_depth(p_depth, orb_points_2d)

        weighted_correction, confidence_map = create_correction_map(
            p_depth, orb_points_2d, color_img
        )

        corr_depth = p_depth + weighted_correction

        # Evaluate results
        corr_depth_percentage = np.abs(t_depth - corr_depth) / (t_depth + 1e-6) * 100
        depth_diff_percentage = np.abs(t_depth - p_depth) / (t_depth + 1e-6) * 100

        # Plot results
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(depth_diff_percentage, cmap="hot", vmin=0, vmax=100)
        cbar = plt.colorbar(label="Difference (%)", fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.title("Original Depth Difference")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(corr_depth_percentage, cmap="hot", vmin=0, vmax=100)
        cbar = plt.colorbar(label="Difference (%)", fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.title("Adjusted Depth Difference")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(weighted_correction, cmap="viridis")
        cbar = plt.colorbar(label="Correction", fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.title("Weighted Correction Map")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(confidence_map, cmap="inferno")
        cbar = plt.colorbar(label="Confidence", fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.title("Confidence Map")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(t_depth, cmap="hot", vmin=0, vmax=100)
        cbar = plt.colorbar(label="Depth (m)", fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.title("True Depth")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(rgb_img[index])
        # print(orb_depths)
        # Create scatter plot with a colormap and store the scatter object
        orb_u = orb_points_2d[0].astype(int)
        orb_v = orb_points_2d[1].astype(int)
        orb_depths = orb_points_2d[2]
        scatter = plt.scatter(
            orb_u, orb_v, c=orb_depths, cmap="viridis", s=20, alpha=0.8
        )

        # Add a colorbar to show depth values
        cbar = plt.colorbar(scatter, label="Depth (m)")

        plt.title("ORB Points (After Outlier Removal)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
from helper import generic_helper

import numpy as np
import cv2
import open3d as o3d

import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import Rbf
from scipy.interpolate import SmoothBivariateSpline

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors

WIDTH = 1024
HEIGHT = 1024
fx = WIDTH / 2
fy = HEIGHT / 2
cx = WIDTH / 2
cy = HEIGHT / 2
intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)

def backproject(depth, fx, fy, cx, cy):
    """Turn an H×W depth map into an (H*W)×3 array of 3D points."""
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    z = depth.ravel()
    x = (uu.ravel() - cx) * z / fx
    y = (vv.ravel() - cy) * z / fy
    return np.vstack((x, y, z)).T  # shape (N,3)

def compute_knn_region_map(dense_depth, orb_points_2d, fx, fy, cx, cy):
    """
    Returns an H×W integer array, where each pixel is labelled by the index (0...M-1)
    of the nearest sparse point in 3D.
    """
    sparse_u = orb_points_2d[0].astype(int)
    sparse_v = orb_points_2d[1].astype(int)
    sparse_z = orb_points_2d[2]
    H, W = dense_depth.shape
    # 1) back-project
    P_d = backproject(dense_depth, fx, fy, cx, cy)         # (N,3), N=H*W
    P_s = backproject(sparse_z.reshape(-1,1), fx, fy, cx, cy)  # wrong shape: need u to z
    # Actually build P_s correctly:
    # sparse_u, sparse_v, sparse_z are 1D arrays of length M
    xs = (sparse_u - cx) * sparse_z / fx
    ys = (sparse_v  - cy) * sparse_z / fy
    zs = sparse_z
    P_s = np.vstack((xs, ys, zs)).T                       # (M,3)

    # 2) KD-tree on sparse
    kd = KDTree(P_s)

    # 3) query nearest neighbor
    _, idx = kd.query(P_d, k=1)  # idx.shape = (N,1)

    # reshape to H×W
    region_map = idx.reshape(H, W)
        # 4) compute per‐region scale factors
    #    at each sparse pt j, look up the corresponding dense depth:
    dense_at_sparse = dense_depth[sparse_v, sparse_u]   # shape (M,)
    scale_per_region = (sparse_z / dense_at_sparse)      # shape (M,)
    # scale_per_region = dense_at_sparse /sparse_z
    print(scale_per_region)

    # 5) build full‐image scale map by broadcasting:
    scale_map = scale_per_region[region_map]            # H×W

    # 6) apply to dense map
    fused_depth = dense_depth * scale_map

    return region_map, scale_map

import numpy as np
from sklearn.cluster import KMeans

def compute_kmeans_dense_clusters(dense_depth, fx, fy, cx, cy,
                                  n_clusters=50, depth_weight=1.0):
    """
    Back-projects the dense depth map to 3D, applies an optional depth bias,
    then runs K-means clustering.

    Parameters
    ----------
    dense_depth : (H, W) ndarray
        Per-pixel depth.
    fx, fy, cx, cy : floats
        Camera intrinsics.
    n_clusters : int
        Number of clusters for K-means.
    depth_weight : float
        Weight to multiply the Z‐axis by. >1 → deeper bias, <1 → lateral bias.

    Returns
    -------
    labels : (H, W) int array
        Cluster ID for each pixel.
    centers : (n_clusters, 3) array
        The (x,y,z) centroids in *weighted* space.
    """
    H, W = dense_depth.shape

    # 1) back-project all pixels
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    z = dense_depth.ravel()
    x = (uu.ravel() - cx) * z / fx
    y = (vv.ravel() - cy) * z / fy

    P = np.vstack((x, y, z)).T     # shape (H*W, 3)

    # 2) apply depth bias
    if depth_weight != 1.0:
        P *= np.array([1.0, 1.0, depth_weight])[None, :]

    # 3) run K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(P)

    # 4) reshape labels back to image
    labels = kmeans.labels_.reshape(H, W)
    return labels, kmeans.cluster_centers_


import numpy as np
from sklearn.cluster import KMeans

def compute_kmeans_with_sparse_correction(
    dense_depth, fx, fy, cx, cy,
    orb_points_2d,
    n_clusters=50, depth_weight=1.0
):
    """
    1) Back-project dense_depth to 3D (with optional depth_weight).
    2) Run K-means to get labels for every pixel.
    3) For each cluster, find all sparse points (sparse_u/v/z) that lie in it.
       Compute median(sparse_z) and median(dense_depth) → scale = med_sparse / med_dense.
    4) Build correction_map[pixel] = scale[label[pixel]].

    Returns
    -------
    labels : (H, W) int array
        Cluster ID [0..n_clusters-1] per pixel.
    correction_map : (H, W) float array
        Per-pixel scale factors to align dense → sparse medians.
    """
    sparse_u = orb_points_2d[0].astype(int)
    sparse_v = orb_points_2d[1].astype(int)
    sparse_z = orb_points_2d[2]
    H, W = dense_depth.shape

    # --- 1) Back-project dense map to 3D points ---
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    z = dense_depth.ravel()
    x = (uu.ravel() - cx) * z / fx
    y = (vv.ravel() - cy) * z / fy
    P = np.vstack((x, y, z)).T   # (H*W, 3)

    # apply depth bias if requested
    if depth_weight != 1.0:
        P *= np.array([1.0, 1.0, depth_weight])[None, :]

    # --- 2) K-means clustering on P ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(P)
    labels = kmeans.labels_.reshape(H, W)

    # --- 3) Assign sparse points to clusters and compute per-cluster scale ---
    # which cluster each sparse point falls into:
    sparse_labels = labels[sparse_v.astype(int), sparse_u.astype(int)]
    
    dense_at_sparse = dense_depth[sparse_v.astype(int), sparse_u.astype(int)]
    point_ratios   = sparse_z / dense_at_sparse

    # 4) median-of-ratios per cluster
    scales = np.ones(n_clusters, dtype=float)
    has_sparse = np.zeros(n_clusters, dtype=bool)
    for c in range(n_clusters):
        mask = (sparse_labels == c)
        if not np.any(mask):
            continue
        has_sparse[c] = True
        scales[c] = np.median(point_ratios[mask])

    
    # --- 4b) fill empty clusters from nearest neighbor in 3D ---
    # cluster_centers_ is (n_clusters, 3) in the same weighted space used for K-means
    centers = kmeans.cluster_centers_
    for c in range(n_clusters):
        if has_sparse[c]:
            continue
        # compute distance to all clusters that do have sparse points
        valid_idxs = np.where(has_sparse)[0]
        # L2 distance in the weighted 3D embedding
        dists = np.linalg.norm(centers[valid_idxs] - centers[c], axis=1)
        nearest = valid_idxs[np.argmin(dists)]
        scales[c] = scales[nearest]

    # --- 5) build correction map by broadcasting scales per label ---
    correction_map = scales[labels]

    return labels, correction_map


import numpy as np
from sklearn.neighbors import KDTree

def smooth_correction_map_3d(correction_map, dense_depth, fx, fy, cx, cy,
                             sigma, k=50):
    """
    Smooth a 2D correction_map by doing Gaussian-weighted averaging in 3D space.

    Parameters
    ----------
    correction_map : (H, W) float array
        Per-pixel scaling factors you want to smooth.
    dense_depth : (H, W) float array
        Original depth map used to back-project.
    fx, fy, cx, cy : floats
        Camera intrinsics.
    sigma : float
        Standard deviation of the Gaussian kernel in the same units as your 3D points.
    k : int
        Number of nearest neighbors to use when approximating the Gaussian.

    Returns
    -------
    smoothed : (H, W) float array, same dtype as correction_map
        The Gaussian-smoothed correction map.
    """
    H, W = dense_depth.shape
    # 1) build (N,H*W) 3D point cloud
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    z = dense_depth.ravel()
    x = (uu.ravel() - cx) * z / fx
    y = (vv.ravel() - cy) * z / fy
    P = np.vstack((x, y, z)).T  # shape (N, 3)

    # 2) flatten corrections
    corr_flat = correction_map.ravel()

    # 3) build a KD-tree on the 3D points
    tree = KDTree(P)

    # 4) for each point, find its k nearest neighbors
    dists, idxs = tree.query(P, k=k)

    # 5) compute Gaussian weights and weighted average
    #    w_ij = exp(-dists_ij^2 / (2*sigma^2))
    weights = np.exp(- (dists**2) / (2 * sigma**2))
    # weighted sum over neighbors j for each i
    numer = np.sum(weights * corr_flat[idxs], axis=1)
    denom = np.sum(weights, axis=1)
    smoothed_flat = numer / denom

    # 6) reshape and cast back to original dtype
    return smoothed_flat.reshape(H, W).astype(correction_map.dtype)

import cv2, cv2.ximgproc as xip

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

        # correction_map, precent_map = compute_knn_region_map(p_depth,
        #                             orb_points_2d, fx, fy, cx, cy)
        # corr_depth = p_depth * precent_map
        # correction_map, precent_map =compute_kmeans_dense_clusters(p_depth, fx, fy, cx, cy, 
        #                                                            n_clusters=25,
        #                                                            depth_weight=1.5)
        p_depth[p_depth > 90] = 100
        precent_map, correction_map = compute_kmeans_with_sparse_correction(
            cv2.erode(p_depth, np.ones((5, 5), np.uint8)), fx, fy, cx, cy,
            orb_points_2d,
            n_clusters=25,
            depth_weight=0.8
        )

        # correction_map=smooth_correction_map_3d(correction_map, p_depth, fx, fy, cx, cy,
        #                      sigma=0.05, k=25)
        
        # correction_map = xip.jointBilateralFilter(guide=p_depth, src=correction_map, d=-1)

        corr_depth = p_depth * correction_map

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
        plt.imshow(precent_map, cmap="viridis")
        cbar = plt.colorbar(label="Correction", fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        plt.title("Weighted Correction Map")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(correction_map, cmap="inferno")
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
        p_depth_eroded = cv2.erode(p_depth, np.ones((5, 5), np.uint8))
        # t_depth_erode = cv2.erode(t_depth, np.ones((5, 5), np.uint8))
        depth_at_orb_points = p_depth_eroded[orb_v, orb_u]
        # depth_at_orb_points = p_depth[orb_v, orb_u]
        depth_diff = (orb_depths / depth_at_orb_points)*100-100
        scatter = plt.scatter(
            orb_u, orb_v, c=depth_diff, cmap="viridis", s=40, alpha=0.8, vmin=-40
        )

        # annotate each point with its depth value
        for x, y, d in zip(orb_u, orb_v, depth_diff):
            plt.text(
                x, y,                   # position
                f"{d:.1f}",             # label (1 decimal place)
                fontsize=15,             # adjust to taste
                ha="center", va="bottom",  # centered horizontally, above the point
                color="black"           # ensure readability
            )

        # Add a colorbar to show depth values
        cbar = plt.colorbar(scatter, label="Depth (m)")

        plt.title("ORB Points (After Outlier Removal)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

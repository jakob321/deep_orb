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


import numpy as np
from sklearn.cluster import KMeans


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
        # orb_points_3d = points_list[frame]  # 3D points (3, n)
        orb_points_2d = points_2d[frame]  # 2D points with depth (u, v, depth)
        orb_points_2d[2] = orb_points_2d[2] * orb_true_scale

        if len(orb_points_2d[0]) == 0:
            print(f"No valid ORB points for frame {frame}")
            continue

        if True:
            p_depth = generic_helper.scale_predicted_depth(p_depth, orb_points_2d)

        # For creating plot with where sky is
        if False:
            p_depth = generic_helper.scale_predicted_depth(p_depth, orb_points_2d)
            sky_mask=generic_helper.get_sky_mask(color_img)

            # Copy the image to avoid modifying the original
            modified_img = color_img.copy()

            # Set sky pixels to red (255, 0, 0)
            modified_img[sky_mask] = [255, 0, 0]

            # Plot the modified image
            plt.imshow(modified_img)
            plt.axis('off')  # Optional: Hide axis
            plt.savefig("/root/fig/sky"+str(frame)+".png", bbox_inches='tight', pad_inches=0)
            plt.show()
            


        # For creating plot with ORB points
        if False:
            # plt.subplot(2, 3, 6)
            plt.imshow(np.log(cv2.erode(t_depth, np.ones((3, 3), np.uint8))))
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
                orb_u, orb_v, s=40, alpha=0.8, vmin=-40
            )

            # Add a colorbar to show depth values
            cbar = plt.colorbar(scatter, label="Depth (m)")

            plt.title("ORB Points (After Outlier Removal)")
            plt.axis("off")

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()

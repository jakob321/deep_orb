from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
from helper import generic_helper
import numpy as np
import open3d as o3d
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage


WIDTH = 1024
HEIGHT = 1024
fx = WIDTH / 2
fy = HEIGHT / 2
cx = WIDTH / 2
cy = HEIGHT / 2

if True:
    intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=WIDTH, height=HEIGHT)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)


def plot_image(rgb_img, depth_map):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rgb_plot = axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")  # Hide axes for cleaner visualization

    # Plot depth map on the right with a colormap
    depth_plot = axes[1].imshow(depth_map, cmap="plasma")
    axes[1].set_title("Depth Map")
    axes[1].axis("off")  # Hide axes for cleaner visualization

    # Add a colorbar for the depth map
    cbar = fig.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Depth")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the figure
    plt.show()

    return fig, axes


def main():
    # Environment config
    dataset = vkitti.dataset("midair", environment="spring")
    dataset.set_sequence(1)
    pose_list, points_list, points_2d = orb.run_if_no_saved_values(
        dataset, override_run=False
    )
    orb_true_scale, pose_list = orb.compute_true_scale(
        pose_list, dataset.load_extrinsic_matrices()
    )
    # dmw = deep.DepthModelWrapper(model_name="depth_anything_v2", load_weights=True)
    dmw = deep.DepthModelWrapper(model_name="depth_pro", load_weights=False)
    depth_paths = dataset.get_rgb_frame_path()

    for index in range(len(depth_paths)):
        if index % 30 != 0 or index > 2000 or index < 10:
            continue
        # if index not in [10]: continue

        pose = pose_list[index]
        pose[:3, 3] *= 1 / (50 * orb_true_scale)
        
        # Create depth prediction for current frame
        depth_path = [depth_paths[index]]
        p_depth, color_image = dmw.process_images(depth_path, caching=True)
        p_depth = p_depth[0]
        color_image = color_image[0]

        # Post processing of depth predicted frame
        orb_points_2d = points_2d[index]  # 2D points with depth (u, v, depth)
        orb_points_2d[2] = orb_points_2d[2] * orb_true_scale
        p_depth = generic_helper.scale_predicted_depth(p_depth, orb_points_2d)
        p_depth = generic_helper.create_correction_map(p_depth, orb_points_2d, color_image)
        p_depth[p_depth > 30] = 0 # remove points farther than x meters
        p_depth[generic_helper.get_sharp_gradients_mask(p_depth, threshold=0.3)] = 0
        p_depth[generic_helper.get_sky_mask(color_image)] = 0

        # Plotting in open3D
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=WIDTH,
            view_height_px=HEIGHT,
            intrinsic=intrinsic.intrinsic_matrix,
            extrinsic=np.linalg.inv(pose),
            scale=0.0005,
        )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(p_depth),
            convert_rgb_to_intensity=False,
        )
        new_points = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic
        )
        new_points.transform(pose)
        vis.add_geometry(new_points) # Add projected depth image to visualizer
        vis.add_geometry(cameraLines)  # Add the camera lines to the visualizer

    vis.run()


if __name__ == "__main__":
    main()

from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
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
fx=WIDTH/2
fy=HEIGHT/2
cx=WIDTH/2
cy=HEIGHT/2

if True:
    intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=WIDTH, height=HEIGHT)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

def plot_image(rgb_img, depth_map):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rgb_plot = axes[0].imshow(rgb_img)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')  # Hide axes for cleaner visualization
    
    # Plot depth map on the right with a colormap
    depth_plot = axes[1].imshow(depth_map, cmap='plasma')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')  # Hide axes for cleaner visualization
    
    # Add a colorbar for the depth map
    cbar = fig.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Display the figure
    plt.show()
    
    return fig, axes

def remove_sharp_gradients(depth_map, threshold=0.5):
    if depth_map.max() > 1.0:
        normalized_depth = depth_map / depth_map.max()
    else:
        normalized_depth = depth_map.copy()
    
    # Calculate gradients in x and y directions using Sobel
    grad_x = cv2.Sobel(normalized_depth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized_depth, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create a mask where gradient is below threshold (not sharp)
    mask = gradient_magnitude < threshold
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    #dilated_mask = ~cv2.dilate(~mask_uint8, kernel, iterations=1)
    return mask_uint8 > 0


def get_sky_mask(rgb_image):
    # Normalize image to 0-1 range (with clearer comment)
    normalized_rgb = rgb_image / 255.0
    
    # Convert to grayscale for gradient detection
    grayscale = cv2.cvtColor(np.float32(normalized_rgb), cv2.COLOR_RGB2GRAY)
    
    # Calculate gradient magnitude directly (combine x and y steps)
    grad_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create initial mask where gradient is below threshold (smooth areas like sky)
    threshold = 0.1
    smooth_mask = gradient_magnitude < threshold
    
    # Convert to uint8 for OpenCV operations
    mask_uint8 = smooth_mask.astype(np.uint8) * 255
    
    # Define kernels once with descriptive names
    small_kernel = np.ones((5, 5), np.uint8)
    large_kernel = np.ones((15, 15), np.uint8)
    
    # Process mask with morphological operations
    # First dilate the inverse mask (non-sky areas)
    expanded_non_sky = cv2.dilate(~mask_uint8, small_kernel, iterations=3)
    # Then dilate the inverse again to get final sky mask
    final_mask = cv2.dilate(~expanded_non_sky, large_kernel, iterations=2)
    
    # Return boolean mask
    return final_mask > 0


def main():
    # Environment config
    dataset = vkitti.dataset("midair", environment="spring")
    dataset.set_sequence(1)
    pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=True)
    orb_true_scale, pose_list = orb.compute_true_scale(pose_list, dataset.load_extrinsic_matrices())
    # dmw = deep.DepthModelWrapper(model_name="depth_anything_v2", load_weights=True)
    dmw = deep.DepthModelWrapper(model_name="depth_pro", load_weights=False)
    depth_paths = dataset.get_rgb_frame_path()

    for index in range(len(depth_paths)):
        if index%30!=0 or index>2000 or index < 10:continue
        # if index not in [10]: continue

        pose = pose_list[index]
        pose[:3, 3] *= (1/(50*orb_true_scale))
        depth_path=[depth_paths[index]]
        point_2d=points_2d[index] # is of format (x,y,d)
        print(point_2d.shape)
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=WIDTH,
            view_height_px=HEIGHT,
            intrinsic=intrinsic.intrinsic_matrix,
            extrinsic=np.linalg.inv(pose),
            scale=0.1/200
        )
        # print(depth_path)
        scale=1
        p_depth, color_image = dmw.process_images(depth_path, caching=True)
        p_depth=p_depth[0]*scale
        color_image=color_image[0]
        p_depth[p_depth>30*scale]=0

        # orb_points_3d = points_list[index]  # 3D points (3, n)
        orb_points_2d = points_2d[index]    # 2D points with depth (u, v, depth)
        # height, width = p_depth.shape
        orb_u = orb_points_2d[0].astype(int)
        orb_v = orb_points_2d[1].astype(int)
        orb_depths = orb_points_2d[2] * orb_true_scale
        p_depth_eroded = cv2.erode(p_depth, np.ones((5, 5), np.uint8))    
        depth_at_orb_points = p_depth_eroded[orb_v, orb_u]
        scale_ratios = orb_depths / depth_at_orb_points
        median_scale = np.median(scale_ratios)
        # print("orb depth")
        # print(orb_depths)
        # print("depth at orb points")
        # print(depth_at_orb_points)
        # print("scales")
        # print(scale_ratios)
        print("median")
        print(median_scale)
        p_depth = p_depth * median_scale
        # p_depth_eroded = cv2.erode(p_depth, np.ones((5, 5), np.uint8))    
        # depth_at_orb_points = p_depth_eroded[orb_v, orb_u]
        p_depth[p_depth>30*scale]=0


        print(np.median(p_depth))
        new_depth=p_depth.copy()
        # new_depth[~remove_sharp_gradients(p_depth, threshold=0.005)]=0
        new_depth[get_sky_mask(color_image)] = 0


        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image), 
            o3d.geometry.Image(new_depth), 
            convert_rgb_to_intensity=False)
        new_points = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        new_points.transform(pose)
        vis.add_geometry(new_points)
        vis.add_geometry(cameraLines) # Add the camera lines to the visualizer

        # new_color = color_image.copy()
        # sky_mask=get_sky_mask(color_image)
        # grad_mask=~remove_sharp_gradients(p_depth, threshold=0.005)
        # new_color[~grad_mask]=(0,255,0)
        # plot_image(new_color, grad_mask)


    vis.run()
    
if __name__ == "__main__":
    main()
from helper import vkitti
from helper import orb
from helper import deep
import numpy as np
import open3d as o3d
import cv2

# Configuration
WIDTH = 1024
HEIGHT = 1024
fx = WIDTH/2
fy = HEIGHT/2
cx = WIDTH/2
cy = HEIGHT/2
VOXEL_SIZE = 0.1  # 1cm voxels - adjust based on your scene scale
TSDF_TRUNC = 1  # Truncation value (typically 4x voxel size)

def get_sky_mask(rgb_image):
    # Your existing sky_mask function
    normalized_rgb = rgb_image / 255.0
    grayscale = cv2.cvtColor(np.float32(normalized_rgb), cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    threshold = 0.1
    smooth_mask = gradient_magnitude < threshold
    mask_uint8 = smooth_mask.astype(np.uint8) * 255
    small_kernel = np.ones((5, 5), np.uint8)
    large_kernel = np.ones((15, 15), np.uint8)
    expanded_non_sky = cv2.dilate(~mask_uint8, small_kernel, iterations=3)
    final_mask = cv2.dilate(~expanded_non_sky, large_kernel, iterations=2)
    return final_mask > 0

def main():
    # Environment config
    dataset = vkitti.dataset("midair", environment="spring")
    dataset.set_sequence(1)
    pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset)
    dmw = deep.DepthModelWrapper(model_name="depth_pro", load_weights=True)
    depth_paths = dataset.get_rgb_frame_path()
    
    # Setup intrinsic camera parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)
    
    # Create a TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=TSDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    # Process frames and integrate into TSDF
    for index in range(len(depth_paths)):
        # if index%30!=0 or index>600 or index < 200:
        if index not in [10,50,100]:
            continue
            
        pose = pose_list[index]
        # pose[:3, 3] *= (1/200)  # Scale according to your dataset
        depth_path = [depth_paths[index]]
        
        # Process depth and color images
        scale = 1
        p_depth, color_image = dmw.process_images(depth_path, caching=True)
        p_depth = p_depth[0] * scale
        color_image = color_image[0]
        
        # Apply depth filtering
        p_depth[p_depth > 30*scale] = 0
        new_depth = p_depth.copy()
        new_depth[get_sky_mask(color_image)] = 0
        
        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image), 
            o3d.geometry.Image(new_depth),
            depth_scale=1.0,  # If your depth is already in meters
            depth_trunc=30.0*scale,  # Maximum depth
            convert_rgb_to_intensity=False
        )
        
        # Integrate into TSDF volume
        # Note: For TSDF, we need camera-to-world extrinsics (already have this in pose)
        volume.integrate(rgbd_image, intrinsic, np.linalg.inv(pose))
        
        print(f"Integrated frame {index}")
    
    # Extract mesh from TSDF volume
    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    
    # Optimize mesh if needed
    mesh.compute_vertex_normals()
    
    # Optional: Simplify mesh if it's too detailed
    # mesh = mesh.simplify_quadric_decimation(100000)
    
    # Save mesh
    o3d.io.write_triangle_mesh("output_mesh.ply", mesh)
    
    # Visualize the mesh
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=WIDTH, height=HEIGHT)
    vis.add_geometry(mesh)
    
    # Add camera frustums if desired
    for index in range(len(depth_paths)):
        if index%30!=0 or index>600 or index < 200:
            continue
            
        pose = pose_list[index]
        # pose[:3, 3] *= (1/200)
        
        # Create camera visualization
        camera_lines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=WIDTH,
            view_height_px=HEIGHT,
            intrinsic=intrinsic.intrinsic_matrix,
            extrinsic=np.linalg.inv(pose),
            scale=0.1
        )
        vis.add_geometry(camera_lines)
    
    vis.run()

if __name__ == "__main__":
    main()
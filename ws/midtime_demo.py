from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np
import open3d as o3d
import threading
import time

WIDTH = 1024
HEIGHT = 1024
fx=WIDTH/2
fy=HEIGHT/2
cx=WIDTH/2
cy=HEIGHT/2
# fx = 721.5377
# fy = 721.5377
# cx = 609.5593
# cy = 172.8540
intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)
vis = o3d.visualization.Visualizer()
vis.create_window(width=WIDTH, height=HEIGHT)
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

def orb_thread_function(settings_file, rgb_folder_path):
    # This function will run in a separate thread
    orb.run_orb_slam(settings_file, rgb_folder_path)


def add_view(p_depth, pose, color_image,scale):
    global pcd, view_initialized
    pose[:3, 3] *= (scale/1000)
    p_depth[p_depth>50]=0
    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image), o3d.geometry.Image(p_depth), convert_rgb_to_intensity=False)
    
    # Create a new point cloud from the RGBD image
    new_points = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )
    
    # Transform the new points
    new_points.transform(pose)
    
    # Combine with existing points rather than replacing
    if len(pcd.points) == 0:
        pcd.points = new_points.points
        pcd.colors = new_points.colors
    else:
        pcd.points = o3d.utility.Vector3dVector(
            np.vstack([np.asarray(pcd.points), np.asarray(new_points.points)])
        )
        pcd.colors = o3d.utility.Vector3dVector(
            np.vstack([np.asarray(pcd.colors), np.asarray(new_points.colors)])
        )
    
    # Update the point cloud geometry
    vis.update_geometry(pcd)
    
    # Draw camera representation
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=WIDTH,
        view_height_px=HEIGHT,
        intrinsic=intrinsic.intrinsic_matrix,
        extrinsic=np.linalg.inv(pose),
        scale=0.001)

    # Add the camera lines to the visualizer
    vis.add_geometry(cameraLines)
    
    # Update visualization
    vis.poll_events()
    vis.update_renderer()


def main():
    # dataset = vkitti.dataset("midair", environment="spring")
    dataset = vkitti.dataset("midair", environment="spring")
    # print(dataset.get_all_seq())
    seq = 1
    depth = deep.DepthModelWrapper(model_name="depth_pro")
    dataset.set_sequence(seq)
    
    # Start ORB-SLAM in a separate thread
    orb_thread = threading.Thread(
        target=orb_thread_function, 
        args=(dataset.settings_file, dataset.get_rgb_folder_path())
    )
    orb_thread.start()
    time.sleep(1) # Give ORB-SLAM a moment to initialize


    
    # Process frames in the main thread
    while orb.is_slam_thread_running():
        if not orb.started(): continue
        try:
            index=len(orb.get_current_pose_list())
            current_frame = dataset.get_rgb_frame_path()[index]
            
        except:
            continue
        # print(current_frame)
        pose_list = orb.get_current_pose_list()
        if len(pose_list) == 0: continue
        current_pose = pose_list[-1]
        pred_depth, rgb_img = depth.process_images([current_frame])
        pred_depth=pred_depth[0]
        # pred_depth = dataset.get_depth_frame_np(index)
        pose_list = orb.get_current_pose_list()
        
        scale,_ = orb.compute_true_scale(pose_list, dataset.load_extrinsic_matrices())
        print(scale)
        
        add_view(pred_depth, current_pose, rgb_img[0],scale)
    
    # Wait for the ORB thread to complete
    orb_thread.join()
    vis.run()
    
if __name__ == "__main__":
    main()
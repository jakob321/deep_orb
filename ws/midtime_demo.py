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

def add_depth(p_depth, pose, color_image,scale):
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

def add_camera(pose,scale):
    global pcd, view_initialized
    # Update the point cloud geometry
    vis.update_geometry(pcd)
    
    # Draw camera representation
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=WIDTH,
        view_height_px=HEIGHT,
        intrinsic=intrinsic.intrinsic_matrix,
        extrinsic=np.linalg.inv(pose),
        scale=0.1
        )

    # Add the camera lines to the visualizer
    vis.add_geometry(cameraLines)
    
    # Update visualization
    vis.poll_events()
    vis.update_renderer()

latest_depth_prediction = None
latest_depth_rgb_img = None
depth_done = True
def run_deep(dataset, depth, index):
    global depth_done
    depth_done = False
    global latest_depth_prediction
    global latest_depth_rgb_img
    current_frame = dataset.get_rgb_frame_path()[index]
    latest_depth_prediction, latest_depth_rgb_img = depth.process_images([current_frame])
    depth_done = True



def main():
    # dataset = vkitti.dataset("midair", environment="spring")
    dataset = vkitti.dataset("midair", environment="spring")
    depth = deep.DepthModelWrapper(model_name="depth_pro")
    seq = 1
    dataset.set_sequence(seq)
    
    # Start ORB-SLAM in a separate thread
    orb_thread = threading.Thread(
        target=orb_thread_function, 
        args=(dataset.settings_file, dataset.get_rgb_folder_path())
    )


    deep_thread = threading.Thread(
        target=run_deep,
        args=(dataset, depth)
    )

    orb_thread.start()
    time.sleep(2) # Give ORB-SLAM a moment to initialize
    
    # Process camera poses in the main thread
    while orb.is_slam_thread_running():
        if not orb.started() or not len(orb.get_current_pose_list()): continue

        current_poses = orb.get_current_pose_list()
        latest_pose = current_poses[-1]
        scale,_ = orb.compute_true_scale(current_poses, dataset.load_extrinsic_matrices())

        global depth_done
        if depth_done:
            global latest_depth_prediction
            global latest_depth_rgb_img
            add_depth(latest_depth_prediction, latest_pose, latest_depth_rgb_img, scale)
            deep_thread.join()
            deep_thread.start()
        else:
            add_camera(latest_pose, scale)
    
    # Wait for the ORB thread to complete
    orb_thread.join()
    deep_thread.join()
    vis.run()
    
if __name__ == "__main__":
    main()
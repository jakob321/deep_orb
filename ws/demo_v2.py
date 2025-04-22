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
if True:
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
    pose[:3, 3] *= (scale/100)
    # p_depth[p_depth>50]=0
    # Create RGBD image
    p_depth=p_depth*10
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image), o3d.geometry.Image(p_depth), convert_rgb_to_intensity=False)
    
    # Create a new point cloud from the RGBD image
    new_points = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )
    
    # Transform the new points
    # new_points.transform(np.linalg.inv(pose))
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
        scale=0.1
    )

    # Add the camera lines to the visualizer
    vis.add_geometry(cameraLines)
    
    # Update visualization
    vis.poll_events()
    vis.update_renderer()

# def add_camera(pose_list,scale):
#     global pcd, view_initialized
#     pose = pose_list[-1]
#     view_pose=pose
#     if len(pose_list) >50:
#         view_pose = pose_list[-40]
#     vis.update_geometry(pcd) # Update the point cloud geometry
    
#     # Draw camera representation
#     cameraLines = o3d.geometry.LineSet.create_camera_visualization(
#         view_width_px=WIDTH,
#         view_height_px=HEIGHT,
#         intrinsic=intrinsic.intrinsic_matrix,
#         extrinsic=np.linalg.inv(pose),
#         scale=0.1
#     )
#     vis.add_geometry(cameraLines) # Add the camera lines to the visualizer
#     vc = vis.get_view_control()
#     camera_params = vc.convert_to_pinhole_camera_parameters()
#     camera_params.extrinsic = np.linalg.inv(view_pose)
#     vc.convert_from_pinhole_camera_parameters(camera_params)
#     # vc.set_zoom(1)

#     vis.poll_events() # Update visualization
#     vis.update_renderer()

def add_camera(pose_list,scale):
    global pcd, view_initialized
    pose = pose_list[-1]
    pose[:3, 3] *= (scale/100)
    view_pose=pose
    if len(pose_list) >50:
        view_pose = pose_list[-40].copy()
        view_pose[:3, 3] *= (scale/100)
    vis.update_geometry(pcd) # Update the point cloud geometry
    
    # Draw camera representation
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=WIDTH,
        view_height_px=HEIGHT,
        intrinsic=intrinsic.intrinsic_matrix,
        extrinsic=np.linalg.inv(pose),
        scale=0.1
    )
    vis.add_geometry(cameraLines) # Add the camera lines to the visualizer
    vc = vis.get_view_control()
    camera_params = vc.convert_to_pinhole_camera_parameters()

    view_matrix = np.linalg.inv(view_pose).copy()
    back_offset = 0.1  # Move backward
    up_offset = 0.1    # Move upward
    R = view_matrix[:3, :3]
    forward = R @ np.array([0, 0, -1])  # Forward direction in world coordinates
    up = R @ np.array([0, -1, 0])       # Up direction in world coordinates
    view_matrix[:3, 3] += back_offset * forward + up_offset * up
    
    camera_params.extrinsic = view_matrix
    vc.convert_from_pinhole_camera_parameters(camera_params)
    # vc.set_zoom(1)

    vis.poll_events() # Update visualization
    vis.update_renderer()

latest_depth_prediction = None
latest_rgb_img = None
is_done = False
should_compute = False

def run_deep(dataset, depth):
    global is_done, latest_depth_prediction, latest_rgb_img, should_compute
    while not should_compute: # Wait for the other threads to start
        time.sleep(0.1)
    while should_compute:
        index = len(orb.get_current_pose_list())-1
        current_frame = dataset.get_rgb_frame_path()[index]
        latest_depth_prediction, latest_rgb_img = depth.process_images([current_frame])
        # latest_depth_prediction = latest_depth_prediction[::2, ::2].copy()
        # latest_rgb_img = latest_rgb_img[::2, ::2, :].copy()
        is_done = True
        while is_done and should_compute: # Waiting for value to be read in main thread
            time.sleep(0.01)


def main():
    # Environment config
    dataset = vkitti.dataset("midair", environment="spring")
    dataset.set_sequence(1)
    # depth = deep.DepthModelWrapper(model_name="depth_pro")
    depth=deep.DepthSim(model_name="depth_pro", inference_time=0.4)
    
    # Start ORB-SLAM in a separate thread
    orb_thread = threading.Thread(
        target=orb_thread_function, 
        args=(dataset.settings_file, dataset.get_rgb_folder_path())
    )

    # Seperate thread for depth prediction
    deep_thread = threading.Thread(
        target=run_deep,
        args=(dataset, depth)
    )

    orb_thread.start()
    time.sleep(5) # Give ORB-SLAM a moment to initialize
    deep_thread.start()
    
    # Process camera poses in the main thread
    while orb.is_slam_thread_running():
        if not orb.started() or not len(orb.get_current_pose_list()): 
            continue # wait until we get our first value
        # print("orb is running")
        global is_done, latest_depth_prediction, latest_rgb_img, should_compute
        should_compute = True # make sure depth predictions is done

        # Retrieve new pose info
        current_poses = orb.get_current_pose_list()
        latest_pose = current_poses[-1]
        scale,_ = orb.compute_true_scale(current_poses, dataset.load_extrinsic_matrices())

        # Check if new depth prediction is done, else only add camera pose
        if is_done:
            print("depth done :)))")
            add_depth(latest_depth_prediction, latest_pose, latest_rgb_img, scale)
            is_done = False
        else:
            pass
            add_camera(current_poses, scale)
    should_compute = False
    
    # Wait for the ORB thread to complete
    orb_thread.join()
    deep_thread.join()
    vis.run()
    
if __name__ == "__main__":
    main()
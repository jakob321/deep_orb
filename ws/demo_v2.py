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
    pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset)
    dmw=deep.DepthModelWrapper(model_name="depth_pro", load_weights=False)
    depth_paths = dataset.get_rgb_frame_path()
    # all_depth, all_orig = dmw.process_images(deep_path, caching=True)

    # Draw camera representation
    for index in range(len(depth_paths)):
        pose = pose_list[index]
        # print("------------")
        # print(depth_paths[0])
        # print(depth_paths[1])
        # print(depth_paths[2])
        # print(index)
        # print(depth_paths[index])
        # print("------------")
        depth_path=[depth_paths[index]]
        
        # if index%100!=0 or index>100:continue
        if index not in [50,60,70,80,90,100,110,120,130]: continue
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(
            view_width_px=WIDTH,
            view_height_px=HEIGHT,
            intrinsic=intrinsic.intrinsic_matrix,
            extrinsic=np.linalg.inv(pose),
            scale=0.1
        )
        # print(depth_path)
        scale=200
        p_depth, color_image = dmw.process_images(depth_path, caching=True)
        p_depth=p_depth[0]*scale
        color_image=color_image[0]
        p_depth[p_depth>50*scale]=0
        print(np.median(p_depth))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image), 
            o3d.geometry.Image(p_depth), 
            convert_rgb_to_intensity=False)
        new_points = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        new_points.transform(pose)
        vis.add_geometry(new_points)
        vis.add_geometry(cameraLines) # Add the camera lines to the visualizer




    vis.run()
    
if __name__ == "__main__":
    main()
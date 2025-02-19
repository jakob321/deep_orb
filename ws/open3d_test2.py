import numpy as np
import open3d as o3d

all_data=np.load("test_data.npz", allow_pickle=True)
all_pose=all_data["pose"]
all_depth=all_data["depth"]
all_orig=all_data["orig"]
print(all_pose.shape)
print(all_depth.shape)
print(all_orig.shape)

WIDTH = all_orig.shape[2]
HEIGHT = all_orig.shape[1]

fx = 721.5377
fy = 721.5377
cx = 609.5593
cy = 172.8540

intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)

cam = o3d.visualization.rendering.Camera
vizualizer = o3d.visualization.Visualizer()
vizualizer.create_window()
vizualizer.create_window(width=WIDTH, height=HEIGHT)

scale_factor = 0.01  # Adjust this value as needed
clouds=[]
for i in range(all_depth.shape[0]):
    # if i not in [100, 101]:
    #     continue
    if i % 10 > 0:
        continue
    color_raw = o3d.geometry.Image(all_orig[i,:,:,:])
    depth_raw = o3d.geometry.Image(all_depth[i,:,:])
    position = np.linalg.inv(np.array(all_pose[:,:,i]))

    # Scale only the translation (last column, first three rows)
    position[:3, 3] *= scale_factor  # Scale translation (Tx, Ty, Tz)

    print("Scaled Position Matrix:\n", position)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)
    print(rgbd_image)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )

    # Apply the transformed (scaled) pose matrix
    pcd.transform(np.linalg.inv(position))
    vizualizer.add_geometry(pcd)

    # draw camera representation
    standardCameraParametersObj = vizualizer.get_view_control().convert_to_pinhole_camera_parameters()
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=WIDTH, 
        view_height_px=HEIGHT, 
        intrinsic=standardCameraParametersObj.intrinsic.intrinsic_matrix, 
        extrinsic=position,
        scale=0.001)
    vizualizer.add_geometry(cameraLines)
vizualizer.run()

print("done")
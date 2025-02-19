import numpy as np
from PIL import Image
import depth_pro
import orbslam3
import time
import threading
import torch
import torch.cuda.amp as amp
import os
import open3d as o3d

pathDataset = "../datasets/kitti/2011_09_26_drive_0093_sync/image_02"
voc_file = "../ORB_SLAM3/Vocabulary/ORBvoc.txt"
settings_file = "../ORB_SLAM3/Examples/Monocular/KITTI03.yaml"
timestamps = "../ORB_SLAM3/Examples/Monocular/EuRoC_TimeStamps/MH01.txt"

fx = 721.5377
fy = 721.5377
cx = 609.5593
cy = 172.8540


def run_inference_mixed(model, images, f_px, label=""):
    with torch.no_grad(), amp.autocast():
        start_time = time.time()
        prediction = model.infer(images, f_px=f_px)
        torch.cuda.synchronize()  # Wait for all GPU ops to complete.
        elapsed = time.time() - start_time
        print(f"Inference time {label} (mixed precision): {elapsed:.3f} seconds")
    return prediction


def run_inference_half(model, images, f_px, label=""):
    # Convert model and images to half precision.
    model.half()
    images = images.half()
    with torch.no_grad():
        start_time = time.time()
        prediction = model.infer(images, f_px=f_px)
        torch.cuda.synchronize()  # Wait for all GPU ops to complete.
        elapsed = time.time() - start_time
        print(f"Inference time {label} (all FP16): {elapsed:.3f} seconds")
    return prediction


if __name__ == "__main__":

    #========================================
    # Run ORB-SLAM3
    #========================================
    def run_slam():
            global final_result
            final_result = orbslam3.run_orb_slam3(voc_file, settings_file, pathDataset, timestamps, fps=100)

    slam_thread = threading.Thread(target=run_slam)
    slam_thread.start()

    pose_list, points_list = (0,0)
    while slam_thread.is_alive():
        pose_list, points_list = orbslam3.get_all_data_np()
        time.sleep(0.01)

    #========================================
    # Run depth-pro
    #========================================

    # Model init
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device)
    model.eval()

    # Create list of images to predict the depth of
    image_files = [file for file in os.listdir(pathDataset+"/data") if file.endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort the files numerically
    image_paths = [os.path.join(pathDataset+"/data", file) for file in image_files]

    # Create full paths
    depth_image_list = []
    original_image_list=[]
    selected_pose_list=[]

    for i, path in enumerate(image_paths):
        if i % 30 > 0: # Not using all of the images
            continue

        print("starting img: ", path)
        original_image = np.asarray(Image.open(path))
        
        image, _, f_px = depth_pro.load_rgb(path)
        image = transform(image).to(device)
        prediction = run_inference_half(model, image, f_px, label="img")
        depth = prediction["depth"].cpu().numpy()  # Convert depth tensor to numpy array

        if depth.max() > 100: # Remove points far away (needs to be done better in future)
            depth[depth > 100] = 0
            dynamic_threshold = np.percentile(depth, 90)
            depth[depth > dynamic_threshold] = 0

        original_image_list.append(original_image)
        depth_image_list.append(depth)
        selected_pose_list.append(pose_list[:,:,i])
    
    print("done with depth prediction")

    all_pose = np.stack(selected_pose_list, axis=0)
    all_depth = np.stack(depth_image_list, axis=0)
    all_orig = np.stack(original_image_list, axis=0)

    #========================================
    # Point projection using open3D
    #========================================
    WIDTH = all_orig.shape[2]
    HEIGHT = all_orig.shape[1]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)
    cam = o3d.visualization.rendering.Camera
    vizualizer = o3d.visualization.Visualizer()
    vizualizer.create_window()
    vizualizer.create_window(width=WIDTH, height=HEIGHT)
    scale_factor = 0.01  # ORB-SLAM do not know scale --> needs to be manualy tuned for now

    for i in range(all_depth.shape[0]):
        color_raw = o3d.geometry.Image(all_orig[i,:,:,:])
        depth_raw = o3d.geometry.Image(all_depth[i,:,:])
        position = np.linalg.inv(np.array(all_pose[i,:,:]))

        # Scale only the translation (last column, first three rows)
        position[:3, 3] *= scale_factor  # Scale translation (Tx, Ty, Tz)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, convert_rgb_to_intensity=False)

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

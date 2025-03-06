import numpy as np
import random
from PIL import Image
import depth_pro
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
import os
import orbslam3
import numpy as np
import time
import threading
import torch
import torch.cuda.amp as amp
matplotlib.use('TkAgg')  # OR 'Qt5Agg' (depending on availability)


# Provide paths to the vocabulary and settings files.
pathDataset = "../datasets/kitti/2011_09_26_drive_0093_sync/image_02"
voc_file = "../ORB_SLAM3/Vocabulary/ORBvoc.txt"
settings_file = "../ORB_SLAM3/Examples/Monocular/KITTI03.yaml"
timestamps = "../ORB_SLAM3/Examples/Monocular/EuRoC_TimeStamps/MH01.txt"

final_result = None

import numpy as np
import pyvista as pv

class PyvistaPlotter:
    def __init__(self):
        pv.set_plot_theme("document")
        self.pl = pv.Plotter()
        self.pl.add_text("3D projection")
        self.camera_exists = False
        self.camera_center = np.array([1, 0, 0])
        self.focal_point = np.array([0, 0, 0])
        self.near_range = 0.1

    def extract_Twc(self, Twc, focal_distance=1.0):
        # In a homogeneous transformation matrix [R t; 0 1] (i.e., Twc), the camera center in world coordinates
        # is simply the translation component.
        camera_center = Twc[:3, 3]

        # Extract the rotation matrix (upper-left 3x3 of Twc).
        R_wc = Twc[:3, :3]

        # Assuming the camera's optical axis is the local z-axis,
        # the forward vector in world coordinates is the third column of the rotation matrix.
        forward_vector = R_wc[:, 2]
        forward_vector = forward_vector / np.linalg.norm(forward_vector)

        # Compute the focal point. Note: the focal point is a point, not a direction.
        focal_point = camera_center + focal_distance * forward_vector

        return camera_center, focal_point

    def create_camera_Twc(self, Twc, focal_distance=1.0, near_range=0.6, far_range=1.2):
        camera_center, focal_point = self.extract_Twc(Twc)
        self.create_camera(camera_center, focal_point, near_range, far_range)

    def create_camera(self, camera_center, focal_point, near_range=0.6, far_range=1.2):
        self.camera_exists = True
        #self.camera_center = camera_center
        #self.focal_point = focal_point
        self.near_range = near_range

        # Create a new camera
        camera = pv.Camera()
        camera.clipping_range = (near_range, far_range)
        camera.position = camera_center
        camera.focal_point = focal_point

        frustum = camera.view_frustum(1.0)
        line = pv.Line(camera_center, focal_point)

        self.pl.add_mesh(frustum, style="wireframe")  # Draw the frustum of the camera
        self.pl.add_mesh(line, color="b")             # Draw a line from the camera to the focal point
        self.pl.add_point_labels(
            [camera_center],
            [""],
            margin=0,
            fill_shape=False,
            font_size=14,
            shape_color="white",
            point_color="red",
            text_color="black",
        )
        #self.pl.add_arrows(np.array([1, 0, 0]), np.array([-1, 0, 0]), mag=0.2)


    def project_points_Twc(self, depth, np_img, Twc):
        camera_center, focal_point = self.extract_Twc(Twc)
        #camera_center = self.camera_center
        #focal_point = self.focal_point
        # Remove outliers
        if depth.max() > 100:
            depth[depth > 100] = 0
            dynamic_threshold = np.percentile(depth, 90)
            depth[depth > dynamic_threshold] = 0
        depth=depth/10
        # Create plane of the image
        v = focal_point - camera_center  # direction vector
        v_norm = v / np.linalg.norm(v)
        distance = self.near_range
        Pc = (
            camera_center + distance * v_norm
        )  # Intersect point with plane of image

        # Create image plane and parametrize the plane using the two basis vectors
        w = (
            np.array([1, 0, 0])
            if v_norm[0] == 0 and v_norm[1] == 0
            else np.array([0, 0, 1])
        )
        u1 = np.cross(v_norm, w)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(v_norm, u1)
        u2 /= np.linalg.norm(u2)

        # Generate points on the image plane
        n_points_h = depth.shape[0]
        n_points_w = depth.shape[1]
        d = 0.0005
        dist_w = depth.shape[1] * d
        dist_h = depth.shape[0] * d
        alphas = np.linspace(-dist_w, dist_w, n_points_w)  # Parameter for u1
        betas = np.linspace(-dist_h, dist_h, n_points_h)  # Parameter for u2
        alpha_grid, beta_grid = np.meshgrid(alphas, betas)
        plane_points = (
            Pc + np.outer(alpha_grid.ravel(), u1) + np.outer(beta_grid.ravel(), u2)
        )
        np_plane_points = np.array(plane_points)

        # Create vectors from camera center through generated "pixel"
        camera_center_repeated = np.tile(
            camera_center, (np_plane_points.shape[0], 1)
        )
        vectors_to_plane = np_plane_points - camera_center_repeated
        vector_magnitudes = np.linalg.norm(
            vectors_to_plane, axis=1, keepdims=True
        )  # Compute magnitudes
        normalized_vectors = vectors_to_plane / vector_magnitudes

        # Scale vectors with predicted depth
        desired_length = np.array(depth).reshape(depth.shape[1] * depth.shape[0], 1)
        points_at_distance = camera_center + normalized_vectors * desired_length

        # visualize the points
        np_img_flat = np.array(np_img).reshape(depth.shape[1] * depth.shape[0], 3)
        self.pl.add_points(
            points_at_distance,
            scalars=np_img_flat,
            render_points_as_spheres=True,
            rgb=True,
            point_size=5,
        )
        # pl.add_points(np_plane_points, color="blue", render_points_as_spheres=True, point_size=5)
        #self.pl.remove_scalar_bar()

    def show(self):
        self.pl.show()

def run_inference_batch(model, images, f_px, label=""):
    with torch.no_grad(), amp.autocast():
        start_time = time.time()
        # Run batched inference.
        # Ensure your model.infer supports batched inputs.
        batched_prediction = model.infer(images, f_px=f_px)
        torch.cuda.synchronize()  # Wait for all GPU ops to complete.
        elapsed = time.time() - start_time
        print(f"Batch inference time {label} (mixed precision): {elapsed:.3f} seconds")
    return batched_prediction

if __name__ == "__main__":
    plotter = PyvistaPlotter()

    def run_slam():
        global final_result
        # This call runs the SLAM system and returns a string result when finished.
        final_result = orbslam3.run_orb_slam3(voc_file, settings_file, pathDataset, timestamps, fps=100)

    slam_thread = threading.Thread(target=run_slam)
    slam_thread.start()

    i=0
    pose_list, points_list = (0,0)
    while slam_thread.is_alive():
        
        pose_list, points_list = orbslam3.get_all_data_np()
        if pose_list.shape[:2] == (4, 4) and pose_list.shape[2] > 0:
            #print(pose_list)
            plotter.create_camera_Twc(pose_list[:,:,pose_list.shape[2]-1],0.1,0.2)
        i=i+1
        time.sleep(0.01)
    
    # np.savez("all_pose.npz",
    #              pose=pose_list)
    

    #Ensure the SLAM thread has completed.
    slam_thread.join()
    print("iiiii: ", i)
    print("pose_list: ", pose_list.shape)
    print("points_list: ", points_list[10].shape)
    print("SLAM thread completed.")
    print("SLAM result:", final_result)

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device)
    model.eval()
    image_files = [file for file in os.listdir(pathDataset+"/data") if file.endswith(('.jpg', '.png', '.jpeg'))]
    depth_images = []
    # Sort the files numerically based on the number in the filename
    image_files.sort(key=lambda x: int(x.split('.')[0]))  # Extract numeric part before extension

    # Create full paths
    image_paths = [os.path.join(pathDataset+"/data", file) for file in image_files]
    selected_index = np.linspace(1, pose_list.shape[2], 30).tolist()
    original_image_list=[]
    #image_paths=image_paths[:4]
    for i, path in enumerate(image_paths):
        print("starting img")
        print(path)
        original_image = np.asarray(Image.open(path))
        # image, _, f_px = depth_pro.load_rgb(path)
        # image = transform(image)
        # image = image.to(device)
        # prediction = run_inference_batch(model, image, f_px, label="img")
        # depth = prediction["depth"].cpu().numpy()  # Convert depth tensor to numpy array
        # print("done with deep")

        # # Append depth image to the list
        # depth_images.append(depth)
        original_image_list.append(original_image)

        # Optional: Continue using the depth image for plotting
        #plotter.project_points_Twc(depth, original_image, pose_list[:, :, i])

    # Convert list of depth images to a NumPy array
    #depth_images_array = np.stack(depth_images, axis=0)  # Shape: (num_images, height, width)
    original_image_array = np.stack(original_image_list, axis=0)
    # Save the array to a .npy file
    #np.save("all_depth_images.npy", depth_images_array)
    np.savez("orig_img.npz",
                 orig_img=original_image_array)

    #plotter.show()


    # Load model and preprocessing transform
    # print("Loading model...")
    # model, transform = depth_pro.create_model_and_transforms()
    # model.eval()

    # Visualize the depth map.
    # plotter = PyvistaPlotter()
    # plotter.create_camera(
    #     camera_center=np.array([1, 0, 0]),
    #     focal_point=np.array([0, 0, 0]),
    #     near_range=0.6,
    #     far_range=1.2,
    # )
    

    # # folder_path = "../kitty_dataset/image_02/data/"
    # # image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
    # # for image_path in image_paths[:5]:
    # #     print(image_path)

    #     # Load and preprocess an image.
    # image_path = "../datasets/kitti/2011_09_26_drive_0093_sync/image_02/data/0000000000.png"
    # original_image = np.asarray(Image.open(image_path))
    # # image_path = "../../../test_images/dark_and_light_forest.jpeg"
    # image, _, f_px = depth_pro.load_rgb(image_path)
    # image = transform(image)

    # # Run inference.
    # prediction = model.infer(image, f_px=f_px)
    # depth = prediction["depth"]  # Depth in [m].
    # focallength_px = prediction["focallength_px"]  # Focal length in pixels.


    # print("done with model now starting visualization")

    # plotter.project_points(depth, original_image)

    # plotter.show()
    
    
import os
import cv2
import numpy as np
import h5py

class dataset:
    def __init__(self, dataset_name):
        available_datasets = ["vkitti", "midair"]
        if dataset_name not in available_datasets:
            raise Exception("No such dataset")
        self.dataset_name = dataset_name
        self.nr_of_img = 0
        self.active_seq = ""
        self.path_dataset = ""
        self.path_depth = ""
        self.seq = []
        self.depth_img_path = []
        self.img_folder_name = ""
        self.settings_file = ""
        self.extrinsic_file = ""
        self.true_pos = []
        self.get_paths()
        self.set_sequence(0)

    def get_paths(self):
        if self.dataset_name == "vkitti":
            # ['0001', '0002', '0006', '0020', '0018']
            # bad: 3, 4
            self.path_dataset = "../datasets/vkitti/vkitti_1.3.1_rgb"
            self.path_depth = "../datasets/vkitti/vkitti_1.3.1_depthgt"
            self.extrinsic_file = "../datasets/vkitti/vkitti_1.3.1_extrinsicsgt"
            self.img_folder_name = "clone"
            self.settings_file = "../ORB_SLAM3/Examples/Monocular/KITTI03.yaml"
        if self.dataset_name == "midair":
            #['trajectory_6015', 'trajectory_6016', 'trajectory_6001', 'trajectory_6013', 'trajectory_6022', 'trajectory_6014', 'trajectory_6009', 'trajectory_6005', 'trajectory_6010', 'trajectory_6021', 'trajectory_6019', 'trajectory_6017', 'trajectory_6012', 'trajectory_6011', 'trajectory_6007', 'trajectory_6023', 'trajectory_6018', 'trajectory_6006', 'trajectory_6008', 'trajectory_6020', 'trajectory_6003', 'trajectory_6002', 'trajectory_6004', 'trajectory_6000']
            # bad: 6, 7, 8, 10, 11
            # ok: 4, 9
            # good: 2, 5, 12, 13
            self.path_dataset = "../datasets/midair/MidAir/PLE_training/winter/color_right"
            self.path_depth = "../datasets/midair/MidAir/PLE_training/winter/depth"
            self.extrinsic_file = "../datasets/midair/MidAir/PLE_training/winter/sensor_records.hdf5"
            self.img_folder_name = ""
            self.settings_file = "../ORB_SLAM3/Examples/Monocular/midair.yaml"

        return self.path_dataset, self.path_depth, self.settings_file
    
    def set_sequence(self, index):
        if not self.seq:
            self.get_all_seq()
        self.active_seq = self.seq[index]

    def get_intrinsics(self):
        if self.dataset_name == "vkitti":
            fx = 721.5377
            fy = 721.5377
            cx = 609.5593
            cy = 172.8540
        return fx,fy,cx,cy

    def get_groundtruth_pos():
        pass

    def load_extrinsic_matrices(self):
        if self.dataset_name == "vkitti":
            file_path = self.extrinsic_file + "/" + self.active_seq + "_" + self.img_folder_name + ".txt"
            with open(file_path, 'r') as f:
                lines = f.readlines()
            data = []
            for line in lines[1:]:  # Skip header
                values = list(map(float, line.strip().split()))
                frame = int(values[0])  # First value is the frame index
                matrix = np.array(values[1:]).reshape(4, 4)  # Convert remaining values to a 4x4 matrix
                data.append(matrix)
            self.true_pos = np.array(data)[:,:3,3]
        elif self.dataset_name == "midair":
            with h5py.File(self.extrinsic_file, "r") as f:
                dataset_path = self.active_seq + "/groundtruth/position"
                if dataset_path in f:
                    self.true_pos = f[dataset_path][:][::4]  # Load the dataset as a NumPy array
                else:
                    print("Dataset not found in the file.")
        return self.true_pos

    def get_rgb_frame_path(self):
        folder_path = self.path_dataset + "/" + self.active_seq + "/" + self.img_folder_name
        all_rgb = self.get_all_subfolderes(folder_path)
        self.nr_of_img = len(all_rgb)
        return all_rgb
    
    def get_rgb_folder_path(self):
        return self.path_dataset + "/" + self.active_seq + "/" + self.img_folder_name
    
    def get_all_seq(self):
        all_items = self.get_all_subfolderes(self.path_dataset)
        self.seq = [os.path.basename(item) for item in all_items if os.path.isdir(item)]
        return self.seq

    def get_all_subfolderes(self, folder_path):
        """Returns a list of absolute paths for all objects in the given folder."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"The path '{folder_path}' is not a valid directory.")
        all_sub = [os.path.abspath(os.path.join(folder_path, item)) for item in os.listdir(folder_path)]
        return all_sub

    def get_depth_frame_np(self, nr):
        frame_path = self.path_depth + "/" + self.active_seq + "/clone"
        if self.nr_of_img == 0 or not self.depth_img_path:
            self.depth_img_path = self.get_all_subfolderes(frame_path)
        img = cv2.imread(self.depth_img_path[nr], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Error: Unable to load image at '{self.depth_img_path[nr]}'.")
        return img

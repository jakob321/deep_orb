import os
import cv2
import numpy as np
import h5py
import re
from PIL import Image

class dataset:
    def __init__(self, dataset_name, environment="winter"):
        available_datasets = ["vkitti", "midair", "vkitti2"]
        if dataset_name not in available_datasets:
            raise Exception("No such dataset")
        self.dataset_name = dataset_name
        self.environment = environment # winter, fall, spring
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
        if self.dataset_name == "vkitti2":
            self.path_dataset = "../datasets/vkitti/vkitti_2.0.3_rgb"
            self.path_depth = "../datasets/vkitti/vkitti_2.0.3_depth"
            self.extrinsic_file = "../datasets/vkitti/vkitti_2.0.3_textgt"
            self.img_folder_name = "clone/frames/rgb/Camera_0"
            self.settings_file = "../ORB_SLAM3/Examples/Monocular/KITTI03.yaml"
        if self.dataset_name == "midair":
            # good winter seq: 0,1,2,3
            # good fall 2, 8, 11, 12, 13

            # seq | season |i | cashed?
            # 1     fall    1   yes
            # 2     fall    2   yes
            # 8     fall    3   yes
            # 11    fall    4   yes
            # 12    fall    5   yes
            # 13    spring  6   yes
            # 14    spring  7   yes 
            # 15    spring  8   yes - mountains
            # 21    spring  9   yes
            # 23    spring  10  yes - tree circle

            self.path_dataset = "../datasets/midair/MidAir/PLE_training/"+self.environment+"/color_left"
            self.path_depth = "../datasets/midair/MidAir/PLE_training/"+self.environment+"/depth"
            self.extrinsic_file = "../datasets/midair/MidAir/PLE_training/"+self.environment+"/sensor_records.hdf5"
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
        elif self.dataset_name == "vkitti2":
            file_path = self.extrinsic_file + "/" + self.active_seq + "/clone/extrinsic.txt"
            with open(file_path, 'r') as f:
                lines = f.readlines()

            data = []
            for line in lines[1:]:  # Skip header
                # Split by spaces and filter out empty strings
                values = [val for val in line.strip().split(' ') if val]
                
                # First two values are frame and cameraID
                frame = int(values[0])
                camera_id = int(values[1])
                
                # The remaining values are the matrix elements (16 values for a 4x4 matrix)
                matrix_values = list(map(float, values[2:18]))
                matrix = np.array(matrix_values).reshape(4, 4)
                
                data.append((frame, camera_id, matrix))

            # If you only need the position (translation) vectors
            positions = np.array([item[2][:3, 3] for item in data])  # Extract t1, t2, t3 from each matrix
            self.true_pos = positions
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
        all_files = self.get_all_subfolders(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.heic', '.raw', '.cr2', '.nef', '.arw']
        image_files = [file for file in all_files if any(file.lower().endswith(ext) for ext in image_extensions)]
        
        # Sort files based on the numeric part in the filename
        def extract_number(filename):
            # Extract numbers from the filename using regex
            numbers = re.findall(r'\d+', filename)
            # Return the last number found or 0 if no numbers
            return int(numbers[-1]) if numbers else 0
        
        image_files.sort(key=extract_number)
        
        self.nr_of_img = len(image_files)
        return image_files

    def get_rgb_folder_path(self):
        return self.path_dataset + "/" + self.active_seq + "/" + self.img_folder_name

    def get_rgb_frame(self, frame_index=0):
        image_paths = self.get_rgb_frame_path()
        
        # Check if the requested frame index is valid
        if frame_index < 0 or frame_index >= len(image_paths):
            raise IndexError(f"Frame index {frame_index} is out of range. Valid range is 0-{len(image_paths)-1}")
        
        img_path = image_paths[frame_index]
        img = cv2.imread(img_path)
        
        # Convert from BGR to RGB (OpenCV loads as BGR by default)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise IOError(f"Failed to read image at path: {img_path}")
        
        return img

    def get_all_seq(self):
        """Gets all subfolder names and sorts them based on the numbers in the name."""
        
        def extract_number(filename):
            """Extracts the first number found in a filename as an integer."""
            match = re.search(r'\d+', filename)  # Find the first sequence of digits
            return int(match.group()) if match else float('inf')  # Default to high value if no number

        all_items = self.get_all_subfolders(self.path_dataset)
        self.seq = [os.path.basename(item) for item in all_items if os.path.isdir(item)]
        self.seq.sort(key=extract_number)  # Sort based on extracted numbers
        return self.seq


    def get_all_subfolders(self, folder_path):
        """Returns a sorted list of absolute paths for all objects in the given folder."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"The path '{folder_path}' is not a valid directory.")
        
        all_sub = [os.path.abspath(os.path.join(folder_path, item)) for item in os.listdir(folder_path)]
        
        # Sort by folder name (not full path)
        return sorted(all_sub, key=lambda x: os.path.basename(x).lower())


    def get_depth_frame_np(self, nr):

        if self.dataset_name == "vkitti":
            frame_path = self.path_depth + "/" + self.active_seq + "/" + self.img_folder_name
            if self.nr_of_img == 0 or not self.depth_img_path:
                self.depth_img_path = self.get_all_subfolders(frame_path)
            img = cv2.imread(self.depth_img_path[nr],cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise ValueError(f"Error: Unable to load image at '{self.depth_img_path[nr]}'.")
            #img=img*2
            return img
        elif self.dataset_name == "midair":
            # frame_path = self.path_depth + "/" + self.active_seq + "/" + self.img_folder_name
            # if self.nr_of_img == 0 or not self.depth_img_path:
            #     self.depth_img_path = self.get_all_subfolders(frame_path)
            # print(self.depth_img_path[nr])
            # pic = Image.open(self.depth_img_path[nr])
            # img = np.asarray(pic, np.uint16)
            # img.dtype = np.float16
            # if img is None:
            #     raise ValueError(f"Error: Unable to load image at '{self.depth_img_path[nr]}'.")
            # return img
            # build path to this frame’s depth‐map

            frame_path = f"{self.path_depth}/{self.active_seq}/{self.img_folder_name}"
            if self.nr_of_img == 0 or not self.depth_img_path:
                self.depth_img_path = self.get_all_subfolders(frame_path)
            
            # load the Euclidean‐depth PNG
            depth_path = self.depth_img_path[nr]
            print(depth_path)
            pic = Image.open(depth_path)
            euclid = np.asarray(pic, np.uint16)  # shape (H, W), units = meters
            euclid.dtype = np.float16
            # intrinsics (for MidAir they’re WIDTH=HEIGHT=1024)
            H, W = euclid.shape
            f  = W / 2.0
            cx = W / 2.0
            cy = H / 2.0

            # build a per‐pixel ray length sqrt((u-cx)^2 + (v-cy)^2 + f^2)
            u = np.arange(W, dtype=np.float32)
            v = np.arange(H, dtype=np.float32)
            uu, vv = np.meshgrid(u, v)
            du = uu - cx
            dv = vv - cy
            ray_norm = np.sqrt(du*du + dv*dv + f*f)

            # convert Euclidean‐distance → axial‐distance: Z = z_euclid * f / sqrt(...)
            axial = euclid * (f / ray_norm)

            return axial
        if self.dataset_name == "vkitti2":
            frame_path = self.path_depth + "/" + self.active_seq + "/clone/frames/depth/Camera_0"
            if self.nr_of_img == 0 or not self.depth_img_path:
                self.depth_img_path = self.get_all_subfolders(frame_path)
            img = cv2.imread(self.depth_img_path[nr], cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Error: Unable to load image at '{self.depth_img_path[nr]}'.")
            #img=img*2
            return img
        raise ValueError(f"Error: Extraction of depth not implemented for this dataset")
        
        
        

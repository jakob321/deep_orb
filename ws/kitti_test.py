import orbslam3
import numpy as np
import time
import threading

# Provide paths to the vocabulary and settings files.
pathDatasetEuroc = "../datasets/kitti/2011_09_26_drive_0093_sync/image_02"
voc_file = "../ORB_SLAM3/Vocabulary/ORBvoc.txt"
settings_file = "../ORB_SLAM3/Examples/Monocular/KITTI03.yaml"
timestamps = "../ORB_SLAM3/Examples/Monocular/EuRoC_TimeStamps/MH01.txt"

final_result = None

def run_slam():
    global final_result
    # This call runs the SLAM system and returns a string result when finished.
    final_result = orbslam3.run_orb_slam3(voc_file, settings_file, pathDatasetEuroc, timestamps)

slam_thread = threading.Thread(target=run_slam)
slam_thread.start()

i=0
pose_list, points_list = (0,0)
while slam_thread.is_alive():
    pose_list, points_list = orbslam3.get_all_data_np()
    i=i+1
    time.sleep(0.1)

# Ensure the SLAM thread has completed.
slam_thread.join()
print("iiiii: ", i)
print("pose_list: ", pose_list.shape)
print("points_list: ", points_list[10].shape)
print("SLAM thread completed.")
print("SLAM result:", final_result)

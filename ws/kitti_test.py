import orbslam3
import numpy as np
import time
import threading

# Provide paths to the vocabulary and settings files.
pathDatasetEuroc = "../datasets/kitti/2011_09_26_drive_0093_sync/image_02"
voc_file = "../ORB_SLAM3/Vocabulary/ORBvoc.txt"
settings_file = "../ORB_SLAM3/Examples/Monocular/KITTI03.yaml"
timestamps = "../ORB_SLAM3/Examples/Monocular/EuRoC_TimeStamps/MH01.txt"

# Global variable to store the final result returned by the SLAM function.
final_result = None

def run_slam():
    global final_result
    # This call runs the SLAM system and returns a string result when finished.
    final_result = orbslam3.run_orb_slam3(voc_file, settings_file, pathDatasetEuroc, timestamps)

# Start the SLAM process in a separate thread.
slam_thread = threading.Thread(target=run_slam)
slam_thread.start()

# Poll for the camera pose in the main thread while SLAM is running.
last_pose = None
i=0
while slam_thread.is_alive():
    # Retrieve the latest camera pose as a NumPy array.
    pose = orbslam3.get_camera_pose()
    i=i+1
    # If this is the first pose or the pose has changed, store and print it.
    if last_pose is None or not np.array_equal(pose, last_pose):
        last_pose = np.array(pose)  # make a copy
        print("New camera pose:\n", last_pose)
    
    # Wait a bit before polling again.
    time.sleep(0.1)

# Ensure the SLAM thread has completed.
slam_thread.join()
print("iiiii: ", i)
print("SLAM thread completed.")
print("Final camera pose:\n", last_pose)
print("SLAM result:", final_result)


# test.py
import orbslam3

# Provide paths to the vocabulary and settings files.
# For a minimal test, you can use dummy paths if you are only verifying the binding.
pathDatasetEuroc="../datasets/EuRoC"
voc_file = "../ORB_SLAM3/Vocabulary/ORBvoc.txt"
settings_file = "../ORB_SLAM3/Examples/Monocular/EuRoC.yaml"
timestamps="../ORB_SLAM3/Examples/Monocular/EuRoC_TimeStamps/MH01.txt"

# Call the minimal SLAM function.
result = orbslam3.run_orb_slam3(voc_file, settings_file,pathDatasetEuroc+"/MH01", timestamps) #
print(result)

from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np
import open3d as o3d
import threading
import time
import cv2
import matplotlib.pyplot as plt

def main():
    dataset = vkitti.dataset("midair", environment="fall")
    dataset.set_sequence(1)
    pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
    
    # Create a window for visualization
    cv2.namedWindow("SLAM Keypoints Visualization", cv2.WINDOW_NORMAL)
    
    # Control parameters
    delay = 20  # milliseconds between frames (adjust for speed)
    radius = 3  # keypoint circle radius
    color = (0, 255, 0)  # Green color for keypoints (BGR)
    thickness = 2  # Circle thickness

    print(len(points_2d))
    print(len(dataset.get_rgb_frame_path()))
    # return
    
    for index in range(len(points_2d)):
        # Get RGB frame and keypoints for current index
        # if index < 21:continue
        rgb_frame = dataset.get_rgb_frame(index)
        orb_2d_points = points_2d[index]
        
        # Create a copy of the frame to draw on
        visualization = rgb_frame.copy()
        
        # Convert from RGB to BGR for OpenCV display
        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        
        # Draw keypoints on the visualization frame
        if orb_2d_points is not None and orb_2d_points.shape[1] > 0:
            # Extract u,v coordinates (first two rows)
            for point_idx in range(orb_2d_points.shape[1]):
                u, v = int(orb_2d_points[0, point_idx]), int(orb_2d_points[1, point_idx])
                cv2.circle(visualization, (u, v), radius, color, thickness)
        
        # Display frame number and keypoint count
        count_text = f"Frame: {index}, Keypoints: {orb_2d_points.shape[1] if orb_2d_points is not None else 0}"
        cv2.putText(visualization, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow("SLAM Keypoints Visualization", visualization)
        
        # Wait for the specified delay, break if user presses 'q'
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
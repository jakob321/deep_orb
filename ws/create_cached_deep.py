from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np
import open3d as o3d
import threading
import time


def main():
    dataset = vkitti.dataset("midair", environment="spring")
    dataset.set_sequence(1)
    depth=deep.DepthSim(model_name="depth_pro", inference_time=0.0)
    all_path = dataset.get_rgb_frame_path()
    latest_depth_prediction, latest_rgb_img = depth.process_images(all_path)
    print(len(latest_depth_prediction))
    print("done :)")

if __name__ == "__main__":
    main()
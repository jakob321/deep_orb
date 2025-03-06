import numpy as np

all_data=np.load("all_pose.npz", allow_pickle=True)
all_pose=all_data["pose"]
all_depth=all_data["depth"]

orig_img=np.load("orig_img.npz", allow_pickle=True)["orig_img"]

np.savez("test_data.npz",
                 pose=all_pose,
                 depth=all_depth,
                 orig=orig_img)

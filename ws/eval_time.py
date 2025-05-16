from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from helper import generic_helper

WIDTH = 1024
HEIGHT = 1024
fx = WIDTH / 2
fy = HEIGHT / 2
cx = WIDTH / 2
cy = HEIGHT / 2

def main():
    # all_seq = [1,2,8,11,12,13,14,15,21,23]
    # seq_fall = [1,2,8,11,12]
    all_seq = [1]
    seq_fall = [1]
    out_list=[]
    depth = deep.DepthModelWrapper(model_name="depth_pro", load_weights=True)   
    time_for_orb_scale=[]
    for seq in all_seq:
        active_env="spring"
        if seq in seq_fall:
            active_env="fall"
        dataset = vkitti.dataset("midair", environment=active_env)
        dataset.set_sequence(seq)

        pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
        number_of_deep_frames = 50
        deep_frames_index = np.linspace(1, len(points_2d)-1, number_of_deep_frames, dtype=int).tolist()
        rgb_path = [dataset.get_rgb_frame_path()[i] for i in deep_frames_index]
        pred_depth, rgb_img = depth.process_images(rgb_path, caching=False)        
        orb_points = [points_2d[i] for i in deep_frames_index]

        

        # Use filtered frames for scale calculation
        
        for i in range(len(pred_depth)):
            start_time = time.perf_counter()

            # scales = orb.all_scales(filtered_selected_orb_frames[:i], filtered_pred_depth[:i])
            # scale_factor, scaled_slam_matrices = orb.compute_true_scale(pose_list[:i], dataset.load_extrinsic_matrices())
            # predicted_scale = orb.predict_orb_scale(scales)
            p_depth=pred_depth[i]
            orb_points_2d=orb_points[i]
            print(i)


            p_depth = generic_helper.scale_predicted_depth(p_depth, orb_points_2d)
            if np.isnan(p_depth).any(): 
                print("NAN :O")
                continue
            p_depth = generic_helper.compute_kmeans_with_sparse_correction(
                cv2.erode(p_depth, np.ones((5, 5), np.uint8)), fx, fy, cx, cy,
                orb_points_2d,
                n_clusters=25,
                depth_weight=0.8
            )

            end_time = time.perf_counter()
            duration = (end_time - start_time)
            time_for_orb_scale.append(duration)

        # temp_dict={}
        # temp_dict["seq"]=seq
        # temp_dict["scale_factor"]=scale_factor
        # temp_dict["predicted_scale"]=predicted_scale
        # temp_dict["active_env"]=active_env
        # out_list.append(temp_dict)
        
        print("done seq:", seq)
        # print("environment: ", env)
        # print("sequence:", curr_seq)
        # print("predicted scale: ", predicted_scale)
        # print("true scale: ", scale_factor)
        # print("Used frame indices:", filtered_frame_indices)
        # print(f"{env.capitalize()} Seq{curr_seq} & {scale_factor:.3f} & {predicted_scale:.3f} & {abs((predicted_scale - scale_factor) / scale_factor * 100):.2f} \\\\")

        
        # plots.plot_error_histograms(scales,
        #     num_bins=100,
        #     log_scale_x=False,
        #     label1="Single Error",
        #     ground_truth_line=scale_factor,
        #     pred_line=predicted_scale)

    # for item in out_list:
    #     env = item["active_env"]
    #     curr_seq = item["seq"]
    #     scale_factor = item["scale_factor"]
    #     predicted_scale = item["predicted_scale"]

    #     percent_error = abs((predicted_scale - scale_factor) / scale_factor * 100)

    #     print(f"{env.capitalize()} Seq{curr_seq} & {scale_factor:.3f} & {predicted_scale:.3f} & {percent_error:.2f} \\\\")
    # import matplotlib.pyplot as plt
    # import numpy as np

    # Your timing data
    if True:
        time_ms = [t * 1000 for t in time_for_orb_scale]
        # time_ms=time_ms[5:]
        # x_val=[]
        # for i in range(len(time_ms)):
        #     x_val.append(i+5)


        # Create figure
        plt.figure(figsize=(10, 5))
        plt.plot(time_ms, marker='o', linestyle='-', label='Execution Time (ms)')

        # Mean and std lines
        mean_time = np.mean(time_ms)
        std_time = np.std(time_ms)
        plt.axhline(mean_time, color='green', linestyle='--', label=f'Mean: {mean_time:.2f} ms')
        plt.axhline(mean_time + std_time, color='red', linestyle=':', label=f'+1 Std: {mean_time + std_time:.2f} ms')
        plt.axhline(mean_time - std_time, color='red', linestyle=':', label=f'-1 Std: {mean_time - std_time:.2f} ms')


        # Labels and formatting
        plt.title('Depth map scale correction')
        plt.xlabel('Frame number')
        plt.ylabel('Execution Time (ms)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    if False:
        time_ms = [t * 1000 for t in time_for_orb_scale]
        time_ms=time_ms[5:]
        x_val=[]
        for i in range(len(time_ms)):
            x_val.append(i+5)


        # Create figure
        plt.figure(figsize=(10, 5))
        plt.plot(x_val, time_ms, marker='o', linestyle='-', label='Execution Time per Frame (ms)')

        # Labels and formatting
        plt.title('ORB-SLAM3 scale estimation execution time depending on number of frames')
        plt.xlabel('Number of frames used in calculation')
        plt.ylabel('Execution Time (ms)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
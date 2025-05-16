from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np

def main():
    all_seq = [1,2,8,11,12,13,14,15,21,23]
    seq_fall = [1,2,8,11,12]
    out_list=[]
    depth = deep.DepthModelWrapper(model_name="depth_pro", load_weights=True)

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
        pred_depth, rgb_img = depth.run_with_caching(rgb_path, override_run=True)
        print(rgb_img[0].shape)
        
        # Filter deep frames based on minimum average depth threshold
        filtered_pred_depth = []
        filtered_selected_orb_frames = []
        filtered_frame_indices = []
        
        selected_orb_frames = [points_2d[i] for i in deep_frames_index]
        
        for i in range(len(pred_depth)):
            # Calculate average depth for this frame
            avg_depth = np.mean(pred_depth[i])
            if avg_depth >= 10.0:
                filtered_pred_depth.append(pred_depth[i])
                filtered_selected_orb_frames.append(selected_orb_frames[i])
                filtered_frame_indices.append(deep_frames_index[i])
            else:
                print("too closee!!!!")
        
        print(f"Original frames: {len(pred_depth)}, Filtered frames (avg depth ≥ 5m): {len(filtered_pred_depth)}")
        
        # If all frames were filtered out, use a fallback
        if len(filtered_pred_depth) == 0:
            print("Warning: All frames have average depth < 5m. Using original frames instead.")
            filtered_pred_depth = pred_depth
            filtered_selected_orb_frames = selected_orb_frames
            filtered_frame_indices = deep_frames_index
        
        # Use filtered frames for scale calculation
        scales = orb.all_scales(filtered_selected_orb_frames, filtered_pred_depth)
        scale_factor, scaled_slam_matrices = orb.compute_true_scale(pose_list, dataset.load_extrinsic_matrices())
        predicted_scale = orb.predict_orb_scale(scales)

        temp_dict={}
        temp_dict["seq"]=seq
        temp_dict["scale_factor"]=scale_factor
        temp_dict["predicted_scale"]=predicted_scale
        temp_dict["active_env"]=active_env
        out_list.append(temp_dict)
        
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

    for item in out_list:
        env = item["active_env"]
        curr_seq = item["seq"]
        scale_factor = item["scale_factor"]
        predicted_scale = item["predicted_scale"]

        percent_error = abs((predicted_scale - scale_factor) / scale_factor * 100)

        print(f"{env.capitalize()} Seq{curr_seq} & {scale_factor:.3f} & {predicted_scale:.3f} & {percent_error:.2f} \\\\")



if __name__ == "__main__":
    main()
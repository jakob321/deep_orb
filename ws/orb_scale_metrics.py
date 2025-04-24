from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np

def main():
    # This file creates metrics for how well the scaling of ORB works using deep points
    # We use the dataset vkitti and midair for testing. The sequences have been choosen for ORB to not lose tracking
    # dataset = vkitti.dataset("vkitti2")
    # dataset = vkitti.dataset("vkitti2")
    dataset = vkitti.dataset("midair", environment="fall")
    print(dataset.get_all_seq())
    vkitti_seq=[1]
    depth = deep.DepthModelWrapper(model_name="depth_pro")
    # depth = deep.DepthModelWrapper(model_name="depth_anything_v2")
    # depth = deep.DepthModelWrapper(model_name="metric3d", metric3d_variant="vit_small")
    # dataset = vkitti.dataset("vkitti2")
    # vkitti_seq=[4]
    for seq in vkitti_seq:
        dataset.set_sequence(seq)
        pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=True)
        number_of_deep_frames = 10
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
        
        print("predicted scale: ", predicted_scale)
        print("true scale: ", scale_factor)
        print("Used frame indices:", filtered_frame_indices)
        
        plots.plot_error_histograms(scales,
            num_bins=100,
            log_scale_x=True,
            label1="Single Error",
            ground_truth_line=scale_factor,
            pred_line=predicted_scale)

if __name__ == "__main__":
    main()
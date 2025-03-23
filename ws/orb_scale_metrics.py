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
    dataset = vkitti.dataset("midair", environment="spring")
    print(dataset.get_all_seq())
    vkitti_seq=[2]
    depth = deep.DepthModelWrapper(model_name="depth_pro")
    # depth = deep.DepthModelWrapper(model_name="depth_anything_v2")
    # depth = deep.DepthModelWrapper(model_name="metric3d", metric3d_variant="vit_small")

    # dataset = vkitti.dataset("vkitti2")
    # vkitti_seq=[4]

    for seq in vkitti_seq:
        dataset.set_sequence(seq) 
        pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
        number_of_deep_frames = 10
        deep_frames_index = np.linspace(1, len(points_2d)-1, number_of_deep_frames, dtype=int).tolist()
        rgb_path = [dataset.get_rgb_frame_path()[i] for i in deep_frames_index]
        pred_depth, rgb_img = depth.run_with_caching(rgb_path, override_run=True)
        print(rgb_img[0].shape)
        selected_orb_frames = [points_2d[i] for i in deep_frames_index]
        scales = orb.all_scales(selected_orb_frames, pred_depth)
        scale_factor, scaled_slam_matrices = orb.compute_true_scale(pose_list, dataset.load_extrinsic_matrices())
        predicted_scale = orb.predict_orb_scale(scales)

        print("predicted scale: ",predicted_scale)
        print("true scale: ", scale_factor)
        plots.plot_error_histograms(scales, 
            num_bins=100, 
            log_scale_x=True, 
            label1="Single Error",
            ground_truth_line=scale_factor,
            pred_line=predicted_scale)


if __name__ == "__main__":
    main()

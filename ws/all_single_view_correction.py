from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
from helper import generic_helper

import numpy as np
import open3d as o3d

import cv2

WIDTH = 1024
HEIGHT = 1024
fx = WIDTH / 2
fy = HEIGHT / 2
cx = WIDTH / 2
cy = HEIGHT / 2
intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, fx, fy, cx, cy)

def main():
    depth = deep.DepthModelWrapper(model_name="depth_pro", load_weights=False)
    all_seq = [1,2,8,11,12,13,14,15,21,23]
    seq_fall = [1,2,8,11,12]
    all_predictions = []
    all_predictions2 = []

    for seq in all_seq:
        active_env="spring"
        if seq in seq_fall:
            active_env="fall"
        dataset = vkitti.dataset("midair", environment=active_env)
        dataset.set_sequence(seq)

        pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
        orb_true_scale, pose_list = orb.compute_true_scale(pose_list, dataset.load_extrinsic_matrices())

        number_of_deep_frames = 10
        deep_frames_index = np.linspace(0, len(dataset.get_rgb_frame_path())-2, number_of_deep_frames, dtype=int).tolist()
        rgb_path = [dataset.get_rgb_frame_path()[i] for i in deep_frames_index]
        pred_depth, rgb_img = depth.process_images(rgb_path, caching=True)


        for i, frame in enumerate(deep_frames_index):

            t_depth = dataset.get_depth_frame_np(frame)
            p_depth = pred_depth[i]
            color_img = rgb_img[i]

            orb_points_3d = points_list[frame]  # 3D points (3, n)
            orb_points_2d = points_2d[frame]  # 2D points with depth (u, v, depth)
            orb_points_2d[2] = orb_points_2d[2] * orb_true_scale

            sky_mask = generic_helper.get_sky_mask(color_img)
            valid_mask = ~sky_mask



            p_depth_scaled = generic_helper.scale_predicted_depth(p_depth, orb_points_2d)
            if np.isnan(p_depth_scaled).any():
                print("some values have nan")
                continue


            depth_diff_percentage = (p_depth_scaled-t_depth) / (t_depth + 1e-6) * 100
            depth_diff_percentage[depth_diff_percentage<-200] = -200
            depth_diff_percentage[depth_diff_percentage>200] = 200
            depth_diff_percentage = depth_diff_percentage[valid_mask]

            p_depth_scaled = generic_helper.compute_kmeans_with_sparse_correction(
                cv2.erode(p_depth_scaled, np.ones((5, 5), np.uint8)), fx, fy, cx, cy,
                orb_points_2d,
                n_clusters=25,
                depth_weight=0.8
            )

            depth_diff_percentage2 = (p_depth_scaled-t_depth) / (t_depth + 1e-6) * 100
            
            depth_diff_percentage2[depth_diff_percentage2<-200] = -200
            depth_diff_percentage2[depth_diff_percentage2>200] = 200
            depth_diff_percentage2 = depth_diff_percentage2[valid_mask]

            all_predictions.append(depth_diff_percentage.flatten())
            all_predictions2.append(depth_diff_percentage2.flatten())


    all_predictions = np.concatenate(all_predictions)
    all_predictions2 = np.concatenate(all_predictions2)
    
    # Now all_predictions contains all error values as a flat list
    # plots.plot_error_histograms(all_predictions, errors2=all_predictions2,
    #                             x_ax_label="Error compared to ground truth (%)",
    #                             num_bins=100,
    #                             log_scale_x=False,
    #                             label1="Before individual scaling",
    #                             label2="After individual scaling",
    #                             ground_truth_line=0,
    #                             title="Depth Pro 3D points compared to ground truth individual scaling")
    plots.plot_error_histograms(all_predictions, errors2=all_predictions2,
                            x_ax_label="Error compared to ground truth (%)",
                            num_bins=100,
                            log_scale_x=False,
                            label1="Before area-based depth scaling",
                            label2="After area-based depth scaling",
                            ground_truth_line=0,
                            title="Depth Pro 3D points compared to ground truth individual scaling")


if __name__ == "__main__":
    main()

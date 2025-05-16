from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helper import generic_helper


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

        number_of_deep_frames = 10
        deep_frames_index = np.linspace(0, len(dataset.get_rgb_frame_path())-2, number_of_deep_frames, dtype=int).tolist()
        rgb_path = [dataset.get_rgb_frame_path()[i] for i in deep_frames_index]
        pred_depth, rgb_img = depth.process_images(rgb_path, caching=True)
        
        # Initialize empty list to collect all depth difference percentages

        print(deep_frames_index)
        for i, frame in enumerate(deep_frames_index):
            print("nr of iter :::()")
            t_depth = dataset.get_depth_frame_np(frame)
            t_depth = np.clip(t_depth, 0, 100)
            p_depth = pred_depth[i]
            avg_depth = np.mean(p_depth)
            # if avg_depth < 20.0:
            #     print(i)
            #     continue
            # p_focal = all_focal[i]
            # t_focal = 512
            # print("scale focal: "+ str(p_focal)+"/512="+str(p_focal/512))
            # print("scale focal2: "+ str(p_focal)+"/512="+str(512/p_focal))
            # plt.imshow((p_depth/t_depth), cmap="viridis")
            # plt.show()
            sky_mask = generic_helper.get_sky_mask(rgb_img[i])
            
            depth_diff_percentage = (p_depth-t_depth) / (t_depth + 1e-6) * 100
            depth_diff_percentage[depth_diff_percentage<-200] = -200
            depth_diff_percentage[depth_diff_percentage>200] = 200
            valid_mask = ~sky_mask

            depth_diff_percentage2 = depth_diff_percentage#[valid_mask]
            depth_diff_percentage = depth_diff_percentage[valid_mask]
            # all_scale.append(depth_diff_percentage)

            # plt.imshow(depth_diff_percentage, cmap='hot', vmin=-50, vmax=50)
            # cbar4 = plt.colorbar(label='Difference (%)', fraction=0.046, pad=0.04)  # Smaller colorbar
            # cbar4.ax.tick_params(labelsize=8)  # Smaller tick labels
            # plt.title('Depth Difference (% of true depth)')
            # plt.axis('off')
            # plt.show()
            
            # Flatten the 2D depth_diff_percentage array and add all values to all_predictions
            all_predictions.append(depth_diff_percentage.flatten())
            # if avg_depth < 10.0:
            #     print(i)
            #     continue
            # if active_env == "fall" and seq == 1:
            all_predictions2.append(depth_diff_percentage2.flatten())

    print(len(all_predictions))
    print(all_predictions[0].shape)
    all_predictions = np.concatenate(all_predictions)
    all_predictions2 = np.concatenate(all_predictions2)
        
    # Now all_predictions contains all error values as a flat list
    plots.plot_error_histograms(all_predictions2, errors2=all_predictions,
                                x_ax_label="Error compared to ground truth (%)",
                                num_bins=100,
                                log_scale_x=False,
                                label1="All selected sequences without removing sky",
                                label2="All selected sequences with removed sky",
                                ground_truth_line=0,
                                title="Depth Pro 3D points compared to ground truth")

if __name__ == "__main__":
    main()

from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    dataset = vkitti.dataset("midair", environment="spring")
    vkitti_seq=[2]
    # depth = deep.DepthModelWrapper(model_name="depth_anything_v2")
    # depth = deep.DepthModelWrapper(model_name="depth_pro")
    depth = deep.DepthModelWrapper(model_name="metric3d")
    
    for seq in vkitti_seq:
        dataset.set_sequence(seq)
        number_of_deep_frames = 10
        deep_frames_index = np.linspace(0, len(dataset.get_rgb_frame_path())-2, number_of_deep_frames, dtype=int).tolist()
        rgb_path = [dataset.get_rgb_frame_path()[i] for i in deep_frames_index]
        pred_depth, rgb_img = depth.run_with_caching(rgb_path, override_run=False)
        
        # Initialize empty list to collect all depth difference percentages
        all_predictions = []
        all_scale = []
        
        for i, frame in enumerate(deep_frames_index):
            t_depth = dataset.get_depth_frame_np(frame)
            t_depth = np.clip(t_depth, 0, 100)
            p_depth = pred_depth[i]*2
            depth_diff_percentage = (p_depth-t_depth) / (t_depth + 1e-6) * 100
            depth_diff_percentage[depth_diff_percentage<-200] = -200
            depth_diff_percentage[depth_diff_percentage>200] = 200
            all_scale.append(depth_diff_percentage)

            # plt.imshow(depth_diff_percentage, cmap='hot', vmin=-50, vmax=50)
            # cbar4 = plt.colorbar(label='Difference (%)', fraction=0.046, pad=0.04)  # Smaller colorbar
            # cbar4.ax.tick_params(labelsize=8)  # Smaller tick labels
            # plt.title('Depth Difference (% of true depth)')
            # plt.axis('off')
            # plt.show()
            
            # Flatten the 2D depth_diff_percentage array and add all values to all_predictions
            all_predictions.append(depth_diff_percentage.flatten())

        print(len(all_predictions))
        print(all_predictions[0].shape)
        all_predictions = np.concatenate(all_predictions)
        
        # Now all_predictions contains all error values as a flat list
        plots.plot_error_histograms(all_predictions,
                                   x_ax_label="percentage wrong prediction",
                                   num_bins=100,
                                   log_scale_x=False,
                                   label1="error",
                                   ground_truth_line=0)

if __name__ == "__main__":
    main()

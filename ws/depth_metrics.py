from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    # This file creates metrics for how well the depth prediction network works on selected seq
    dataset = vkitti.dataset("midair", environment="spring")
    print(dataset.get_all_seq())
    print(dataset.get_all_seq()[1])
    # vkitti_seq=[14]
    vkitti_seq=[2]

    # dataset = vkitti.dataset("vkitti2")
    # print(dataset.get_all_seq())
    # vkitti_seq=[0]
    # print(dataset.get_rgb_folder_path())
    # return
    # return

    true_scale_list = []
    pred_scale_list = []

    # depth = deep.DepthModelWrapper(model_name="metric3d", metric3d_variant="vit_small")
    depth = deep.DepthModelWrapper(model_name="depth_anything_v2")
    # depth = deep.DepthModelWrapper(model_name="depth_pro")

    for seq in vkitti_seq:
        dataset.set_sequence(seq) 
        pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
        number_of_deep_frames = 10
        deep_frames_index = np.linspace(0, len(points_2d)-1, number_of_deep_frames, dtype=int).tolist()
        rgb_path = [dataset.get_rgb_frame_path()[i] for i in deep_frames_index]
        pred_depth, rgb_img = depth.run_with_caching(rgb_path, override_run=True)
        selected_orb_frames = [points_2d[i] for i in deep_frames_index]
        scales = orb.all_scales(selected_orb_frames, pred_depth)
        scale_factor, scaled_slam_matrices = orb.compute_true_scale(pose_list, dataset.load_extrinsic_matrices())
        predicted_scale = orb.predict_orb_scale(scales)

        print("predicted scale: ",predicted_scale)
        print("true scale: ", scale_factor)
        # plots.plot_error_histograms(scales, 
        #     num_bins=100, 
        #     log_scale_x=True, 
        #     label1="Single Error",
        #     ground_truth_line=scale_factor,
        #     pred_line=predicted_scale)

        pred_scale_list.append(predicted_scale)
        true_scale_list.append(scale_factor)

        comparison_frame = 1
        print("deep frame index")
        print(deep_frames_index)
        t_depth = dataset.get_depth_frame_np(deep_frames_index[comparison_frame])
        t_depth=np.clip(t_depth, 0, 100)
        p_depth = pred_depth[comparison_frame]*2
        orig_rgb = rgb_img[comparison_frame]
        # Calculate the difference in percentage
        depth_diff_percentage = np.abs(t_depth - p_depth) / (t_depth + 1e-6) * 100  # Adding small epsilon to avoid division by zero

        # Create a 2x2 figure with better proportions and control
        plt.figure(figsize=(16, 12))  # More reasonable figure size

        # 1. True depth
        plt.subplot(2, 2, 1)
        plt.imshow(t_depth, cmap='plasma', vmin=0, vmax=80)
        cbar1 = plt.colorbar(label='Depth (m)', fraction=0.046, pad=0.04)  # Smaller colorbar
        cbar1.ax.tick_params(labelsize=8)  # Smaller tick labels
        plt.title('True Depth')
        plt.axis('off')

        # 2. Predicted depth
        plt.subplot(2, 2, 2)
        plt.imshow(p_depth, cmap='plasma', vmin=0, vmax=80)
        cbar2 = plt.colorbar(label='Depth (m)', fraction=0.046, pad=0.04)  # Smaller colorbar
        cbar2.ax.tick_params(labelsize=8)  # Smaller tick labels
        plt.title('Predicted Depth')
        plt.axis('off')

        # 3. Original RGB image
        plt.subplot(2, 2, 3)
        # Maintain aspect ratio but constrain size
        h, w, _ = orig_rgb.shape
        aspect_ratio = w / h
        plt.imshow(orig_rgb)
        plt.title('Original RGB Image')
        plt.axis('off')

        # 4. Depth difference in percentage
        plt.subplot(2, 2, 4)
        plt.imshow(depth_diff_percentage, cmap='hot', vmin=0, vmax=50)
        cbar4 = plt.colorbar(label='Difference (%)', fraction=0.046, pad=0.04)  # Smaller colorbar
        cbar4.ax.tick_params(labelsize=8)  # Smaller tick labels
        plt.title('Depth Difference (% of true depth)')
        plt.axis('off')

        # Add more space between subplots
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        plt.savefig('depth_comparison.png', dpi=150, bbox_inches='tight')  # Save with good resolution
        plt.show()
    print(true_scale_list)
    print(pred_scale_list)

    compare_to_true=[]
    for i in range(len(pred_scale_list)):
        compare_to_true.append(pred_scale_list[i]/true_scale_list[i])

    print(compare_to_true)



if __name__ == "__main__":
    main()

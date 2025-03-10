from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    # This file creates metrics for how well the scaling of ORB works using deep points
    
    # We use the dataset vkitti and midair for testing. The sequences have been choosen for ORB to not lose tracking
    
    dataset = vkitti.dataset("midair")
    vkitti_seq=[1]

    # dataset = vkitti.dataset("vkitti")
    # vkitti_seq=[0]

    for seq in vkitti_seq:
        dataset.set_sequence(seq) 
        pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
        number_of_deep_frames = 10
        deep_frames_index = np.linspace(1, len(points_2d)-1, number_of_deep_frames, dtype=int).tolist()
        rgb_path = [dataset.get_rgb_frame_path()[i] for i in deep_frames_index]
        pred_depth, rgb_img = deep.run_if_no_saved_values(rgb_path, override_run=False)
        selected_orb_frames = [points_2d[i] for i in deep_frames_index]

        comparison_frame = 5
        t_depth = dataset.get_depth_frame_np(deep_frames_index[comparison_frame])
        print(t_depth.shape)
        # print(t_depth[0,0,:])
        print(len(rgb_img))
        print(len(deep_frames_index))

        diff=deep.compute_full_depth_errors(t_depth, pred_depth[comparison_frame])
        avg=np.median(diff)
        print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
        print(avg)
        # Plot the images
        # Create a figure with 2 rows and 1 column
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))  # Adjust size as needed
        epsilon = 1e-8  # Small positive value
        a = np.log10(t_depth + epsilon)  # Avoid log(0)
        a = (a - a.min()) / (a.max() - a.min())
        print(a.shape)
        # Plot first image
        ax[0].imshow(a, cmap='gray')  
        ax[0].set_title("Depth Image")  
        ax[0].axis('off')  # Hide axes

        # Plot second image
        ax[1].imshow(rgb_img[comparison_frame], cmap='gray')  
        ax[1].set_title("RGB Image")  
        ax[1].axis('off')  # Hide axes

        # Show the figure
        plt.tight_layout()  # Adjust layout
        plt.show()

        # continue

        # scales = orb.all_scales(selected_orb_frames, pred_depth)
        # scale_factor, scaled_slam_matrices = orb.compute_true_scale(pose_list, dataset.load_extrinsic_matrices())
        # predicted_scale = orb.predict_orb_scale(scales)

        # print("predicted scale: ",predicted_scale)
        # print("true scale: ", scale_factor)
        # plots.plot_error_histograms(scales, 
        #     num_bins=100, 
        #     log_scale_x=True, 
        #     label1="Single Error",
        #     ground_truth_line=scale_factor,
        #     pred_line=predicted_scale)


if __name__ == "__main__":
    main()

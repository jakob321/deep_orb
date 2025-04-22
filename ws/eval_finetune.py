from helper import vkitti
from helper import orb
from helper import deep
from helper import plots
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter

def main():
    # Initialize dataset
    dataset = vkitti.dataset("midair", environment="fall")
    # dataset = vkitti.dataset("vkitti")
    print(dataset.get_all_seq())
    vkitti_seq = [2]  # Using list format as in your original code
    
    # Initialize models with different configurations
    models = {
        "Depth Anything V2 (Default)": deep.DepthModelWrapper(model_name="depth_anything_v2"),
        "Depth Anything V2 (Finetune V1)": deep.DepthModelWrapper(model_name="depth_anything_v2", checkpoint_path="checkpoints/finetune_v1.pth"),
        "Depth Anything V2 (Finetune V2)": deep.DepthModelWrapper(model_name="depth_anything_v2", checkpoint_path="checkpoints/finetune_v4.pth")
    }
    
    # Optional: Add other models if needed
    # models["Metric3D"] = deep.DepthModelWrapper(model_name="metric3d", metric3d_variant="vit_small")
    # models["Depth Pro"] = deep.DepthModelWrapper(model_name="depth_pro")
    
    for seq in vkitti_seq:
        dataset.set_sequence(seq)
        pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
        
        # Select frames to evaluate
        number_of_frames = 5  # Number of frames to show in comparison
        frames_to_evaluate = np.linspace(1, len(points_2d)-1, number_of_frames, dtype=int).tolist()
        rgb_paths = [dataset.get_rgb_frame_path()[i] for i in frames_to_evaluate]
        
        # Get ground truth depths for selected frames
        true_depths = [np.clip(dataset.get_depth_frame_np(frame), 0, 100) for frame in frames_to_evaluate]
        
        # Dictionary to store all model predictions
        all_predictions = {}
        rgb_images = None  # Will be assigned from the first model
        
        # Run each model and store predictions
        for model_name, model in models.items():
            print(f"Running {model_name}...")
            pred_depths, rgb_imgs = model.run_with_caching(rgb_paths, override_run=True)
            
            # Store first RGB images if not already stored
            if rgb_images is None:
                rgb_images = rgb_imgs
                
            # Store predictions scaled by 2 as in your original code
            all_predictions[model_name] = [pred *1 for pred in pred_depths]
        
        # Plot comparisons for each frame
        for frame_idx, frame_num in enumerate(frames_to_evaluate):
            true_depth = true_depths[frame_idx]
            rgb_img = rgb_images[frame_idx]
            
            # Calculate number of rows and columns needed for the grid
            # Ground truth + RGB + all models
            num_plots = 2 + len(models)
            num_cols = 3  # Adjust based on how many items you want per row
            num_rows = (num_plots + num_cols - 1) // num_cols
            
            # Make sure we have at least 2 rows to separate ground truth from predictions
            num_rows = max(num_rows, 2)
            
            plt.figure(figsize=(5*num_cols, 4*num_rows))
            plt.suptitle(f"Depth Comparison - Frame {frame_num}", fontsize=16)
            
            # Plot RGB image
            plt.subplot(num_rows, num_cols, 1)
            plt.imshow(rgb_img)
            plt.title("RGB Image", fontweight='bold')
            plt.axis('off')
            
            # Plot ground truth depth
            plt.subplot(num_rows, num_cols, 2)
            plt.imshow(true_depth, cmap='hot', vmin=0, vmax=50)
            cbar = plt.colorbar(label='Depth (m)', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            plt.title("Ground Truth Depth", fontweight='bold')
            plt.axis('off')
            
            # Plot each model's prediction
            for i, (model_name, predictions) in enumerate(all_predictions.items()):
                pred_depth = predictions[frame_idx]
                
                # Calculate depth difference percentage for evaluation
                depth_diff_percentage = np.abs(true_depth - pred_depth) / (true_depth + 1e-6) * 100
                
                # Plot predicted depth
                plt.subplot(num_rows, num_cols, i+3)  # +3 because RGB and GT take positions 1 and 2
                plt.imshow(pred_depth, cmap='hot', vmin=0, vmax=50)
                cbar = plt.colorbar(label='Depth (m)', fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
                
                # Calculate average error for title
                valid_mask = true_depth > 0  # Only consider pixels with valid ground truth
                if valid_mask.sum() > 0:
                    avg_error = np.mean(depth_diff_percentage[valid_mask])
                    plt.title(f"{model_name}\nAvg Error: {avg_error:.2f}%")
                else:
                    plt.title(f"{model_name}")
                    
                plt.axis('off')
                
                # Also plot the error map to show where the model is making mistakes
                if i < num_cols:  # Only if we have space in the grid
                    error_pos = num_cols + i + 3  # Position in the second row
                    if error_pos <= num_rows * num_cols:  # Check if position exists in grid
                        plt.subplot(num_rows, num_cols, error_pos)
                        plt.imshow(depth_diff_percentage, cmap='jet', vmin=0, vmax=25)  # Limit to 25% error for better visualization
                        cbar = plt.colorbar(label='Error (%)', fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=8)
                        plt.title(f"Error Map: {model_name}")
                        plt.axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)  # Adjust to make room for suptitle
            plt.show()
            
            # Optional: Save the figure
            # plt.savefig(f"depth_comparison_frame_{frame_num}.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
from helper import vkitti
from helper import orb
import numpy as np
import matplotlib.pyplot as plt
from helper import plots

def main():
    # Initialize dataset
    dataset = vkitti.dataset("midair", environment="spring")
    vkitti_seq = [0]
    
    # Image dimensions (assuming these are defined somewhere, adding placeholders)
    w, h = 1242, 376  # Adjust based on your dataset's image dimensions
    
    # Camera parameters
    focal_length = w/2
    # [fx, fy, cx, cy]
    intrinsic = [w/2, h/2, w/2, h/2]
    
    for seq in vkitti_seq:
        dataset.set_sequence(seq)
        
        # Get ORB points (3D) and their 2D projections
        pose_list, _, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
        scale_factor, _ = orb.compute_true_scale(pose_list, dataset.load_extrinsic_matrices())
        # Points_2d is of type (u,v,depth)
        
        # Get the true scale factor
        
        print(f"True scale factor: {scale_factor}")
        
        # Collect all percentage differences in one go
        all_diffs = []
        
        # Process each frame with ORB points using vectorized operations
        for frame_idx in range(len(points_2d)):               
            # Get ground truth depth map for this frame
            gt_depth = dataset.get_depth_frame_np(frame_idx)
            
            # Convert frame_points_2d to numpy array for vectorized operations
            frame_points = np.array(points_2d[frame_idx])
            # print(len(frame_points))
            # print(frame_points.shape)
            frame_points=frame_points.T
            # Extract coordinates and depths
            u_coords = np.round(frame_points[:, 0]).astype(int)
            v_coords = np.round(frame_points[:, 1]).astype(int)
            
            # Scale the ORB depths by the scale factor
            orb_depths = frame_points[:, 2] * scale_factor
            # print(orb_depths)
            
            # Get ground truth depths for all points at once
            try:
                gt_depths = gt_depth[v_coords, u_coords]
            except:
                continue
            
            # Create mask for valid depths (both GT and ORB > 0)
            valid_mask = (gt_depths > 0) & (orb_depths > 0)
            
            # Calculate percentage differences for valid points
            if np.any(valid_mask):
                valid_gt_depths = gt_depths[valid_mask]
                valid_orb_depths = orb_depths[valid_mask]
                
                # Vectorized calculation of percentage differences
                perc_diffs = ((valid_orb_depths - valid_gt_depths) / valid_gt_depths) * 100
                all_diffs.append(perc_diffs)
        
        # Combine all differences into a single array
        percentage_differences = np.concatenate(all_diffs) if all_diffs else np.array([])
        percentage_differences[percentage_differences<-200] = -200
        percentage_differences[percentage_differences>200] = 200

        plots.plot_error_histograms(percentage_differences,
                            x_ax_label="percentage wrong prediction",
                            num_bins=100,
                            log_scale_x=False,
                            label1="error",
                            ground_truth_line=0)
        
        

if __name__ == "__main__":
    main()
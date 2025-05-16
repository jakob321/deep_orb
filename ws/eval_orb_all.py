from helper import vkitti
from helper import orb
import numpy as np
import matplotlib.pyplot as plt
from helper import plots
import cv2

def main():
    all_seq = [1,2,8,11,12,13,14,15,21,23]
    seq_fall = [1,2,8,11,12]
    
    for seq in all_seq:
        active_env="spring"
        if seq in seq_fall:
            active_env="fall"
        dataset = vkitti.dataset("midair", environment=active_env)
        dataset.set_sequence(seq)
        
        # Get ORB points (3D) and their 2D projections
        pose_list, _, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
        scale_factor, _ = orb.compute_true_scale(pose_list, dataset.load_extrinsic_matrices())
        # Points_2d is of type (u,v,depth)
        
        # Get the true scale factor
        
        print(f"True scale factor: {scale_factor}")
        
        # Collect all percentage differences in one go
        all_diffs = []
        all_diffs2 = []
        
        # Process each frame with ORB points using vectorized operations
        for frame_idx in range(len(points_2d)):      
            if frame_idx % 10 != 0: continue         
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
            gt_depth = gt_depth.astype(np.float32)

            # ----------------------------------------
            # No erode
            # ----------------------------------------
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
                all_diffs2.append(perc_diffs)

            # ----------------------------------------
            # With erode
            # ----------------------------------------
            gt_depth = cv2.erode(gt_depth, np.ones((3, 3), np.uint8))
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
    percentage_differences2 = np.concatenate(all_diffs2) if all_diffs2 else np.array([])
    percentage_differences[percentage_differences<-200] = -200
    percentage_differences[percentage_differences>200] = 200

    plots.plot_error_histograms(percentage_differences2, errors2=percentage_differences,
                        x_ax_label="Error compared to ground truth (%)",
                        num_bins=100,
                        log_scale_x=False,
                        label1="Before erode",
                        label2="After erode",
                        ground_truth_line=0,
                        title="ORB_SLAM3 3D points compared to ground truth")
        
        

if __name__ == "__main__":
    main()
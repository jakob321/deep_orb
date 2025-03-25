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
    dataset = vkitti.dataset("midair", environment="winter")
    print(dataset.get_all_seq())
    vkitti_seq=[0]
    depth = deep.DepthModelWrapper(model_name="metric3d", metric3d_variant="vit_small")
    # depth = deep.DepthModelWrapper(model_name="depth_pro")
    
    for seq in vkitti_seq:
        dataset.set_sequence(seq)
        pose_list, points_list, points_2d = orb.run_if_no_saved_values(dataset, override_run=False)
        
        # points_2d: [(u,v,depth),...frames]
        # points_list: [(3,n),...frames]
        # pose_list: [(4,4,n),...frames]
        
        number_of_deep_frames = 10
        deep_frames_index = np.linspace(1, len(points_2d)-1, number_of_deep_frames, dtype=int).tolist()
        rgb_path = [dataset.get_rgb_frame_path()[i] for i in deep_frames_index]
        pred_depth, rgb_img = depth.run_with_caching(rgb_path, override_run=False)
        
        for i, frame in enumerate(deep_frames_index):
            t_depth = dataset.get_depth_frame_np(frame)
            t_depth = np.clip(t_depth, 0, 100)
            p_depth = pred_depth[i]*2
            
            # Get corresponding ORB points for this frame
            orb_points_3d = points_list[frame]  # 3D points (3, n)
            orb_points_2d = points_2d[frame]    # 2D points with depth (u, v, depth)
            if orb_points_2d.shape[1] == 0: continue
            
            # Create empty correction map with same dimensions as p_depth
            height, width = p_depth.shape
            correction_map = np.zeros_like(p_depth)
            confidence_map = np.zeros_like(p_depth)
            
            # Extract ORB points' coordinates and depths
            orb_u = orb_points_2d[0].astype(int)
            orb_v = orb_points_2d[1].astype(int)
            orb_depths = orb_points_2d[2]
            
            # Filter out points that are out of image bounds
            valid_idx = (orb_v >= 0) & (orb_v < height) & (orb_u >= 0) & (orb_u < width)
            orb_u = orb_u[valid_idx]
            orb_v = orb_v[valid_idx]
            orb_depths = orb_depths[valid_idx]
            
            if len(orb_u) == 0:
                print(f"No valid ORB points for frame {frame}")
                continue
                
            depth_at_orb_points = p_depth[orb_v, orb_u]
            depth_diff = orb_depths - (depth_at_orb_points/100)
            
            # 1. Remove outliers using IQR method
            q1 = np.percentile(depth_diff, 25)
            q3 = np.percentile(depth_diff, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter out outliers
            inlier_mask = (depth_diff >= lower_bound) & (depth_diff <= upper_bound)
            orb_u_filtered = orb_u[inlier_mask]
            orb_v_filtered = orb_v[inlier_mask]
            depth_diff_filtered = depth_diff[inlier_mask]
            
            print(f"Removed {len(orb_u) - len(orb_u_filtered)} outliers out of {len(orb_u)} points")
            
            if len(orb_u_filtered) == 0:
                print(f"No valid inliers for frame {frame} after outlier removal")
                continue
            
            # Calculate median difference for boundary points
            median_diff = np.median(depth_diff_filtered)
            
            # 2. Generate confidence map based on point density
            points = np.vstack((orb_v_filtered, orb_u_filtered)).T
            # Create KD-Tree for efficient distance calculation
            tree = KDTree(points)
            
            # Generate grid points for the entire image
            y_grid, x_grid = np.mgrid[0:height, 0:width]
            grid_points = np.vstack((y_grid.flatten(), x_grid.flatten())).T
            
            # Calculate distance to nearest point for each pixel
            distances, _ = tree.query(grid_points, k=1)
            distances = distances.reshape(height, width)
            
            # Convert distances to confidence values (closer points = higher confidence)
            max_influence_distance = 100  # Pixels beyond this distance have minimal influence
            # *** TUNING PARAMETER 1: Increase this value to extend influence range ***
            # Try values between 50-200 for wider influence
            
            confidence = np.exp(-distances / max_influence_distance)
            confidence_map = confidence.reshape(height, width)
            
            # Create boundary points to help with extrapolation
            # Add virtual points at the image boundaries with median difference value
            boundary_u = np.concatenate([
                np.zeros(height, dtype=int),                   # Left edge
                np.ones(height, dtype=int) * (width - 1),      # Right edge
                np.arange(width, dtype=int),                   # Top edge
                np.arange(width, dtype=int)                    # Bottom edge
            ])
            
            boundary_v = np.concatenate([
                np.arange(height, dtype=int),                  # Left edge
                np.arange(height, dtype=int),                  # Right edge
                np.zeros(width, dtype=int),                    # Top edge
                np.ones(width, dtype=int) * (height - 1)       # Bottom edge
            ])
            
            # Combine filtered ORB points with boundary points
            all_u = np.concatenate([orb_u_filtered, boundary_u])
            all_v = np.concatenate([orb_v_filtered, boundary_v])
            all_points = np.vstack((all_v, all_u)).T  # Note: v,u order for y,x coordinates
            
            # Use a smaller boundary effect
            # *** TUNING PARAMETER 5: Adjust boundary influence ***
            # 0.0 = no influence, 1.0 = full median difference at boundaries
            boundary_factor = 0.3  
            boundary_values = np.ones(len(boundary_u)) * (median_diff * boundary_factor)
            all_values = np.concatenate([depth_diff_filtered, boundary_values])

            # Interpolate correction map
            correction_map = griddata(all_points, all_values, (y_grid, y_grid), method='linear', fill_value=median_diff*0.1)
            
            # Apply smoothing to reduce abrupt changes
            # *** TUNING PARAMETER 2: Increase sigma for wider/smoother Gaussians ***
            # Try values between 1.0-5.0 for smoother transitions
            sigma = 1.0  
            correction_map = gaussian_filter(correction_map, sigma=sigma)
            
            # Scale correction by confidence
            # *** TUNING PARAMETER 3: Adjust scaling factor for correction strength ***
            # Decrease for subtler corrections, increase for stronger effect
            scaling_factor = 5.0  
            weighted_correction = correction_map * confidence_map * scaling_factor
            
            # Ensure exact matching at ORB points
            # *** TUNING PARAMETER 4: Match scaling factor with the one above ***
            # This ensures consistent scaling at the exact points
            point_scaling_factor = scaling_factor  
            for u, v, diff in zip(orb_u_filtered, orb_v_filtered, depth_diff_filtered):
                weighted_correction[v, u] = diff * point_scaling_factor
            
            # Apply correction
            corrected_depth = p_depth + weighted_correction
            
            # Evaluate results
            corr_depth_percentage = np.abs(t_depth - corrected_depth) / (t_depth + 1e-6) * 100
            depth_diff_percentage = np.abs(t_depth - p_depth) / (t_depth + 1e-6) * 100

            # Plot results
            plt.figure(figsize=(14, 10))
            
            plt.subplot(2, 3, 1)
            plt.imshow(depth_diff_percentage, cmap='hot', vmin=0, vmax=50)
            cbar = plt.colorbar(label='Difference (%)', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            plt.title('Original Depth Difference')
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.imshow(corr_depth_percentage, cmap='hot', vmin=0, vmax=50)
            cbar = plt.colorbar(label='Difference (%)', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            plt.title('Adjusted Depth Difference')
            plt.axis('off')

            plt.subplot(2, 3, 3)
            plt.imshow(weighted_correction, cmap='viridis')
            cbar = plt.colorbar(label='Correction', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            plt.title('Weighted Correction Map')
            plt.axis('off')
            
            plt.subplot(2, 3, 4)
            plt.imshow(confidence_map, cmap='inferno')
            cbar = plt.colorbar(label='Confidence', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            plt.title('Confidence Map')
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.imshow(t_depth, cmap='hot', vmin=0, vmax=50)
            cbar = plt.colorbar(label='Depth (m)', fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            plt.title('True Depth')
            plt.axis('off')
            
            plt.subplot(2, 3, 6)
            plt.imshow(rgb_img[i])
            plt.scatter(orb_u_filtered, orb_v_filtered, c=depth_diff_filtered, cmap='coolwarm', 
                       alpha=0.7, s=15, edgecolors='white', linewidths=0.5)
            plt.colorbar(label='Depth Diff at Points', fraction=0.046, pad=0.04)
            plt.title('ORB Points (After Outlier Removal)')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np


def compute_full_depth_errors(true_depth, predicted_depth, clip_value=655.35):
    """
    Compute absolute error for the full depth map.
    """
    p_depth = predicted_depth.copy()
    p_depth[p_depth > clip_value] = clip_value
    full_errors = np.abs(true_depth - p_depth)
    return full_errors

def prepare_histogram_bins(errors1, errors2, num_bins=50, epsilon=1e-8):
    """
    Prepare common logarithmic bins for histogram plotting.
    Returns bins and flattened arrays for both error types.
    """
    errors1_flat = errors1.ravel()
    errors2_flat = errors2.ravel()
    errors1_flat = errors1_flat[errors1_flat > 0]
    errors2_flat = errors2_flat[errors2_flat > 0]
    min_val = np.minimum(errors1_flat.min(), errors2_flat.min())
    max_val = np.maximum(errors1_flat.max(), errors2_flat.max())
    bins = np.logspace(np.log10(min_val + epsilon), np.log10(max_val + epsilon), num_bins)
    return bins, errors1_flat, errors2_flat

def plot_error_histograms(errors1, errors2, bins):
    """
    Plot a combined histogram comparing keypoint and full depth errors.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(errors1, bins=bins, density=True, alpha=0.5,
             label="Keypoint Absolute Error", color='blue', edgecolor='black')
    plt.hist(errors2, bins=bins, density=True, alpha=0.5,
             label="Depth Map Absolute Error", color='red', edgecolor='black')
    plt.xscale('log')
    plt.xlabel("Absolute Error (meters, log scale)")
    plt.ylabel("Normalized Frequency")
    plt.title("Comparison of Depth Absolute Errors")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
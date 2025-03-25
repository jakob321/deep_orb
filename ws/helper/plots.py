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

import numpy as np
import matplotlib.pyplot as plt

def prepare_histogram_bins(errors1, errors2=None, num_bins=50, epsilon=1e-8, log_scale_x=True):
    """
    Prepare common bins for histogram plotting.

    If log_scale_x is True, logarithmic bins are used; otherwise, linear bins are used.
    If errors2 is None, only errors1 is used to compute the bins.
    
    Returns:
      bins: array of bin edges.
      errors1_flat: flattened positive error values from errors1.
      errors2_flat: flattened positive error values from errors2 (or None if errors2 is None).
    """
    errors1_flat = errors1.ravel()
    # errors1_flat = errors1_flat[errors1_flat > 0]
    
    if errors2 is None:
        min_val = errors1_flat.min()
        max_val = errors1_flat.max()
        if log_scale_x:
            bins = np.logspace(np.log10(min_val + epsilon), np.log10(max_val + epsilon), num_bins)
        else:
            bins = np.linspace(min_val + epsilon, max_val + epsilon, num_bins)
        return bins, errors1_flat, None
    else:
        errors2_flat = errors2.ravel()
        # errors2_flat = errors2_flat[errors2_flat > 0]
        min_val = np.minimum(errors1_flat.min(), errors2_flat.min())
        max_val = np.maximum(errors1_flat.max(), errors2_flat.max())
        if log_scale_x:
            bins = np.logspace(np.log10(min_val + epsilon), np.log10(max_val + epsilon), num_bins)
        else:
            bins = np.linspace(min_val + epsilon, max_val + epsilon, num_bins)
        return bins, errors1_flat, errors2_flat

def plot_error_histograms(errors1, errors2=None, num_bins=50, epsilon=1e-8,
                          label1="Error 1", label2="Error 2", x_ax_label=" ",
                          ground_truth_line=None, pred_line=None,
                          log_scale_x=True):
    """
    Plot one or two histograms of error values with optional vertical reference lines.

    Parameters:
      errors1 (array): First set of errors.
      errors2 (array, optional): Second set of errors (if provided, both are plotted).
      num_bins (int): Number of histogram bins.
      epsilon (float): Small value to avoid log(0) issues.
      label1 (str): Label for the first histogram.
      label2 (str): Label for the second histogram.
      ground_truth_line (float, optional): Draws a vertical red line at this value.
      pred_line (float, optional): Draws a vertical blue line at this value.
      log_scale_x (bool): If True, the x-axis is logarithmic.
    """
    bins, errors1_flat, errors2_flat = prepare_histogram_bins(errors1, errors2, num_bins, epsilon, log_scale_x)

    plt.figure(figsize=(8, 6))
    plt.hist(errors1_flat, bins=bins, density=True, alpha=0.5,
             label=label1, color='blue', edgecolor='black')
    if errors2_flat is not None:
        plt.hist(errors2_flat, bins=bins, density=True, alpha=0.5,
                 label=label2, color='red', edgecolor='black')

    # Add ground truth and predicted lines with annotations
    if ground_truth_line is not None:
        plt.axvline(ground_truth_line, color='red', linestyle='-', label=f'Ground Truth ({ground_truth_line:.3f})')
        plt.text(ground_truth_line, plt.ylim()[1] * 0.9, f'{ground_truth_line:.3f}', 
                 color='red', ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    if pred_line is not None:
        plt.axvline(pred_line, color='blue', linestyle='-', label=f'Predicted ({pred_line:.3f})')
        plt.text(pred_line, plt.ylim()[1] * 0.8, f'{pred_line:.3f}', 
                 color='blue', ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    if log_scale_x:
        plt.xscale('log')

    plt.xlabel(x_ax_label + (" [log scale]" if log_scale_x else ""))
    plt.ylabel("Normalized Frequency")
    plt.title("Comparison of Depth Absolute Errors")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

import pickle
import os
import cv2
import numpy as np
from scipy.interpolate import Rbf
import cv2

def save_data(file_name, *objects, path="", use_full_path=False):
    """
    Save multiple arbitrary objects to a file using pickle.
    
    Args:
        file_name (str): Name of the file to save to
        *objects: Objects to save
        path (str): Optional subdirectory path
        use_full_path (bool): If True, file_name is used as is; otherwise it's processed
    """
    print("saving data to file")
    
    if use_full_path:
        s = file_name
    else:
        file_name = file_name.replace(".", "").replace("/", "_")
        s = os.path.join("saved_values", path, file_name)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(s), exist_ok=True)
    
    try:
        with open(s, "wb") as f:
            pickle.dump(objects, f)  # Save all objects as a tuple
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False


def load_data(file_name, use_full_path=False):
    """
    Load saved data from a file if it exists, otherwise return False.
    
    Args:
        file_name (str): Name of the file to load from
        use_full_path (bool): If True, file_name is used as is; otherwise it's processed
    
    Returns:
        The loaded objects or False if loading failed
    """
    if use_full_path:
        s = file_name
    else:
        file_name = file_name.replace(".", "").replace("/", "_")
        s = os.path.join("saved_values", file_name)
    
    print(f"trying to load: {s}")
    
    if not os.path.exists(s):  # Check if file exists
        return False
    
    try:
        print("loading data from file")
        with open(s, "rb") as f:
            loaded_data = pickle.load(f)  # Load all saved objects
            # If there's only one object in the tuple, return just that object
            if isinstance(loaded_data, tuple) and len(loaded_data) == 1:
                return loaded_data[0]
            return loaded_data
    except (pickle.UnpicklingError, EOFError, Exception) as e:  # Handle corrupted files
        print(f"No file to load or corrupted file: {e}")
        return False


def get_sharp_gradients_mask(depth_map, threshold=0.5):
    if depth_map.max() > 1.0:
        normalized_depth = depth_map / depth_map.max()
    else:
        normalized_depth = depth_map.copy()
    
    # Calculate gradients in x and y directions using Sobel
    grad_x = cv2.Sobel(normalized_depth, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized_depth, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create a mask where gradient is below threshold (not sharp)
    mask = gradient_magnitude < threshold
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    #dilated_mask = ~cv2.dilate(~mask_uint8, kernel, iterations=1)
    return ~(mask_uint8 > 0)


def get_sky_mask(rgb_image):
    # Normalize image to 0-1 range (with clearer comment)
    normalized_rgb = rgb_image / 255.0
    
    # Convert to grayscale for gradient detection
    grayscale = cv2.cvtColor(np.float32(normalized_rgb), cv2.COLOR_RGB2GRAY)
    
    # Calculate gradient magnitude directly (combine x and y steps)
    grad_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create initial mask where gradient is below threshold (smooth areas like sky)
    threshold = 0.1
    smooth_mask = gradient_magnitude < threshold
    
    # Convert to uint8 for OpenCV operations
    mask_uint8 = smooth_mask.astype(np.uint8) * 255
    
    # Define kernels once with descriptive names
    small_kernel = np.ones((5, 5), np.uint8)
    large_kernel = np.ones((15, 15), np.uint8)
    
    # Process mask with morphological operations
    # First dilate the inverse mask (non-sky areas)
    expanded_non_sky = cv2.dilate(~mask_uint8, small_kernel, iterations=3)
    # Then dilate the inverse again to get final sky mask
    final_mask = cv2.dilate(~expanded_non_sky, large_kernel, iterations=2)
    
    # Return boolean mask
    return final_mask > 0

def scale_predicted_depth(p_depth, orb_points_2d):
    orb_u = orb_points_2d[0].astype(int)
    orb_v = orb_points_2d[1].astype(int)
    orb_depths = orb_points_2d[2]

    # We erode so that the ORB points hit correct surface with greater chance
    p_depth_eroded = cv2.erode(p_depth, np.ones((5, 5), np.uint8))    
    depth_at_orb_points = p_depth_eroded[orb_v, orb_u]
    scale_ratios = orb_depths / depth_at_orb_points
    median_scale = np.median(scale_ratios)

    return p_depth * median_scale

def create_correction_map(p_depth, orb_points_2d, rgb_image):
    # Extract ORB coords & depths
    orb_u = orb_points_2d[0].astype(int)
    orb_v = orb_points_2d[1].astype(int)
    orb_depths = orb_points_2d[2]
    height, width = p_depth.shape

    # Robustify sensor depth
    p_depth_eroded = cv2.erode(p_depth, np.ones((5, 5), np.uint8))
    depth_at_orb = p_depth_eroded[orb_v, orb_u]

    # Compute residuals
    depth_diff = orb_depths - depth_at_orb

    # Sky mask & filter
    sky_mask = get_sky_mask(rgb_image)
    valid_u, valid_v, valid_diffs = [], [], []
    for u, v, d in zip(orb_u, orb_v, depth_diff):
        if (0 <= v < height and 0 <= u < width
            and not sky_mask[v, u]
            and abs(d) <= 10.0):
            valid_u.append(u)
            valid_v.append(v)
            valid_diffs.append(d)

    if not valid_u:
        return np.zeros_like(p_depth), np.zeros_like(p_depth)

    valid_u = np.array(valid_u)
    valid_v = np.array(valid_v)
    valid_diffs = np.array(valid_diffs)

    # 1) Linear RBF on a coarse grid, then upsample to full resolution
    rbf = Rbf(valid_u, valid_v, valid_diffs, function='linear', smooth=1.0)

    # Downscale factor
    downscale = 4
    coarse_h = max(2, height // downscale)
    coarse_w = max(2, width  // downscale)

    # Create coarse-grid coordinates in original pixel space
    coarse_x = np.linspace(0, width - 1, coarse_w)
    coarse_y = np.linspace(0, height - 1, coarse_h)
    cx, cy   = np.meshgrid(coarse_x, coarse_y)

    # Evaluate RBF on the small grid
    coarse_corr = rbf(cx, cy)

    # Upsample back to full resolution
    # Note: cv2.resize takes (width, height)
    global_correction = cv2.resize(
        coarse_corr.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC
    )

    # 2) Build a confidence mask of Gaussians via union (local-windowed)
    influence_radius = 70.0
    sigma = influence_radius
    radius = int(3 * sigma)

    # Precompute Gaussian kernel patch
    ax = np.arange(-radius, radius+1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))

    mask = np.zeros_like(global_correction, dtype=float)
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))

    for u, v in zip(valid_u, valid_v):
        # Window bounds
        x0 = max(u - radius, 0)
        x1 = min(u + radius + 1, width)
        y0 = max(v - radius, 0)
        y1 = min(v + radius + 1, height)

        # Kernel slice
        kx0 = x0 - (u - radius)
        kx1 = kx0 + (x1 - x0)
        ky0 = y0 - (v - radius)
        ky1 = ky0 + (y1 - y0)

        sub = mask[y0:y1, x0:x1]
        gsub = kernel[ky0:ky1, kx0:kx1]
        mask[y0:y1, x0:x1] = 1 - (1 - sub) * (1 - gsub)

    # 3) Localize the correction
    weighted_correction = global_correction * mask

    # 4) Apply it
    corrected_map = p_depth + weighted_correction

    # Cast corrected_map to the same dtype as p_depth
    corrected_map = corrected_map.astype(p_depth.dtype)

    return corrected_map



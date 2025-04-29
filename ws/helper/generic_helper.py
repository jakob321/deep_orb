import pickle
import os
import cv2
import numpy as np

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


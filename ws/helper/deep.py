import numpy as np
from PIL import Image
import depth_pro
import torch
import torch.cuda.amp as amp
import time
from . import generic_helper

# ====== Inference Functions ======
def run_inference_mixed(model, images, f_px, label=""):
    with torch.no_grad(), amp.autocast():
        start_time = time.time()
        prediction = model.infer(images, f_px=f_px)
        torch.cuda.synchronize()  # Wait for all GPU ops to complete.
        elapsed = time.time() - start_time
        print(f"Inference time {label} (mixed precision): {elapsed:.3f} seconds")
    return prediction

def run_inference_half(model, images, f_px, label=""):
    # Convert model and images to half precision.
    model.half()
    images = images.half()
    with torch.no_grad():
        start_time = time.time()
        prediction = model.infer(images, f_px=f_px)
        torch.cuda.synchronize()  # Wait for all GPU ops to complete.
        elapsed = time.time() - start_time
        print(f"Inference time {label} (all FP16): {elapsed:.3f} seconds")
    return prediction

def run_depth_pro(paths):
    # Model initialization
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device)
    model.eval()

    all_depth = []
    all_orig = []
    for path in paths:
        original_image = np.asarray(Image.open(path))
        all_orig.append(original_image)
        
        image, _, f_px = depth_pro.load_rgb(path)
        image = transform(image).to(device)
        prediction = run_inference_half(model, image, f_px, label="img")
        depth = prediction["depth"].cpu().numpy()  # Convert depth tensor to numpy array
        all_depth.append(depth)

    return all_depth, all_orig

def compute_full_depth_errors(true_depth, predicted_depth, clip_value=655.35):
    """
    Compute absolute error for the full depth map.
    """
    p_depth = predicted_depth.copy()
    p_depth[p_depth > clip_value] = clip_value
    full_errors = np.abs(true_depth - p_depth)
    return full_errors

def run_if_no_saved_values(paths, override_run=False):
    save_file_path = paths[0]+str(len(paths))
    result = generic_helper.load_data(save_file_path)
    if result == False or override_run:
        new_result = run_depth_pro(paths)
        generic_helper.save_data(save_file_path, *new_result)
        return new_result
    
    return result
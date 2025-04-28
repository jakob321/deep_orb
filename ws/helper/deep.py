import numpy as np
from PIL import Image
import depth_pro
import torch
import torch.cuda.amp as amp
import time
import cv2
from . import generic_helper

# Import for Depth Anything V2
from depth_anything_v2.dpt import DepthAnythingV2

# Import for Metric3D
import os
import torch.nn.functional as F


class DepthModelWrapper:
    def __init__(
        self,
        model_name="depth_pro",
        encoder="vitb",
        max_depth=20,
        checkpoint_path=None,
        metric3d_variant="vit_small",
        load_weights=True
    ):
        """
        Initialize the depth model wrapper with the specified model.

        Args:
            model_name (str): The name of the model to use. Supports 'depth_pro', 'depth_anything_v2', or 'metric3d'.
            encoder (str): Encoder type for depth_anything_v2 ('vits', 'vitb', 'vitl', 'vitg').
            max_depth (float): Maximum depth value for depth_anything_v2.
            checkpoint_path (str): Path to the checkpoint file for depth_anything_v2.
            metric3d_variant (str): Variant of metric3d model ('vit_small', 'vit_large', 'vit_giant2',
                                   'convnext_tiny', 'convnext_large').
        """
        self.model_name = model_name
        # Check device compatibility for all models
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print("Using device:", self.device)

        if not load_weights:
            return

        # Initialize the appropriate model based on model_name
        if model_name == "depth_pro":
            print("Loading depth_pro model...")
            self.model, self.transform = depth_pro.create_model_and_transforms()
            self.model = self.model.to(self.device)
            self.model.eval()
        elif model_name == "depth_anything_v2":
            print(f"Loading depth_anything_v2 model with {encoder} encoder...")
            self.encoder = encoder
            self.max_depth = max_depth
            self.input_size = 518  # Default input size

            # Model configurations for different encoder types
            model_configs = {
                "vits": {
                    "encoder": "vits",
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                },
                "vitb": {
                    "encoder": "vitb",
                    "features": 128,
                    "out_channels": [96, 192, 384, 768],
                },
                "vitl": {
                    "encoder": "vitl",
                    "features": 256,
                    "out_channels": [256, 512, 1024, 1024],
                },
                "vitg": {
                    "encoder": "vitg",
                    "features": 384,
                    "out_channels": [1536, 1536, 1536, 1536],
                },
            }
            # default_checkpoint = f"checkpoints/depth_anything_v2_metric_vkitti_{encoder}.pth"

            # Create the model
            self.model = DepthAnythingV2(
                **{**model_configs[encoder], "max_depth": max_depth}
            )

            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
            else:
                # default_checkpoint = f"checkpoints/finetune_v2.pth"
                default_checkpoint = (
                    f"checkpoints/depth_anything_v2_metric_vkitti_{encoder}.pth"
                )
                print(
                    f"No checkpoint provided, trying to load from default path: {default_checkpoint}"
                )
                checkpoint = torch.load(default_checkpoint, map_location="cpu")

            # Check if the checkpoint contains a nested 'model' key
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                print("Loading model from training checkpoint structure")
                state_dict = checkpoint["model"]
            else:
                print("Loading model directly from state dictionary")
                state_dict = checkpoint

            # Check for 'module.' prefix from DistributedDataParallel and remove it if needed
            if all(k.startswith("module.") for k in state_dict.keys()):
                print("Removing 'module.' prefix from state dictionary keys")
                state_dict = {
                    k[7:]: v for k, v in state_dict.items()
                }  # Remove the 'module.' prefix

            self.model.load_state_dict(state_dict)

            self.model = self.model.to(self.device).eval()
        elif model_name == "metric3d":
            print(f"Loading metric3d model with {metric3d_variant} variant...")
            self.metric3d_variant = metric3d_variant

            # Map variant names to function names
            variant_to_function = {
                "vit_small": "metric3d_vit_small",
                "vit_large": "metric3d_vit_large",
                "vit_giant2": "metric3d_vit_giant2",
                "convnext_tiny": "metric3d_convnext_tiny",
                "convnext_large": "metric3d_convnext_large",
            }

            if metric3d_variant not in variant_to_function:
                raise ValueError(
                    f"Metric3D variant '{metric3d_variant}' not supported. "
                    f"Available variants: {list(variant_to_function.keys())}"
                )

            # Set default input size based on model variant
            if "vit" in metric3d_variant:
                self.input_size = (616, 1064)  # Default for ViT models
            else:
                self.input_size = (544, 1216)  # Default for ConvNeXt models

            # Load the model using torch.hub
            try:
                self.model = torch.hub.load(
                    "yvanyin/metric3d",
                    variant_to_function[metric3d_variant],
                    pretrain=True,
                )
                self.model = self.model.to(self.device).eval()

                # Setup normalization parameters for metric3d
                self.mean = (
                    torch.tensor([123.675, 116.28, 103.53])
                    .float()
                    .to(self.device)[:, None, None]
                )
                self.std = (
                    torch.tensor([58.395, 57.12, 57.375])
                    .float()
                    .to(self.device)[:, None, None]
                )
                self.padding = [
                    123.675,
                    116.28,
                    103.53,
                ]  # Used for padding during preprocessing

            except Exception as e:
                raise RuntimeError(f"Failed to load Metric3D model: {e}")
        else:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Currently supports 'depth_pro', 'depth_anything_v2', and 'metric3d'."
            )

    def run_inference_mixed(self, images, f_px, label=""):
        """Run inference with mixed precision."""
        with torch.no_grad(), amp.autocast():
            start_time = time.time()
            prediction = self.model.infer(images, f_px=f_px)
            torch.cuda.synchronize()  # Wait for all GPU ops to complete.
            elapsed = time.time() - start_time

        print(f"Inference time {label} (mixed precision): {elapsed:.3f} seconds")
        return prediction

    def run_inference_half(self, images, f_px, label=""):
        """Run inference with half precision (FP16)."""
        # Convert model and images to half precision.
        self.model.half()
        images = images.half()

        with torch.no_grad():
            start_time = time.time()
            prediction = self.model.infer(images, f_px=f_px)
            torch.cuda.synchronize()  # Wait for all GPU ops to complete.
            elapsed = time.time() - start_time

        print(f"Inference time {label} (all FP16): {elapsed:.3f} seconds")
        return prediction

    def process_metric3d_image(self, image_path):
        """
        Process an image using the Metric3D model.

        Args:
            image_path (str): Path to the image file

        Returns:
            tuple: (depth_map, original_image)
        """
        # Read original image
        original_image = cv2.imread(image_path)
        original_image_rgb = original_image[:, :, ::-1]  # Convert BGR to RGB
        h, w = original_image.shape[:2]

        # Default intrinsic parameters (can be adjusted if camera parameters are known)
        # [fx, fy, cx, cy]
        intrinsic = [707.0493, 707.0493, w / 2, h / 2]

        focal_length = w / 2
        # [fx, fy, cx, cy]
        intrinsic = [w / 2, h / 2, w / 2, h / 2]

        # Resize to fit model's input size
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        resized_image = cv2.resize(
            original_image_rgb,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

        # Scale intrinsic parameters
        scaled_intrinsic = [
            intrinsic[0] * scale,
            intrinsic[1] * scale,
            intrinsic[2] * scale,
            intrinsic[3] * scale,
        ]

        # Pad image to match input size
        h_resized, w_resized = resized_image.shape[:2]
        pad_h = self.input_size[0] - h_resized
        pad_w = self.input_size[1] - w_resized
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        padded_image = cv2.copyMakeBorder(
            resized_image,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=self.padding,
        )
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        # Normalize and convert to tensor
        rgb_tensor = torch.from_numpy(padded_image.transpose((2, 0, 1))).float()
        # Move the tensor to the device before normalization
        rgb_tensor = rgb_tensor.to(self.device)
        rgb_tensor = torch.div((rgb_tensor - self.mean), self.std)
        rgb_tensor = rgb_tensor[None, :, :, :]

        # Run inference
        start_time = time.time()
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.model.inference(
                {"input": rgb_tensor}
            )
        elapsed = time.time() - start_time
        print(f"Inference time for metric3d: {elapsed:.3f} seconds")

        # Process the depth output
        pred_depth = pred_depth.squeeze()

        # Remove padding
        pred_depth = pred_depth[
            pad_info[0] : pred_depth.shape[0] - pad_info[1],
            pad_info[2] : pred_depth.shape[1] - pad_info[3],
        ]

        # Resize to original image dimensions
        pred_depth = F.interpolate(
            pred_depth[None, None, :, :], (h, w), mode="bilinear"
        ).squeeze()

        # Apply scaling to get metric depth
        canonical_to_real_scale = (
            scaled_intrinsic[0] / 1000.0
        )  # 1000.0 is canonical camera focal length
        pred_depth = pred_depth * canonical_to_real_scale

        # Clip to reasonable depth range
        pred_depth = torch.clamp(pred_depth, 0, 300)

        # Convert to numpy for consistency with other model outputs
        depth_map = pred_depth.cpu().numpy()

        return depth_map, original_image

    def process_images(self, paths, caching=False):
        """
        Process a list of image paths and compute depth maps.
        Args:
            paths (list): List of image file paths
            caching (bool): Whether to cache results
        Returns:
            tuple: (all_depth, all_orig) containing depth maps and original images
        """
        all_depth = []
        all_orig = []
        
        # Create a unique identifier for this sequence
        if caching:
            # Extract sequence folder path for creating a unique cache directory
            sequence_dir = os.path.dirname(paths[0])
            sequence_name = os.path.basename(sequence_dir)
            cache_base_dir = os.path.join("saved_values", sequence_name, self.model_name)
            # Ensure the cache directory exists
            os.makedirs(cache_base_dir, exist_ok=True)
        
        for i, path in enumerate(paths):
            print(f"{i} / {len(paths)}")
            
            if caching:
                # Create a unique cache filename for this specific frame
                frame_name = os.path.basename(path).replace(".", "_")
                cache_file = os.path.join(cache_base_dir, f"frame_{frame_name}")
                
                # Check if cached result exists
                cached_data = generic_helper.load_data(cache_file, use_full_path=True)
                # print(cached_data)
                if cached_data:
                    print(f"Using cached result for frame {i}")
                    depth, original_image = cached_data
                    all_depth.append(depth)
                    all_orig.append(original_image)
                    continue
            
            # Handle different models differently
            if self.model_name == "depth_pro":
                print(path)
                original_image = np.asarray(Image.open(path))
                image, _, f_px = depth_pro.load_rgb(path)
                image = self.transform(image).to(self.device)
                prediction = self.run_inference_half(image, f_px, label="img")
                depth = prediction["depth"].cpu().numpy()  # Convert depth tensor to numpy array
                pred_focal_length = prediction["focallength_px"].cpu().numpy()
            
            elif self.model_name == "depth_anything_v2":
                # For depth_anything_v2, we use cv2 to read the image
                original_image = cv2.imread(path)
                # Use the infer_image method from depth_anything_v2
                start_time = time.time()
                depth = self.model.infer_image(original_image, self.input_size)
                elapsed = time.time() - start_time
                print(f"Inference time for depth_anything_v2: {elapsed:.3f} seconds")
            
            elif self.model_name == "metric3d":
                # Process with Metric3D model
                depth, original_image = self.process_metric3d_image(path)
            
            all_orig.append(original_image)
            all_depth.append(depth)
            
            # Save this individual frame if caching is enabled
            if caching:
                print(cache_file)
                generic_helper.save_data(cache_file, depth, original_image, use_full_path=True)
        
        return all_depth, all_orig

    @staticmethod
    def compute_full_depth_errors(true_depth, predicted_depth, clip_value=655.35):
        """
        Compute absolute error for the full depth map.

        Args:
            true_depth (np.ndarray): Ground truth depth map
            predicted_depth (np.ndarray): Predicted depth map
            clip_value (float): Maximum depth value to consider

        Returns:
            np.ndarray: Absolute errors
        """
        p_depth = predicted_depth.copy()
        p_depth[p_depth > clip_value] = clip_value
        full_errors = np.abs(true_depth - p_depth)
        return full_errors

    def run_with_caching(self, paths, override_run=False):
        """
        Run depth estimation with caching of results.

        Args:
            paths (list): List of image file paths
            override_run (bool): Force recomputation even if cached results exist

        Returns:
            tuple: (all_depth, all_orig) containing depth maps and original images
        """
        save_file_path = paths[0] + str(len(paths)) + self.model_name
        result = generic_helper.load_data(save_file_path)

        if result == False or override_run:
            new_result = self.process_images(paths, caching=not override_run)
            generic_helper.save_data(save_file_path, *new_result)
            return new_result

        return result


class DepthSim:
    def __init__(self, model_name, inference_time):
        self.dmw=DepthModelWrapper(model_name=model_name, load_weights=False)
        self.inference_time = inference_time

    def process_images(self, paths, index=0):
        print("sim depth timer...")
        time.sleep(self.inference_time) # This assumes the data is already cached
        all_depth, all_orig = self.dmw.process_images(paths, caching=True)
        return all_depth[index], all_orig[index]


# Example usage:
# Basic usage with depth_pro
# depth_model = DepthModelWrapper(model_name="depth_pro")

# Using depth_anything_v2 with a custom encoder
# depth_model = DepthModelWrapper(model_name="depth_anything_v2", encoder="vitl")

# Using metric3d with a specific variant
# depth_model = DepthModelWrapper(model_name="metric3d", metric3d_variant="vit_large")

# depth_maps, original_images = depth_model.run_with_caching(image_paths)

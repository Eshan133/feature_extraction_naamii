import torch
import torch.nn as nn
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys
from scipy.ndimage import zoom

def extract_features(model, volume, device):
    """
    Extract features from the last, third-last, and fifth-last Conv3d layers in denseblock4 of the 3D DenseNet121 model,
    and apply global average pooling to produce fixed N-dimensional feature vectors.
    
    Args:
        model (nn.Module): Initialized 3D DenseNet121 model.
        volume (np.ndarray): 3D volume (e.g., tibia or femur region) of shape (depth, height, width).
        device (str): Device to run the model on ('cuda' or 'cpu').
    
    Returns:
        dict: Dictionary containing N-dimensional feature vectors after global average pooling.
    """
    try:
        logging.info("Starting feature extraction...")

        # Validate input volume
        if not isinstance(volume, np.ndarray) or volume.ndim != 3:
            raise ValueError("Input volume must be a 3D NumPy array")
        logging.info(f"tibia_volume shape: {volume.shape}")

        # Downsample if volume is too large (e.g., >256 in any dimension)
        if any(dim > 256 for dim in volume.shape):
            scale = min(256 / max(volume.shape), 1.0)
            volume = zoom(volume, (scale, scale, scale), order=1)
            logging.info(f"Downsampled volume shape: {volume.shape}")

        # Convert volume to tensor and add batch/channel dimensions
        volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        logging.info(f"Input volume tensor shape: {volume_tensor.shape}")

        # Set model to evaluation mode
        model.eval()

        # Store features from target layers
        features = {}

        # Function to recursively find Conv3d layers and their names
        def get_conv3d_layers(module, conv_layers=None, prefix=''):
            if conv_layers is None:
                conv_layers = []
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Conv3d):
                    conv_layers.append(full_name)
                elif isinstance(child, (nn.Sequential, nn.Module)):
                    get_conv3d_layers(child, conv_layers, full_name)
            return conv_layers

        # Get list of Conv3d layers
        conv_layers = get_conv3d_layers(model.features)
        if len(conv_layers) < 5:
            raise ValueError(f"Not enough Conv3d layers found ({len(conv_layers)})")
        
        # Debug: Print all Conv3d layers
        logging.info(f"Found {len(conv_layers)} Conv3d layers:")
        for i, layer_name in enumerate(conv_layers):
            logging.info(f"  {i+1}: {layer_name}")
        logging.info(f"Last 5 Conv3d layers: {conv_layers[-5:]}")

        # Target layers: denseblock4.denselayer16.conv2, denseblock4.denselayer16.conv1, denseblock4.denselayer15.conv2
        target_layers = [
            conv_layers[-1],  # denseblock4.denselayer16.conv2
            conv_layers[-2],  # denseblock4.denselayer16.conv1
            conv_layers[-3]   # denseblock4.denselayer15.conv2
        ]
        logging.info(f"Target Conv3d layers: {target_layers}")

        # Function to get module by name, handling 'features.' prefix
        def get_module_by_name(module, name):
            for n, m in module.named_modules():
                if n == f"features.{name}":
                    return m
                if n == name:
                    return m
            return None

        # Register hooks to capture outputs
        def hook_fn(layer_name):
            def hook(module, input, output):
                # Apply global average pooling
                pooled = torch.mean(output, dim=[2, 3, 4])  # Average over D, H, W
                features[layer_name] = pooled
            return hook

        hooks = []
        for layer_name in target_layers:
            layer = get_module_by_name(model, layer_name)
            if layer is None:
                logging.error(f"Layer {layer_name} not found in model. Available layers: {conv_layers[-10:]}")
                all_modules = [n for n, _ in model.named_modules()]
                logging.error(f"All module names (last 20): {all_modules[-20:]}")
                raise ValueError(f"Layer {layer_name} not found in model")
            hook = layer.register_forward_hook(hook_fn(layer_name))
            hooks.append(hook)
            logging.info(f"Registered hook for layer: {layer_name}")

        # Run forward pass
        with torch.no_grad():
            _ = model(volume_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Verify features
        for layer_name in target_layers:
            if layer_name not in features:
                raise ValueError(f"Features not captured for layer {layer_name}")
            logging.info(f"Extracted feature vector from {layer_name}: shape {features[layer_name].shape}")

        # Save feature vectors to disk
        for layer_name, feat in features.items():
            np.save(f"features_{layer_name.replace('.', '_')}.npy", feat.cpu().numpy())
            logging.info(f"Saved feature vector for {layer_name} to features_{layer_name.replace('.', '_')}.npy")

        logging.info("Feature extraction and global average pooling completed successfully.")
        return features

    except Exception as e:
        logging.error(f"Error during feature extraction: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    from initialize_densenet121_3d import initialize_model
    import nibabel as nib

    model, device = initialize_model()
    ct_path = "data/3702_left_knee.nii.gz"
    mask_path = "data/original_mask.nii.gz"
    ct_data = nib.load(ct_path).get_fdata()
    mask_data = nib.load(mask_path).get_fdata()
    tibia_volume = (mask_data == 2).astype(np.float32)

    features = extract_features(model, tibia_volume, device)
    for layer_name, feat in features.items():
        print(f"Feature vector from {layer_name}: shape {feat.shape}")

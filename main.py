import torch
import numpy as np
import nibabel as nib
import os
from src.exception import CustomException
from src.logger import logging
from src.components.initialize_densenet121_3d import initialize_model
from src.components.feature_comparison import feature_comparison
from src.components.data_ingestion import load_and_segment_ct
import sys
import matplotlib.pyplot as plt

def main(ct_path, mask_path, output_dir="output"):
    """
    Main function to process CT volume, extract features, compute cosine similarities,
    and save results for Tasks III, IV, and V.
    
    Args:
        ct_path (str): Path to CT volume (.nii.gz).
        mask_path (str): Path to mask file (.nii.gz).
        output_dir (str): Directory to save results (default: 'results').
    
    Returns:
        dict: Cosine similarities for the processed image.
    """
    try:
        logging.info("Starting pipeline execution...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory: {output_dir}")

        # Initialize model
        model, device = initialize_model()
        logging.info(f"Model initialized on device: {device}")

        # Load and segment CT data
        logging.info(f"Loading and segmenting CT: {ct_path}, Mask: {mask_path}")
        tibia_region, femur_region, background_region = load_and_segment_ct(ct_path, mask_path)
        logging.info(f"Tibia region shape: {tibia_region.shape}")
        logging.info(f"Femur region shape: {femur_region.shape}")
        logging.info(f"Background region shape: {background_region.shape}")

        # Inspect mask for diagnostics
        mask_data = nib.load(mask_path).get_fdata()
        unique_mask_values = np.unique(mask_data)
        tibia_voxels = np.sum(mask_data == 2)
        femur_voxels = np.sum(mask_data == 1)
        background_voxels = np.sum(mask_data == 0)
        overlap = np.any((mask_data == 2) & (mask_data == 1))
        logging.info(f"Unique mask values: {unique_mask_values}")
        logging.info(f"Tibia voxel count (mask=2): {tibia_voxels}")
        logging.info(f"Femur voxel count (mask=1): {femur_voxels}")
        logging.info(f"Background voxel count (mask=0): {background_voxels}")
        logging.info(f"Overlap (Tibia & Femur): {overlap}")
        if overlap:
            raise ValueError("Tibia and femur masks overlap, check mask file.")
        if not (2 in unique_mask_values and 1 in unique_mask_values and 0 in unique_mask_values):
            raise ValueError(f"Expected mask values [0, 1, 2], got {unique_mask_values}")

        # Visualize regions for debugging
        slice_idx = tibia_region.shape[0] // 2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(tibia_region[slice_idx], cmap='gray'); axes[0].set_title('Tibia')
        axes[1].imshow(femur_region[slice_idx], cmap='gray'); axes[1].set_title('Femur')
        axes[2].imshow(background_region[slice_idx], cmap='gray'); axes[2].set_title('Background')
        plt.savefig(os.path.join(output_dir, "region_slices.png"))
        plt.close()
        logging.info("Saved region slices to region_slices.png")

        # Get image ID
        image_id = os.path.basename(ct_path).replace(".nii.gz", "")
        logging.info(f"Processing image ID: {image_id}")

        # Compute features and similarities
        similarities = feature_comparison(
            tibia_region,
            femur_region,
            background_region,
            model,
            device,
            image_id=image_id
        )

        # Save similarities to CSV in output directory
        csv_path = os.path.join(output_dir, "cosine_similarities.csv")
        logging.info(f"Cosine similarities saved to {csv_path}")

        # Log results
        for pair_name, layer_sims in similarities.items():
            logging.info(f"Similarities for {pair_name}:")
            for layer_name, sim in layer_sims.items():
                logging.info(f"  {layer_name}: {sim}")

        logging.info("Pipeline execution completed successfully.")
        return similarities

    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":

    ct_path = "data/3702_left_knee.nii.gz"
    mask_path = "data/original_mask.nii.gz"
    output_dir = "output"
    
    similarities = main(ct_path, mask_path, output_dir)
    

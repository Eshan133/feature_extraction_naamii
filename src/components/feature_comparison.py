import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import sys
from src.components.extract_features import extract_features
from src.components.initialize_densenet121_3d import initialize_model
import nibabel as nib

def compute_cosine_similarity(feature1, feature2):
    """
    Compute cosine similarity between two feature vectors.
    
    Args:
        feature1 (Tensor): First feature vector.
        feature2 (Tensor): Second feature vector.
    
    Returns:
        float: Cosine similarity score.
    """
    feature1 = feature1.flatten()
    feature2 = feature2.flatten()
    if torch.norm(feature1) == 0 or torch.norm(feature2) == 0:
        return 0.0
    cos_sim = torch.nn.functional.cosine_similarity(feature1.unsqueeze(0), feature2.unsqueeze(0), dim=1).item()
    return cos_sim

def save_similarities_to_csv(similarities, output_file="output/cosine_similarities.csv"):
    """
    Save cosine similarities to a CSV file.
    
    Args:
        similarities (dict): Dictionary of similarities for each pair and layer.
        output_file (str): Path to output CSV file.
    """
    try:
        # Prepare data for CSV
        layers = ['denselayer15.conv2', 'denselayer16.conv1', 'denselayer16.conv2']
        pairs = list(similarities.keys())
        data = {
            'Pair': pairs,
            layers[0]: [similarities[pair]['denseblock4.denselayer15.conv2'] for pair in pairs],
            layers[1]: [similarities[pair]['denseblock4.denselayer16.conv1'] for pair in pairs],
            layers[2]: [similarities[pair]['denseblock4.denselayer16.conv2'] for pair in pairs]
        }
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        logging.info(f"Saved cosine similarities to {output_file}")
    except Exception as e:
        logging.error(f"Error saving CSV: {str(e)}")
        raise CustomException(e, sys)

def feature_comparison(tibia_volume, femur_volume, background_volume, model, device, image_id="3702_left_knee"):
    """
    Compute cosine similarity between feature vectors of tibia, femur, and background regions.
    
    Args:
        tibia_volume (np.ndarray): Volume for tibia region.
        femur_volume (np.ndarray): Volume for femur region.
        background_volume (np.ndarray): Volume for background region.
        model (nn.Module): 3D DenseNet121 model.
        device (str): Device ('cuda' or 'cpu').
        image_id (str): Identifier for the image (default: CT filename).
    
    Returns:
        dict: Dictionary of cosine similarities for each pair and layer.
    """
    try:
        logging.info(f"Starting Task 4 & 5: Feature comparison for image {image_id}...")

        # Extract features for each region
        logging.info("Extracting features for tibia...")
        tibia_features = extract_features(model, tibia_volume, device)
        logging.info("Extracting features for femur...")
        femur_features = extract_features(model, femur_volume, device)
        logging.info("Extracting features for background...")
        background_features = extract_features(model, background_volume, device)

        # Define pairs for comparison
        pairs = [
            ("Tibia-Femur", tibia_features, femur_features),
            ("Tibia-Background", tibia_features, background_features),
            ("Femur-Background", femur_features, background_features)
        ]

        # Compute cosine similarities
        similarities = {}
        for pair_name, feat1, feat2 in pairs:
            similarities[pair_name] = {}
            for layer_name in feat1:
                cos_sim = compute_cosine_similarity(feat1[layer_name], feat2[layer_name])
                similarities[pair_name][layer_name] = cos_sim
                logging.info(f"Cosine similarity for {pair_name}, {layer_name}: {cos_sim}")

        # Save similarities to CSV
        save_similarities_to_csv(similarities)

        # Save similarities as .npy for compatibility
        np.save(f"cosine_similarities_{image_id}.npy", similarities)
        logging.info(f"Saved cosine similarities to cosine_similarities_{image_id}.npy")

        return similarities

    except Exception as e:
        logging.error(f"Error in Task 4 & 5: {str(e)}")
        raise CustomException(e, sys)

# if __name__ == "__main__":
#     model, device = initialize_model()
#     ct_path = "data/3702_left_knee.nii.gz"
#     mask_path = "data/original_mask.nii.gz"
#     ct_data = nib.load(ct_path).get_fdata()
#     mask_data = nib.load(mask_path).get_fdata()

#     # Extract volumes for each region
#     tibia_volume = (mask_data == 2).astype(np.float32)  # Tibia mask value = 2
#     femur_volume = (mask_data == 1).astype(np.float32)  # Femur mask value = 1
#     background_volume = (mask_data == 0).astype(np.float32)  # Background mask value = 0

#     similarities = feature_comparison(tibia_volume, femur_volume, background_volume, model, device)
#     for pair_name, layer_sims in similarities.items():
#         print(f"Similarities for {pair_name}:")
#         for layer_name, sim in layer_sims.items():
#             print(f"  {layer_name}: {sim}")
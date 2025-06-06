import nibabel as nib
import numpy as np
from src.exception import CustomException
from src.logger import logging
import sys

def load_and_segment_ct(ct_path, mask_path):
    try:
        # Load CT scan and mask
        ct_img = nib.load(ct_path).get_fdata()
        mask_img = nib.load(mask_path).get_fdata()
        
        # Account for swapped labels: tibia = 2, femur = 1
        tibia_mask = (mask_img == 2).astype(np.float32)  # Tibia (value=2)
        femur_mask = (mask_img == 1).astype(np.float32)  # Femur (value=1)
        background_mask = (mask_img == 0).astype(np.float32)  # Background

        # Apply masks to CT scan
        tibia_region = ct_img * tibia_mask
        femur_region = ct_img * femur_mask
        background_region = ct_img * background_mask

        return tibia_region, femur_region, background_region
    
    except Exception as e:
        logging.info("Error during Data Ingestion")
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    tibia_region, femur_region, background_region = load_and_segment_ct('../../data/','../../data/')
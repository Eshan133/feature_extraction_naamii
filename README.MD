
# CT Volume Feature Analysis Pipeline

This pipeline processes CT volumes to extract features from tibia, femur, and background regions, compute cosine similarities, and save results in a CSV file. It uses a 3D DenseNet121 model to analyze CT scans (e.g., `3702_left_knee.nii.gz`) with masks (`original_mask.nii.gz`).

## Features

- **Feature Extraction**: Extracts features from `denseblock3` layers of 3D DenseNet121 for tibia (mask=2), femur (mask=1), and background (mask=0).
- **Cosine Similarity**: Computes similarities for Tibia ↔ Femur, Tibia ↔ Background, and Femur ↔ Background.
- **CSV Output**: Saves similarities with rows for region pairs and columns for layers (`denselayer24.conv2`, `denselayer24.conv1`, `denselayer23.conv2`).

## Prerequisites

- Python 3.8+
- ipykernel
- nibabel
- numpy
- torch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- pandas
- Libraries:
  
- Data: CT volume (`3702_left_knee.nii.gz`) and mask (`original_mask.nii.gz`) in `data/`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Eshan133/feature_extraction_naamii.git
   cd feature_extraction_naamii
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
  - This will also trigger setup.py file. Making the project modular

3. Verify data:
   - Place `3702_left_knee.nii.gz` and `original_mask.nii.gz` in `data/`.
   - Ensure mask values: tibia=2, femur=1, background=0.

## Project Structure

```
├── data/
│   ├── 3702_left_knee.nii.gz
│   └── original_mask.nii.gz
├── notebook/
│   ├── Convert_2D_to_3D.ipynb
│   └── EDA.ipynb
│   └── pipeline.ipynb
├── output/                            
│   ├── cosine_similarities.csv
│   └── region_slices.png
├── src/
│   ├── components/
│   │   ├── data_ingestion.py           
│   │   ├── extract_features.py         
│   │   ├── feature_comparison.py      
│   │   └── initialize_densenet121_3d.py 
|   |── exception.py 
|   └── logger.py   
├── main.py                         
├── README.md                        
├── requirements.txt                         
└── setup.py
```

## Usage

**Run the pipeline**:
   ```bash
   python main.py
   ```
   Processes a single CT volume, saving results in `output/`.


## Outputs

- **Cosine Similarities**:
  - `output/cosine_similarities.csv`:
    ```
    Pair,denselayer24.conv2,denselayer24.conv1,denselayer23.conv2
    Tibia-Femur,<value>,<value>,<value>
    Tibia-Background,<value>,<value>,<value>
    Femur-Background,<value>,<value>,<value>
    ```

- **Visualization**:
  - `output/region_slices.png`: Tibia, femur, background slices.

- **Logs**: Mask diagnostics, feature shapes, similarities.

- **File Path Errors**:
  - Verify `data/` paths in `output.py`.

- **Memory Issues**:
  - Adjust downsampling in `extract_features.py`:
    ```python
    if any(dim > 128 for dim in volume.shape):
        scale = min(128 / max(volume.shape), 1.0)
    ```

## Future Enhancements

- Fine-tune 3D DenseNet121 on CT data.
- Automate mask pairing for multiple volumes.
- Add similarity analysis (e.g., thresholding).


## Contact
- ishan.maharjan5@gmail.com

---
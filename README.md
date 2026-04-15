# Land Use Land Cover (LULC) Classification - Japan

This repository contains a full end-to-end deep learning pipeline for mapping Land Use and Land Cover (LULC) in Japan (specifically the Kyoto region) using USGS Landsat 9 imagery, Google Earth Engine, and `TensorFlow`/`Keras`.

It utilizes a Tunable CNN (Convolutional Neural Network) architecture to classify 16x16 geographical patches into 11 discrete ESRI LULC classes.

## Features

- **Automated Data Acquisition**: Pulls cloud-free Landsat 9 data locally via the USGS EarthExplorer (M2M API) and stacks it with high-resolution 10m ESRI LULC ground-truth via Google Earth Engine (`geemap`).
- **Feature Engineering**: Calculates and appends continuous Normalized Difference Vegetation/Built-up Index (NDVI/NDBI) layers.
- **Geospatial Patching**: Extracts 16x16 geographical bounding arrays out of stacked datasets mapped by `rasterio`.
- **Class Balancing**: Undersamples dominant classes (like Trees) to prevent extreme misclassification priors within the model.
- **Tunable Deep Learning**: Keras Tuner automatically locates the best parameters (learning rate and Conv2D filters) for a custom 2-block Convolutional CNN with Dropout layers.
- **Full Evaluation metrics**: Jaccard index (IoU) and plotted Confusion Matrix.

## Workflow

The project is structured linearly into 4 interactive Jupyter Notebooks:

### 1. `01_Data_Acquisition.ipynb`
Handles USGS authentication and Landsat Scene downloads. Overlays a 50 km² AOI around the target city, crops the massive Landsat payload, calculates NDVI/NDBI, and visualizes the stacked rasters.

### 2. `02_Patch_Extraction.ipynb`
Extracts local TIF slices into array mappings (`.npy`) and builds physically representative 16x16 datasets to pump into the Keras CNN.

### 3. `03_Train_Tunable_CNN.ipynb`
Constructs the actual DL Model and uses `keras_tuner.RandomSearch` to discover the mathematically optimum weights over multiple epochs.

### 4. `04_Evaluation.ipynb`
Draws the `sklearn` Confusion Matrix against unseen test areas to visually document overall classification precision across the 11 classes.

## Setup & Installation

### Option 1: Automated Script (Windows)

A `setup.ps1` PowerShell script is included. By running:
```bash
.\setup.ps1
```
It will automatically create a Python virtual environment (`.venv`), activate it, and download all necessary packages via `pip`.

### Option 2: Manual Installation

Requires Python 3.8+. Create an environment and install the dependencies:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # (On Windows)
source .venv/bin/activate    # (On Linux/Mac)

pip install earthengine-api geemap rasterio tensorflow scikit-learn matplotlib keras-tuner jupyter landsatxplore
```

## Requirements
- **USGS EarthExplorer Account** to fetch raw multiband Landsat composites via API.
- **Google Account (GCP)** to authorize `geemap` usage dynamically in the Jupyter backend.

## Authors
Generated for the TEB-Project repository.

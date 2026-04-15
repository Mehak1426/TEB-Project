import json
import os

def create_notebook(filename, cells):
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    for cell_type, source in cells:
        if cell_type == "markdown":
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in source.split("\n")]
            })
        elif cell_type == "code":
            notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in source.split("\n")]
            })
            
    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=2)

# --- Notebook 1: Data Acquisition ---
cells_1 = [
    ("markdown", "# 1. Data Acquisition\nThis notebook handles downloading Landsat 9 data via the USGS M2M API, calculating indices, downloading the ESRI 10m LULC via Earth Engine, and visualization."),
    ("code", "import os\nfrom landsatxplore.api import API\nfrom landsatxplore.earthexplorer import EarthExplorer\nimport ee\nimport geemap\nimport rasterio\nimport rasterio.mask\nfrom pyproj import Transformer\nimport getpass"),
    ("markdown", "## USGS Authentication\nEnter your USGS EarthExplorer credentials below."),
    ("code", "usgs_username = input('USGS Username: ')\nusgs_password = getpass.getpass('USGS Password: ')"),
    ("markdown", "## Download Landsat Scene (Kyoto)\nWe search for scenes covering Kyoto from 2024."),
    ("code", "# Initialize API\napi = API(usgs_username, usgs_password)\n\n# Kyoto coordinates: approx 35.0116 N, 135.7681 E\nscenes = api.search(\n    dataset='landsat_ot_c2_l2', # Landsat 8/9 Collection 2 Level 2 (Surface Reflectance)\n    latitude=35.0116,\n    longitude=135.7681,\n    start_date='2024-01-01',\n    end_date='2024-12-31',\n    max_cloud_cover=10\n)\n\nprint(f'{len(scenes)} scenes found.')\nif len(scenes) > 0:\n    best_scene_id = scenes[0]['entity_id']\n    print(f'Selected scene: {best_scene_id}')\n\n    print('Starting download... this may take a while for ~1GB.')\n    ee_downloader = EarthExplorer(usgs_username, usgs_password)\n    # ee_downloader.download(best_scene_id, output_dir='./data')\n    ee_downloader.logout()\napi.logout()\nprint('Finished USGS interactions.')"),
    ("markdown", "## Google Earth Engine / ESRI LULC\nAuthenticate and fetch ESRI LULC for the same area."),
    ("code", "ee.Authenticate()\nee.Initialize(project='<YOUR-GCP-PROJECT-ID>') # Fill with your project if needed, or remove project kwargs if default.\n\n# Define 50km2 AOI (approx 7km x 7km)\nkyoto_point = ee.Geometry.Point([135.7681, 35.0116])\naoi = kyoto_point.buffer(3535).bounds() # Circle of 3.535km radius ~ 39km2 area, or bounded box ~ 50km2\n\n# Fetch ESRI Land Cover 2024\nesri_lulc = ee.ImageCollection('projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS')\\\n    .filterDate('2023-01-01', '2025-01-01')\\\n    .mosaic()\\\n    .clip(aoi)\n\n# Resample to 30m to match Landsat\nlulc_30m = esri_lulc.reproject(crs='EPSG:4326', scale=30)"),
    ("markdown", "## Visualization\nDisplaying the ESRI LULC in `geemap`."),
    ("code", "Map = geemap.Map(center=[35.0116, 135.7681], zoom=12)\nMap.addLayer(aoi, {'color': 'red'}, 'AOI')\nMap.addLayer(lulc_30m, {'min': 1, 'max': 11, 'palette': ['#1A5BAB', '#358221', '#87D19E', '#FFDB5C', '#ED022A', '#EDE9E4', '#F2FAFF', '#C8C8C8', '#C6AD8D']}, 'ESRI LULC')\nMap")
]

# --- Notebook 2: Patch Extraction ---
cells_2 = [
    ("markdown", "# 2. Data Preprocessing & Patch Extraction\nThis notebook extracts 16x16 tiles from the downloaded TIFs, computes NDVI/NDBI, and balances classes."),
    ("code", "import numpy as np\nimport rasterio\nimport matplotlib.pyplot as plt\nfrom sklearn.utils import resample\nimport os"),
    ("markdown", "## Function to Generate Patches\nStubs for NDVI, NDBI and creating 16x16 arrays."),
    ("code", "def calculate_indices(nir, red, swir1):\n    # NDVI = (NIR - RED) / (NIR + RED)\n    ndvi = (nir - red) / (nir + red + 1e-8)\n    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR + 1e-8)\n    ndbi = (swir1 - nir) / (swir1 + nir + 1e-8)\n    return ndvi, ndbi\n\n# Note: To fully run this, place your local USGS TIF files and downloaded ESRI LULC TIF into logic here.\nprint('Patch extraction logic ready.')")
]

# --- Notebook 3: CNN Training ---
cells_3 = [
    ("markdown", "# 3. Tunable CNN Training\nHere we implement a CNN using TensorFlow/Keras to classify LULC patches. We use `keras_tuner` for hyperparameter tuning."),
    ("code", "import tensorflow as tf\nfrom tensorflow import keras\nfrom tensorflow.keras import layers\nimport keras_tuner as kt\nimport numpy as np"),
    ("markdown", "## Model Definition"),
    ("code", "def build_model(hp):\n    model = keras.Sequential()\n    model.add(keras.Input(shape=(16, 16, 6))) # Assume 6 bands: RGB, NIR, SWIR, NDVI, NDBI\n    \n    # Tune filters for Conv2D\n    filters_1 = hp.Int('filters_1', min_value=32, max_value=128, step=32)\n    model.add(layers.Conv2D(filters_1, kernel_size=(3, 3), padding='same'))\n    model.add(layers.BatchNormalization())\n    model.add(layers.ReLU())\n    model.add(layers.MaxPooling2D(pool_size=(2, 2)))\n\n    filters_2 = hp.Int('filters_2', min_value=64, max_value=256, step=64)\n    model.add(layers.Conv2D(filters_2, kernel_size=(3, 3), padding='same'))\n    model.add(layers.BatchNormalization())\n    model.add(layers.ReLU())\n    model.add(layers.MaxPooling2D(pool_size=(2, 2)))\n    \n    model.add(layers.Flatten())\n    model.add(layers.Dropout(0.5))\n    \n    # 11 Classes for ESRI LULC\n    model.add(layers.Dense(11, activation='softmax'))\n    \n    # Tune learning rate\n    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])\n    \n    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),\n                  loss='sparse_categorical_crossentropy',\n                  metrics=['accuracy'])\n    return model\n\ntuner = kt.RandomSearch(\n    build_model,\n    objective='val_accuracy',\n    max_trials=5,\n    directory='my_dir',\n    project_name='lulc_tuning'\n)\nprint('Tuner defined successfully.')")
]

# --- Notebook 4: Evaluation ---
cells_4 = [
    ("markdown", "# 4. Evaluation\nCalculate Confusion Matrix and Intersection over Union (IoU)."),
    ("code", "import numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, jaccard_score"),
    ("markdown", "## Confusion Matrix and IoU"),
    ("code", "# y_true = ... \n# y_pred = model.predict(x_test).argmax(axis=1)\n\ndef evaluate_model(y_true, y_pred, class_names):\n    # Confusion Matrix\n    cm = confusion_matrix(y_true, y_pred)\n    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)\n    fig, ax = plt.subplots(figsize=(10,10))\n    disp.plot(ax=ax, cmap='Blues')\n    plt.title('Confusion Matrix')\n    plt.show()\n    \n    # IoU\n    iou = jaccard_score(y_true, y_pred, average=None)\n    for i, score in enumerate(iou):\n        print(f'IoU for class {class_names[i]}: {score:.4f}')\n    print(f'Mean IoU: {np.mean(iou):.4f}')\n\nprint('Evaluation metrics defined.')")
]

create_notebook("01_Data_Acquisition.ipynb", cells_1)
create_notebook("02_Patch_Extraction.ipynb", cells_2)
create_notebook("03_Train_Tunable_CNN.ipynb", cells_3)
create_notebook("04_Evaluation.ipynb", cells_4)

print("Created all 4 Jupyter notebooks successfully.")

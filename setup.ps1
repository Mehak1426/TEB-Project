# Setup script for LULC Classification Project

# Create virtual environment if it doesn't exist
if (-Not (Test-Path ".venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..."
pip install earthengine-api geemap rasterio tensorflow scikit-learn matplotlib keras-tuner jupyter notebook landsatxplore pyproj geopandas

# Initialize Git Repository
Write-Host "Configuring Git..."
if (-Not (Test-Path ".git")) {
    git init
    git remote add origin https://github.com/Mehak1426/TEB-Project
    Write-Host "Initialized local Git repository and added remote origin."
} else {
    Write-Host "Git repository already initialized."
}

Write-Host "Creating notebooks..."
python generate_notebooks.py

Write-Host "Setup complete. To activate the environment, run: .venv\Scripts\Activate.ps1"

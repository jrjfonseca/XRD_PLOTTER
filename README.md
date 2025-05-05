# XRD Plotter

A Streamlit application for visualizing and analyzing XRD (X-ray Diffraction) data.

## Features

- Load multiple XRD data files
- Normalize spectra
- Customize labels and colors
- Adjust Y-offset for better visualization
- Apply smoothing to the data
- Export processed data
- Interactive plot with legend

## Installation

1. Create the conda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate xrd_plotter
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload your XRD data files (supported formats: .txt, .csv, .dat)

4. Use the controls to:
   - Normalize the data
   - Add custom labels
   - Adjust Y-offset
   - Change colors
   - Apply smoothing
   - Export processed data

## Data Format

The application accepts XRD data in various formats:
- CSV files with two columns (2θ and intensity)
- Text files with space or tab-separated values
- Files with two columns of numerical data

## Requirements

All required packages are listed in the `environment.yml` file and will be installed automatically when creating the conda environment. 
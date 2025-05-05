# XRD Plotter

A Streamlit application for visualizing and analyzing XRD (X-ray Diffraction) data.

## Features

- Load multiple XRD data files
- Normalize spectra
- Customize labels and colors
- Position labels directly on the plot above spectral lines
- Adjust Y-offset for better visualization
- Control the 2θ degree range to display
- Custom legend positioning
- Publication-quality plotting using SciencePlots
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

3. Upload your XRD data files (supported formats: .txt, .csv, .dat, .xy)

4. Use the controls to:
   - Normalize the data
   - Add custom labels
   - Adjust Y-offset
   - Change colors
   - Apply smoothing
   - Position labels directly on the plot with custom alignment
   - Export processed data

5. Use the sidebar to:
   - Set custom 2θ range for the plot
   - Position the legend anywhere in the plot
   - Enable publication-quality plots using SciencePlots
   - Select different scientific plotting styles

6. Additional features:
   - Save high-resolution figures in PNG and PDF formats
   - Export processed data as CSV files

## Data Format

The application accepts XRD data in various formats:
- .xy files (standard XRD data format, typically space or tab-separated values)
- CSV files with two columns (2θ and intensity)
- Text files with space or tab-separated values
- Files with two columns of numerical data

The application will automatically detect and parse the file format, ignoring comment lines (starting with '#') and handling different separators.

## Requirements

All required packages are listed in the `environment.yml` file and will be installed automatically when creating the conda environment. 
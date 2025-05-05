#!/bin/bash
# Script to install dependencies for Streamlit app

echo "Installing dependencies for XRD Plotter..."
pip install --upgrade pip
pip install matplotlib scipy numpy pandas scienceplots xrayutilities
echo "Installation complete!" 
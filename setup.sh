#!/bin/bash
# Setup script for XRD Plotter dependencies

echo "Setting up dependencies for XRD Plotter..."

# Check if running on Linux and install system dependencies
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  echo "Detected Linux system, installing system dependencies..."
  sudo apt-get update
  sudo apt-get install -y build-essential libcairo2-dev pkg-config python3-dev libfreetype6-dev libpng-dev libgl1-mesa-glx libglib2.0-0
fi

# Upgrade pip and install Python dependencies
echo "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete! You can now run the application with:"
echo "streamlit run streamlit_app.py" 
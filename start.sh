#!/bin/bash
set -e

echo "Starting deployment process..."

# Check Python version
python_version=$(python3 --version)
echo "Using Python: $python_version"

# Check pip
pip_version=$(pip --version)
echo "Using pip: $pip_version"

# Verify installation of key packages
echo "Verifying installed packages..."
pip list | grep streamlit
pip list | grep plotly
pip list | grep numpy
pip list | grep pandas

# Set up streamlit config
echo "Configuring Streamlit..."
mkdir -p ~/.streamlit
echo "[server]" > ~/.streamlit/config.toml
echo "headless = true" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml
echo "enableStaticServing = true" >> ~/.streamlit/config.toml

# Define environment variables
export STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true
export STREAMLIT_SERVER_HEADLESS=true

# Get port from environment or use default
PORT=${PORT:-8501}
echo "Port set to: $PORT"

# Start Streamlit with explicit arguments
echo "Starting Streamlit application..."
streamlit run streamlit_app.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.enableStaticServing=true \
  --server.headless=true 
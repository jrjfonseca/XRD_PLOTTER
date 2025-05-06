#!/bin/bash
set -e

# Set up streamlit config
mkdir -p ~/.streamlit
echo "[server]" > ~/.streamlit/config.toml
echo "headless = true" >> ~/.streamlit/config.toml
echo "enableCORS = false" >> ~/.streamlit/config.toml
echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml
echo "enableStaticServing = true" >> ~/.streamlit/config.toml

# Get port from environment or use default
PORT=${PORT:-8501}

# Start Streamlit
echo "Starting Streamlit on port $PORT..."
streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 
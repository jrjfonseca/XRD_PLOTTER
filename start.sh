#!/bin/bash
# Special startup script for Streamlit Cloud

# Set environment variables for Streamlit Cloud
export STREAMLIT_CLOUD=true
export MPLBACKEND=Agg
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Check python version
python --version

# Check if matplotlib is installed and usable
python -c "import matplotlib.pyplot as plt; print('Matplotlib works')" || echo "Matplotlib not working"

# Start the Streamlit application
streamlit run streamlit_app.py "$@" 
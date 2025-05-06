#!/bin/bash
set -e

echo "XRD Plotter - Deployment Initialization"

# Diagnostic information
echo "Environment:"
echo "- Python: $(python3 --version)"
echo "- Working directory: $(pwd)"

# Create streamlit config directory if it doesn't exist
mkdir -p ~/.streamlit

# Create streamlit configuration for production
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
enableCORS = false
enableXsrfProtection = false
enableStaticServing = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#636EFA"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[runner]
magicEnabled = true
fastReruns = true
EOF

# Get port from environment or use default
PORT=${PORT:-10000}
echo "Starting server on port $PORT"

# Start application with production settings
exec streamlit run streamlit_app.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.enableStaticServing=true \
  --server.headless=true 
# XRD Plotter

Interactive X-ray Diffraction (XRD) data visualization tool built with Streamlit.

## Features

- Upload and visualize multiple XRD files
- Normalize and smooth data
- Adjust vertical offsets for better comparison
- Customize colors and labels
- Interactive zoom, pan, and hover
- Export data and interactive plots

## Deployment on Render.com

### Automatic Deployment

1. Fork or clone this repository
2. Sign up for [Render.com](https://render.com/)
3. Create a new Web Service
4. Connect your GitHub repository
5. Render will automatically detect the configuration from `render.yaml`
6. Click "Create Web Service"

### Manual Configuration

If you need to configure the service manually:

- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt && chmod +x start.sh`
- **Start Command**: `./start.sh`
- **Python Version**: 3.9.12

## Local Development

```bash
# Clone the repository
git clone https://github.com/jrjfonseca/XRD_PLOTTER.git
cd XRD_PLOTTER

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## Troubleshooting

If you encounter issues with the deployment:

1. Check the Render logs for specific error messages
2. Ensure all dependencies are properly installed
3. Try deploying with a different Python version if needed
4. For static file issues, ensure the Streamlit configuration is correct

## License

MIT 
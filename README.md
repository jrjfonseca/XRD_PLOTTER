# XRD Plotter

A production-ready, interactive X-ray Diffraction (XRD) data visualization application built with Streamlit and optimized for Render.com deployment.

![XRD Plotter](https://raw.githubusercontent.com/jrjfonseca/XRD_PLOTTER/main/xrd-plotter-screenshot.png)

## Features

- **Data Import**: Upload and visualize multiple XRD files (.txt, .csv, .dat, .xy formats)
- **Interactive Visualization**: Zoomable, pannable plots with tooltips and data exploration
- **Data Processing**: Normalize, smooth, and adjust spectral data
- **Customization**: Modify colors, labels, and offsets for publication-ready visualization
- **Publication-Quality Export**: Generate high-resolution plots using SciencePlots styles
- **Export Options**: Download processed data, interactive HTML plots, or high-DPI PNG images
- **Responsive Design**: Works on mobile, tablet, and desktop browsers

## Live Demo

Try the application live at: [https://xrd-plotter.onrender.com](https://xrd-plotter.onrender.com)

## Quick Deployment on Render.com

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Fork this repository
2. Connect your GitHub account to Render.com
3. Create a new Web Service, selecting your forked repository
4. Render.com will automatically detect the configuration and deploy the app

## Architecture

This application is designed with modern web principles and cloud-native architecture:

- **Frontend/Backend**: Single Streamlit application for both UI and data processing
- **Caching**: Optimized performance with Streamlit caching for data processing
- **Fallbacks**: Graceful degradation when optional dependencies are unavailable
- **Error Handling**: Comprehensive error handling for robustness
- **Configuration**: Clear separation of configuration and code

## Local Development

```bash
# Clone the repository
git clone https://github.com/jrjfonseca/XRD_PLOTTER.git
cd XRD_PLOTTER

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## Production Deployment Details

### Environment Configuration

The application is configured for production deployment through:

1. **render.yaml**: Service definition for Render.com
2. **start.sh**: Production startup script with proper configuration
3. **requirements.txt**: Pinned dependencies for reproducible builds

### Performance Optimizations

- Data caching with `@st.cache_data` decorators
- Lazy-loading of optional dependencies
- Streamlined UI rendering
- Minimal dependencies with exact version specifications

## Troubleshooting Deployment

### Common Issues

If you encounter issues during deployment:

1. **Static Files Not Loading**: Check that the Streamlit configuration has `enableStaticServing = true`
2. **Dependency Errors**: Verify that all dependencies in requirements.txt are compatible
3. **Port Configuration**: Ensure PORT environment variable is set correctly (10000 by default)

### Render.com Logs

Always check the Render.com logs for detailed error information. Common solutions:

- Rebuild without cache if dependencies seem inconsistent
- Verify Python version (3.9.12 recommended)
- Check for memory limitations in the free tier

## License

MIT License

## Contribute

Contributions are welcome! Please feel free to submit a Pull Request. 
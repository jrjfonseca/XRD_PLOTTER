import streamlit as st
import numpy as np
import pandas as pd
import re
import warnings
from io import BytesIO
from typing import Tuple, List, Dict, Optional, Any, Union

# Configure page immediately (must be first Streamlit command)
st.set_page_config(
    page_title="XRD Plotter",
    page_icon="📊",
    layout="wide"
)

# Application constants
CONFIG = {
    "DEFAULT_THETA_MIN": 0,
    "DEFAULT_THETA_MAX": 100,
    "PADDING": 5,
    "SUPPORTED_FORMATS": ['txt', 'csv', 'dat', 'xy'],
    "LABEL_OFFSET_PERCENT": 0.05,  # Position labels 5% from the end of visible range
}

# Safely import optional dependencies
SCIPY_AVAILABLE = False
PLOTLY_AVAILABLE = False

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    warnings.warn("SciPy not available. Smoothing features will be disabled.")
    
    # Define a dummy savgol_filter function for fallback
    def savgol_filter(data, window_length, polyorder, **kwargs):
        """Dummy function when scipy is not available."""
        return data

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
    # Default color palette from Plotly
    CONFIG["COLORS"] = px.colors.qualitative.Plotly
except ImportError:
    warnings.warn("Plotly not available. Using fallback visualization.")
    # Fallback colors if Plotly isn't available
    CONFIG["COLORS"] = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", 
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
    ]

# ===== Data Processing Functions =====

def read_xrd_data(file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse XRD data from various file formats.
    
    Args:
        file: Uploaded file object
    
    Returns:
        Tuple of (x_data, y_data) as numpy arrays, or (None, None) if parsing fails
    """
    if file is None:
        return None, None
        
    file_name = file.name.lower()
    
    try:
        # Handle .xy files (common XRD format)
        if file_name.endswith('.xy'):
            return _parse_xy_file(file)
            
        # Try as CSV
        data = pd.read_csv(file, header=None)
        if data.shape[1] >= 2:
            return data.iloc[:, 0].values, data.iloc[:, 1].values
            
    except Exception as e:
        st.error(f"Error parsing file as CSV: {str(e)}")
        
    # Fall back to general text parsing
    try:
        content = file.getvalue().decode('utf-8')
        return _parse_general_text(content)
    except Exception as e:
        st.error(f"Could not parse file {file.name}: {str(e)}")
        
    return None, None


def _parse_xy_file(file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse space/tab-separated XY file format."""
    try:
        content = file.getvalue().decode('utf-8')
        x_data, y_data = [], []
        
        for line in content.split('\n'):
            if not line.strip() or line.strip().startswith('#'):
                continue
                
            fields = re.split(r'\s+', line.strip())
            if len(fields) >= 2:
                try:
                    x_data.append(float(fields[0]))
                    y_data.append(float(fields[1]))
                except ValueError:
                    continue
                    
        if not x_data:
            return None, None
            
        return np.array(x_data), np.array(y_data)
    except Exception as e:
        st.error(f"Error parsing XY file: {str(e)}")
        return None, None


def _parse_general_text(content: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Attempt to parse text with numeric data in any reasonable format."""
    try:
        x_data, y_data = [], []
        
        for line in content.split('\n'):
            if not line.strip() or line.strip().startswith('#'):
                continue
                
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
            if len(numbers) >= 2:
                x_data.append(float(numbers[0]))
                y_data.append(float(numbers[1]))
                
        if not x_data:
            return None, None
            
        return np.array(x_data), np.array(y_data)
    except Exception as e:
        st.error(f"Error parsing text data: {str(e)}")
        return None, None


def normalize_data(y_data: np.ndarray) -> np.ndarray:
    """Scale data to range [0,1]."""
    if y_data is None or len(y_data) == 0:
        return np.array([])
        
    y_min, y_max = np.min(y_data), np.max(y_data)
    if y_max == y_min:
        return np.zeros_like(y_data)
    return (y_data - y_min) / (y_max - y_min)


def apply_smooth(y_data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply Savitzky-Golay smoothing with error handling."""
    if not SCIPY_AVAILABLE:
        st.warning("Smoothing unavailable - SciPy not installed")
        return y_data
    
    if y_data is None or len(y_data) == 0:
        return np.array([])
        
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
        
    # Handle case where window is larger than data
    if window_size >= len(y_data):
        window_size = min(len(y_data) - (1 if len(y_data) % 2 == 0 else 2), 5)
        if window_size < 3:
            return y_data  # Not enough points to smooth
    
    try:
        return savgol_filter(y_data.copy(), window_size, 2)
    except Exception as e:
        st.warning(f"Smoothing error: {str(e)}")
        return y_data


def find_label_position(x_data, y_data, is_custom_range, min_theta=None, max_theta=None):
    """Find appropriate default position for spectrum label."""
    if x_data is None or y_data is None or not len(x_data) or not len(y_data):
        return 0, 0
        
    if is_custom_range and min_theta is not None and max_theta is not None:
        mask = (x_data >= min_theta) & (x_data <= max_theta)
        x_visible = x_data[mask] if any(mask) else x_data
        y_visible = y_data[mask] if any(mask) else y_data
        
        if len(x_visible) == 0:
            return max_theta, 0
            
        max_x = max_theta - (max_theta - min_theta) * CONFIG["LABEL_OFFSET_PERCENT"]
    else:
        if len(x_data) == 0:
            return 0, 0
            
        x_min = np.min(x_data)
        x_max = np.max(x_data)
        max_x = x_max - (x_max - x_min) * CONFIG["LABEL_OFFSET_PERCENT"]
    
    # Find closest y-value to the chosen x-position
    x_visible = x_data
    y_visible = y_data
    
    if is_custom_range and min_theta is not None and max_theta is not None:
        mask = (x_data >= min_theta) & (x_data <= max_theta)
        if any(mask):
            x_visible = x_data[mask]
            y_visible = y_data[mask]
    
    if len(x_visible) > 0:
        nearest_idx = np.abs(x_visible - max_x).argmin()
        default_y = y_visible[nearest_idx]
        return max_x, default_y
    
    return max_x, 0


def get_data_range(files) -> Tuple[float, float]:
    """Calculate the data range from all uploaded files."""
    min_x, max_x = float('inf'), float('-inf')
    
    for file in files:
        x_data, _ = read_xrd_data(file)
        if x_data is not None and len(x_data) > 0:
            min_x = min(min_x, np.min(x_data))
            max_x = max(max_x, np.max(x_data))
    
    # Set reasonable defaults if no range found
    if min_x == float('inf'):
        min_x, max_x = CONFIG["DEFAULT_THETA_MIN"], CONFIG["DEFAULT_THETA_MAX"]
    
    # Add padding
    min_x = max(0, min_x - CONFIG["PADDING"])
    max_x = max_x + CONFIG["PADDING"]
    
    return min_x, max_x


# ===== Plotting Functions =====

def create_interactive_plot(
    processed_data: List[Dict[str, Any]],
    label_positions: List[Dict[str, Any]],
    plot_config: Dict[str, Any]
) -> Any:
    """Create an interactive Plotly figure with the processed data."""
    if not PLOTLY_AVAILABLE:
        return create_fallback_plot(processed_data, label_positions, plot_config)
        
    # Create the figure
    fig = go.Figure()
    
    # Add each dataset as a trace
    for i, data in enumerate(processed_data):
        # Apply custom range if needed
        if plot_config.get('use_custom_range', False):
            min_theta = plot_config['min_theta']
            max_theta = plot_config['max_theta']
            mask = (data['x'] >= min_theta) & (data['x'] <= max_theta)
            x_plot = data['x'][mask]
            y_plot = data['y'][mask]
        else:
            x_plot = data['x']
            y_plot = data['y']
        
        # Add the line to the plot
        fig.add_trace(go.Scatter(
            x=x_plot, 
            y=y_plot,
            mode='lines',
            name=data['label'],
            line=dict(color=data['color'], width=2)
        ))
    
    # Add custom labels as annotations
    annotations = []
    for label_info in label_positions:
        if label_info.get('show', False):
            annotations.append(dict(
                x=label_info['x'],
                y=label_info['y'],
                text=label_info['text'],
                showarrow=False,
                font=dict(
                    color=label_info['color'],
                    size=12,
                    family="Arial"
                ),
                xanchor='right',
                yanchor='middle'
            ))
    
    # Add the annotations to the figure
    fig.update_layout(annotations=annotations)
    
    # Configure layout
    fig.update_layout(
        title=None,
        xaxis_title='2θ (degrees)',
        yaxis_title='Intensity (a.u.)',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ) if plot_config.get('show_legend', True) else dict(visible=False),
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='closest'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.3)')
    
    # Set axis limits if custom range is specified
    if plot_config.get('use_custom_range', False):
        fig.update_xaxes(range=[plot_config['min_theta'], plot_config['max_theta']])
    
    # Apply theme
    if plot_config.get('use_dark_theme', False):
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(50, 50, 50, 1)',
            paper_bgcolor='rgba(50, 50, 50, 1)',
            font=dict(color='white')
        )
    
    return fig


def create_fallback_plot(
    processed_data: List[Dict[str, Any]],
    label_positions: List[Dict[str, Any]],
    plot_config: Dict[str, Any]
) -> None:
    """Create a fallback visualization using Streamlit's native plotting capabilities."""
    st.warning("Using Streamlit's native plotting as Plotly is not available")
    
    # Process each dataset
    for i, data in enumerate(processed_data):
        # Apply custom range if needed
        if plot_config.get('use_custom_range', False):
            min_theta = plot_config['min_theta']
            max_theta = plot_config['max_theta']
            mask = (data['x'] >= min_theta) & (data['x'] <= max_theta)
            x_plot = data['x'][mask]
            y_plot = data['y'][mask]
        else:
            x_plot = data['x']
            y_plot = data['y']
            
        # Create a dataframe for this dataset
        df = pd.DataFrame({
            '2θ (degrees)': x_plot,
            data['label']: y_plot
        })
        
        # Display using Streamlit's line chart
        st.subheader(f"{data['label']}")
        st.line_chart(df.set_index('2θ (degrees)'), color=data['color'][1:] if len(data['color']) > 1 else None)
    
    # Show label positions in a table if any
    labeled_data = [label for label in label_positions if label.get('show', False)]
    if labeled_data:
        st.subheader("Labels")
        label_df = pd.DataFrame([
            {'Label': l['text'], 'X Position (2θ)': l['x'], 'Y Position': l['y']} 
            for l in labeled_data
        ])
        st.dataframe(label_df)


# ===== UI Components =====

def render_sidebar_controls():
    """Render the sidebar controls and return configuration."""
    plot_config = {}
    
    with st.sidebar:
        st.header("Global Settings")
        
        # Plot range settings
        st.subheader("Plot Range")
        plot_config['use_custom_range'] = st.checkbox("Use custom 2θ range", value=False)
        
        # These values will be set by the caller if needed
        plot_config['min_theta'] = None
        plot_config['max_theta'] = None
        
        # Legend settings
        st.subheader("Display Settings")
        plot_config['show_legend'] = st.checkbox("Show legend", value=True)
        plot_config['use_dark_theme'] = st.checkbox("Dark theme", value=False)
        
        # System status
        st.subheader("System Status")
        st.info(f"SciPy (for smoothing): {'Available' if SCIPY_AVAILABLE else 'Not Available'}")
        st.info(f"Plotly (for interactive plots): {'Available' if PLOTLY_AVAILABLE else 'Not Available'}")
        
        # Help info
        st.markdown("---")
        st.markdown("""
        **Tip:** Use the toolbar in the upper right of the plot to:
        - Zoom in/out
        - Pan
        - Download as PNG
        - Reset view
        """)
    
    return plot_config


def render_file_controls(file, idx, use_custom_range, min_theta=None, max_theta=None):
    """Render controls for a single file and return the processed data."""
    x_data, y_data = read_xrd_data(file)
    
    if x_data is None or y_data is None:
        st.error(f"Could not read data from {file.name}")
        return None, None, None
    
    # Store original data
    original_data = {
        'x': x_data,
        'y': y_data,
        'filename': file.name
    }
    
    file_config = {}
    label_info = {}
    
    # Create controls for this file
    with st.expander(f"Controls for {file.name}", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_config['normalize'] = st.checkbox("Normalize", key=f"norm_{idx}")
        
        with col2:
            file_config['label'] = st.text_input("Label", value=file.name, key=f"label_{idx}")
        
        with col3:
            file_config['y_offset'] = st.number_input(
                "Y-offset", 
                min_value=0.0, 
                max_value=100.0, 
                value=0.0, 
                step=0.5, 
                key=f"offset_{idx}"
            )
        
        # Use a color from the default palette or let user pick
        default_color = CONFIG["COLORS"][idx % len(CONFIG["COLORS"])]
        custom_color = st.color_picker("Color", default_color, key=f"color_{idx}")
        file_config['color'] = custom_color
        
        # Apply smoothing if requested and available
        if SCIPY_AVAILABLE:
            file_config['apply_smooth'] = st.checkbox("Apply smoothing", key=f"smooth_{idx}")
            if file_config['apply_smooth']:
                file_config['window_size'] = st.slider(
                    "Smoothing window", 
                    3, 51, 5, 2, 
                    key=f"window_{idx}"
                )
                y_processed = apply_smooth(y_data.copy(), file_config['window_size'])
            else:
                y_processed = y_data.copy()
        else:
            y_processed = y_data.copy()
        
        # Normalize if requested
        if file_config['normalize']:
            y_processed = normalize_data(y_processed)
        
        # Apply offset
        y_processed = y_processed + file_config['y_offset']
        
        # Label positioning
        st.subheader("Label Position")
        label_info['show'] = st.checkbox("Show label on plot", value=True, key=f"show_label_{idx}")
        
        if label_info['show']:
            # Get default positions
            default_x, default_y = find_label_position(
                x_data, y_processed, use_custom_range, min_theta, max_theta
            )
            
            # Position inputs
            label_col1, label_col2 = st.columns(2)
            
            with label_col1:
                min_x_value = float(min_theta) if use_custom_range else float(np.min(x_data)) if len(x_data) > 0 else 0.0
                max_x_value = float(max_theta) if use_custom_range else float(np.max(x_data)) if len(x_data) > 0 else 100.0
                
                label_info['x'] = st.number_input(
                    "X Position",
                    value=float(default_x),
                    min_value=min_x_value,
                    max_value=max_x_value,
                    key=f"label_x_{idx}"
                )
            
            with label_col2:
                label_info['y'] = st.number_input(
                    "Y Position",
                    value=float(default_y),
                    key=f"label_y_{idx}"
                )
            
            # Store other label information
            label_info['text'] = file_config['label']
            label_info['color'] = file_config['color']
        
        # Create processed data record
        processed_data = {
            'x': x_data,
            'y': y_processed,
            'label': file_config['label'],
            'color': file_config['color']
        }
    
    return original_data, processed_data, label_info


def render_export_controls(processed_data, fig=None):
    """Render export controls for saving data."""
    if st.button("Export Data"):
        for data in processed_data:
            df = pd.DataFrame({
                '2θ': data['x'],
                'Intensity': data['y']
            })
            st.download_button(
                label=f"Download {data['label']}",
                data=df.to_csv(index=False),
                file_name=f"{data['label']}.csv",
                mime="text/csv"
            )
        
        # Offer HTML export of the interactive plot if plotly is available
        if PLOTLY_AVAILABLE and fig is not None:
            try:
                buffer = BytesIO()
                fig.write_html(buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="Download Interactive Plot (HTML)",
                    data=buffer,
                    file_name="xrd_plot.html",
                    mime="text/html"
                )
            except Exception as e:
                st.error(f"Error exporting HTML: {str(e)}")


def render_tutorial():
    """Render the tutorial section."""
    with st.expander("How to use this app", expanded=True):
        st.markdown("""
        ### XRD Plotter Usage Guide
        
        1. **Upload Files**: Use the file uploader to select one or more XRD data files (.txt, .csv, .dat, or .xy formats)
        2. **Adjust Settings**: Each file has its own control panel where you can:
           - Normalize the data
           - Apply smoothing
           - Add a vertical offset
           - Change the color
           - Position labels precisely where you want them
        3. **Global Settings**: Use the sidebar to control:
           - 2θ range to display
           - Legend visibility
           - Dark/light theme
        4. **Interactive Features**: The plot is fully interactive:
           - Hover over data points to see exact values
           - Zoom in/out with mouse wheel or toolbar
           - Pan by dragging
           - Double-click to reset view
           - Click on legend items to hide/show traces
        5. **Export**: Save your processed data or the interactive plot
        
        Try it now by uploading your XRD data files!
        """)


# ===== Main Application =====

def main():
    try:
        st.title("XRD Data Plotter")
        st.write("Upload XRD files to visualize, compare, and analyze X-ray diffraction patterns")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload XRD files", 
            accept_multiple_files=True, 
            type=CONFIG["SUPPORTED_FORMATS"]
        )
        
        if not uploaded_files:
            render_tutorial()
            return
        
        # Render sidebar and get plot configuration
        plot_config = render_sidebar_controls()
        
        # Calculate data range if needed
        if plot_config['use_custom_range']:
            min_x, max_x = get_data_range(uploaded_files)
            
            # Create range inputs
            col1, col2 = st.sidebar.columns(2)
            with col1:
                plot_config['min_theta'] = st.number_input("Min 2θ", value=min_x, min_value=0.0)
            with col2:
                plot_config['max_theta'] = st.number_input(
                    "Max 2θ", 
                    value=max_x, 
                    min_value=plot_config['min_theta'] + 1.0
                )
        
        # Process files
        all_data = []
        all_processed_data = []
        all_label_positions = []
        
        for i, file in enumerate(uploaded_files):
            original_data, processed_data, label_info = render_file_controls(
                file, 
                i, 
                plot_config['use_custom_range'],
                plot_config.get('min_theta'),
                plot_config.get('max_theta')
            )
            
            if original_data and processed_data:
                all_data.append(original_data)
                all_processed_data.append(processed_data)
                all_label_positions.append(label_info)
        
        if not all_processed_data:
            st.error("No valid data files were uploaded. Please check your files.")
            return
        
        # Create interactive figure
        fig = create_interactive_plot(all_processed_data, all_label_positions, plot_config)
        
        # Display plot - different handling for Plotly vs fallback
        if PLOTLY_AVAILABLE:
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
            
            # Export controls
            render_export_controls(all_processed_data, fig)
        else:
            # Fallback mode doesn't return a figure object
            render_export_controls(all_processed_data)
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please report this issue to the repository maintainers.")


# Add footer with attribution
st.markdown("""
---
Made with [Streamlit](https://streamlit.io) • [GitHub Repository](https://github.com/jrjfonseca/XRD_PLOTTER)
""")

# Run the application
if __name__ == "__main__":
    main() 
import streamlit as st
import numpy as np
import pandas as pd
import re
import warnings
from io import BytesIO
from typing import Tuple, List, Dict, Optional, Any, Union

# Configure page 
st.set_page_config(
    page_title="XRD Plotter",
    page_icon="📊",
    layout="wide"
)

# Configuration and constants
CONFIG = {
    "DEFAULT_THETA_MIN": 0,
    "DEFAULT_THETA_MAX": 100,
    "PADDING": 5,
    "SUPPORTED_FORMATS": ['txt', 'csv', 'dat', 'xy'],
    "DEFAULT_DPI": 300,
    "LABEL_OFFSET_PERCENT": 0.05,  # Position labels 5% from the end of visible range
    "FALLBACK_MODE": False
}

# Try importing optional dependencies with fallbacks
try:
    import matplotlib
    matplotlib.use('Agg')  # Force Agg backend which works better in headless environments
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    CONFIG["FALLBACK_MODE"] = True
    warnings.warn("Matplotlib not available. Using simplified plotting mode.")

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Smoothing features will be disabled.")
    
    # Define a dummy savgol_filter function for fallback
    def savgol_filter(data, window_length, polyorder, **kwargs):
        """Dummy function when scipy is not available."""
        return data

try:
    import scienceplots
    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False
    warnings.warn("SciencePlots not available. Publication quality styling disabled.")

# ===== Data Processing Functions =====

def read_xrd_data(file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Parse XRD data from various file formats.
    
    Args:
        file: Uploaded file object
    
    Returns:
        Tuple of (x_data, y_data) as numpy arrays, or (None, None) if parsing fails
    """
    file_name = file.name.lower()
    
    try:
        # Handle .xy files (common XRD format)
        if file_name.endswith('.xy'):
            return _parse_xy_file(file)
            
        # Try as CSV
        data = pd.read_csv(file, header=None)
        if data.shape[1] >= 2:
            return data.iloc[:, 0].values, data.iloc[:, 1].values
            
    except Exception:
        pass
        
    # Fall back to general text parsing
    try:
        content = file.getvalue().decode('utf-8')
        return _parse_general_text(content)
    except Exception:
        st.error(f"Could not parse file: {file.name}")
        
    return None, None


def _parse_xy_file(file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Parse space/tab-separated XY file format."""
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


def _parse_general_text(content: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Attempt to parse text with numeric data in any reasonable format."""
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


def normalize_data(y_data: np.ndarray) -> np.ndarray:
    """Scale data to range [0,1]."""
    y_min, y_max = np.min(y_data), np.max(y_data)
    if y_max == y_min:
        return np.zeros_like(y_data)
    return (y_data - y_min) / (y_max - y_min)


def apply_smooth(y_data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply Savitzky-Golay smoothing with error handling."""
    if not SCIPY_AVAILABLE:
        st.warning("Smoothing unavailable - SciPy not installed")
        return y_data
        
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


# ===== Plotting Functions =====

def setup_plot_style(use_publication_style: bool, style_name: str, use_latex: bool) -> None:
    """Configure plot styling based on user preferences."""
    if not MATPLOTLIB_AVAILABLE:
        return
        
    if use_publication_style and SCIENCEPLOTS_AVAILABLE:
        try:
            plt.style.use(['science', style_name])
            
            if use_latex:
                plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                })
        except Exception as e:
            st.warning(f"Style application failed: {str(e)}")


def create_figure_with_data(
    processed_data: List[Dict[str, Any]],
    label_positions: List[Dict[str, Any]],
    plot_config: Dict[str, Any]
) -> Any:
    """Create and configure a matplotlib figure with the dataset."""
    if not MATPLOTLIB_AVAILABLE:
        return create_fallback_chart(processed_data, label_positions, plot_config)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each dataset
    for i, data in enumerate(processed_data):
        if plot_config.get('use_custom_range', False):
            min_theta = plot_config['min_theta']
            max_theta = plot_config['max_theta']
            mask = (data['x'] >= min_theta) & (data['x'] <= max_theta)
            x_plot = data['x'][mask]
            y_plot = data['y'][mask]
        else:
            x_plot = data['x']
            y_plot = data['y']
        
        ax.plot(x_plot, y_plot, label=data['label'], color=data['color'])
    
    # Add custom labels
    for label_info in label_positions:
        if label_info.get('show', False):
            ax.text(
                label_info['x'],
                label_info['y'],
                label_info['text'],
                color=label_info['color'],
                ha='right',
                va='center',
                fontweight='bold'
            )
    
    # Configure axes
    if plot_config.get('use_latex', False):
        ax.set_xlabel(r"$2\theta$ (degrees)")
    else:
        ax.set_xlabel("2$\theta$ (degrees)")
    
    ax.set_ylabel("Intensity (a.u.)")
    
    # Set axis limits if needed
    if plot_config.get('use_custom_range', False):
        ax.set_xlim(plot_config['min_theta'], plot_config['max_theta'])
    
    ax.grid(True, alpha=0.3)
    
    # Add legend if requested
    if plot_config.get('show_legend', True):
        bbox_to_anchor = plot_config.get('legend_bbox_to_anchor')
        if bbox_to_anchor:
            ax.legend(loc=plot_config['legend_position'], bbox_to_anchor=bbox_to_anchor)
        else:
            ax.legend(loc=plot_config['legend_position'])
    
    return fig


def create_fallback_chart(
    processed_data: List[Dict[str, Any]],
    label_positions: List[Dict[str, Any]],
    plot_config: Dict[str, Any]
) -> None:
    """Create a chart using Streamlit's native charting when matplotlib is unavailable."""
    # Convert the data into a format suitable for Streamlit
    chart_data = []
    
    for data in processed_data:
        if plot_config.get('use_custom_range', False):
            min_theta = plot_config['min_theta']
            max_theta = plot_config['max_theta']
            mask = (data['x'] >= min_theta) & (data['x'] <= max_theta)
            x = data['x'][mask]
            y = data['y'][mask]
        else:
            x = data['x']
            y = data['y']
            
        # Create DataFrame for this dataset
        df = pd.DataFrame({
            '2θ': x,
            data['label']: y
        })
        chart_data.append((df, data['color']))
    
    # Display using Streamlit's line chart
    st.subheader("XRD Data Plot (Fallback Mode)")
    st.write("*Note: Using simplified plotting due to matplotlib unavailability.*")
    
    # Create a chart for each dataset (Streamlit doesn't support multiple lines with different colors easily)
    for df, color in chart_data:
        st.line_chart(df.set_index('2θ'), use_container_width=True)
    
    # Show label positions in a table
    if any(label.get('show', False) for label in label_positions):
        st.subheader("Label Positions")
        labels_df = pd.DataFrame([
            {
                "Label": label['text'],
                "X Position (2θ)": label['x'],
                "Y Position (Intensity)": label['y']
            }
            for label in label_positions if label.get('show', False)
        ])
        if not labels_df.empty:
            st.dataframe(labels_df)
    
    # No return needed as we're directly rendering to Streamlit


def export_figure(
    processed_data: List[Dict[str, Any]],
    label_positions: List[Dict[str, Any]],
    plot_config: Dict[str, Any],
    width: float,
    height: float
) -> Tuple[Optional[BytesIO], Optional[BytesIO]]:
    """Create high-quality figure for export and return buffers."""
    if not MATPLOTLIB_AVAILABLE:
        st.error("Export functionality not available without matplotlib")
        return None, None
        
    try:
        # Create figure with settings for export
        save_fig = plt.figure(figsize=(width, height))
        save_ax = save_fig.add_subplot(111)
        
        # Plot each dataset
        for data in processed_data:
            if plot_config.get('use_custom_range', False):
                min_theta = plot_config['min_theta']
                max_theta = plot_config['max_theta']
                mask = (data['x'] >= min_theta) & (data['x'] <= max_theta)
                x_plot = data['x'][mask]
                y_plot = data['y'][mask]
            else:
                x_plot = data['x']
                y_plot = data['y']
            
            save_ax.plot(x_plot, y_plot, label=data['label'], color=data['color'])
        
        # Add custom labels
        for label_info in label_positions:
            if label_info.get('show', False):
                save_ax.text(
                    label_info['x'],
                    label_info['y'],
                    label_info['text'],
                    color=label_info['color'],
                    ha='right',
                    va='center',
                    fontweight='bold'
                )
        
        # Configure axes
        if plot_config.get('use_latex', False):
            save_ax.set_xlabel(r"$2\theta$ (degrees)")
        else:
            save_ax.set_xlabel("2$\theta$ (degrees)")
        
        save_ax.set_ylabel("Intensity (a.u.)")
        
        # Set axis limits if needed
        if plot_config.get('use_custom_range', False):
            save_ax.set_xlim(plot_config['min_theta'], plot_config['max_theta'])
        
        save_ax.grid(True, alpha=0.3)
        
        # Add legend if requested
        if plot_config.get('show_legend', True):
            bbox_to_anchor = plot_config.get('legend_bbox_to_anchor')
            if bbox_to_anchor:
                save_ax.legend(loc=plot_config['legend_position'], bbox_to_anchor=bbox_to_anchor)
            else:
                save_ax.legend(loc=plot_config['legend_position'])
        
        # Create PNG buffer
        png_buf = BytesIO()
        save_fig.savefig(png_buf, format="png", dpi=CONFIG["DEFAULT_DPI"], bbox_inches="tight")
        png_buf.seek(0)
        
        # Try PDF export
        pdf_buf = None
        try:
            pdf_buf = BytesIO()
            save_fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
            pdf_buf.seek(0)
        except Exception:
            pass
            
        plt.close(save_fig)
        return png_buf, pdf_buf
    except Exception as e:
        st.error(f"Error creating export: {str(e)}")
        return None, None


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


def find_label_position(x_data, y_data, is_custom_range, min_theta=None, max_theta=None):
    """Find appropriate default position for spectrum label."""
    if not len(x_data) or not len(y_data):
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
        st.subheader("Legend Settings")
        plot_config['show_legend'] = st.checkbox("Show legend", value=True)
        
        if plot_config['show_legend'] and MATPLOTLIB_AVAILABLE:
            legend_options = ["best", "upper right", "upper left", "lower left", "lower right",
                             "right", "center left", "center right", "lower center", "upper center",
                             "center", "outside"]
            plot_config['legend_position'] = st.selectbox("Legend position", legend_options, index=0)
            
            # Special handling for "outside" position
            if plot_config['legend_position'] == "outside":
                plot_config['legend_position'] = "center left"
                plot_config['legend_bbox_to_anchor'] = (1.05, 0.5)
            else:
                plot_config['legend_bbox_to_anchor'] = None
        
        # Plot style settings - only if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            st.subheader("Plot Style")
            can_use_pub_style = SCIENCEPLOTS_AVAILABLE
            plot_config['use_publication_style'] = st.checkbox(
                "Use publication quality style", 
                value=False, 
                disabled=not can_use_pub_style,
                help="Requires SciencePlots package" if not can_use_pub_style else None
            )
            
            if plot_config['use_publication_style']:
                plot_config['science_style'] = st.selectbox(
                    "Science style",
                    ["science", "ieee", "nature", "grid"],
                    index=0
                )
                plot_config['use_latex'] = st.checkbox("Use LaTeX for text rendering", value=False)
        
        # System status
        st.subheader("System Status")
        if CONFIG["FALLBACK_MODE"]:
            st.warning("Running in fallback mode - some features are limited")
            
        st.info(f"Matplotlib: {'Available' if MATPLOTLIB_AVAILABLE else 'Not Available'}")
        st.info(f"SciPy: {'Available' if SCIPY_AVAILABLE else 'Not Available'}")
        st.info(f"SciencePlots: {'Available' if SCIENCEPLOTS_AVAILABLE else 'Not Available'}")
    
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
        
        file_config['color'] = st.color_picker("Color", key=f"color_{idx}")
        
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


def render_export_controls(processed_data, label_positions, plot_config):
    """Render export controls for saving figures and data."""
    col1, col2 = st.columns(2)
    
    with col1:
        # Export figure - only available with matplotlib
        if MATPLOTLIB_AVAILABLE:
            if st.button("Save Figure"):
                # Get figure dimensions
                width = st.sidebar.number_input("Width (inches)", value=8.0, min_value=1.0, max_value=20.0)
                height = st.sidebar.number_input("Height (inches)", value=6.0, min_value=1.0, max_value=20.0)
                
                # Generate export figure
                png_buf, pdf_buf = export_figure(
                    processed_data,
                    label_positions,
                    plot_config,
                    width,
                    height
                )
                
                # Offer downloads
                if png_buf:
                    st.download_button(
                        label="Download PNG",
                        data=png_buf,
                        file_name="xrd_plot.png",
                        mime="image/png"
                    )
                
                if pdf_buf:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_buf,
                        file_name="xrd_plot.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.warning("PDF export failed. Try a different configuration.")
        else:
            st.warning("Figure export requires matplotlib which is not available")
    
    with col2:
        # Export data - always available
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
           - Legend position and visibility
           - Publication-quality styling
        4. **Export**: Save your processed data or download publication-ready figures
        
        Try it now by uploading your XRD data files!
        """)


# ===== Main Application =====

def main():
    st.title("XRD Data Plotter")
    st.write("Upload XRD files to visualize, compare, and analyze X-ray diffraction patterns")
    
    # Display matplotlib status alert if in fallback mode
    if CONFIG["FALLBACK_MODE"]:
        st.warning(
            "Running in compatibility mode: matplotlib is not available. "
            "Basic plotting functionality is enabled, but advanced features are limited."
        )
    
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
    
    # Apply plot styles if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        setup_plot_style(
            plot_config.get('use_publication_style', False),
            plot_config.get('science_style', 'science'),
            plot_config.get('use_latex', False)
        )
    
    # Create and display the plot
    fig = create_figure_with_data(all_processed_data, all_label_positions, plot_config)
    if MATPLOTLIB_AVAILABLE and fig:
        st.pyplot(fig)
    
    # Export controls
    render_export_controls(all_processed_data, all_label_positions, plot_config)


# Add footer with attribution
st.markdown("""
---
Made with [Streamlit](https://streamlit.io) • [GitHub Repository](https://github.com/jrjfonseca/XRD_PLOTTER)
""")

# Run the application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please report this issue to the repository maintainers.") 
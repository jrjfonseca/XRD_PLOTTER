import streamlit as st

# Set page title and favicon - THIS MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="XRD Plotter",
    page_icon="📊",
    layout="wide"
)

# Now import other libraries and set up dependencies
import sys
import os
import re
import base64
from io import BytesIO
import importlib.util

# Check if we're running in Streamlit Cloud
STREAMLIT_CLOUD = os.environ.get('STREAMLIT_CLOUD', 'false').lower() == 'true'

# Initialize dependency status tracking
MATPLOTLIB_AVAILABLE = False
SCIPY_AVAILABLE = False

# Attempt to configure matplotlib early with a compatible backend
try:
    # Force Agg backend which is most compatible with headless environments
    import matplotlib
    matplotlib.use('Agg')
    MATPLOTLIB_BACKEND_SET = True
except:
    MATPLOTLIB_BACKEND_SET = False

# Function to check if a package is installed
def is_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

# Function to lazily load numpy and pandas when needed
@st.cache_resource
def load_core_dependencies():
    import numpy as np
    import pandas as pd
    return np, pd

# Load these essential libraries only once they're needed
np, pd = load_core_dependencies()

# Try to import matplotlib with expanded error handling
@st.cache_resource
def load_matplotlib():
    try:
        # If we already configured the backend, just import plt
        if MATPLOTLIB_BACKEND_SET:
            import matplotlib.pyplot as plt
            return plt, True, None
        else:
            # Try with explicit backend configuration
            import matplotlib
            matplotlib.use('Agg')  # Use the Agg backend which works in headless environments
            import matplotlib.pyplot as plt
            return plt, True, None
    except ImportError as e:
        st.error(f"Could not import matplotlib. Error: {str(e)}")
        # Try an alternative approach with a different backend
        try:
            import matplotlib
            matplotlib.use('agg')  # Try lowercase 'agg' (sometimes matters)
            import matplotlib.pyplot as plt
            return plt, True, "Using alternative matplotlib configuration."
        except ImportError:
            # Last resort: try without specifying backend
            try:
                import matplotlib.pyplot as plt
                return plt, True, "Imported matplotlib without specifying backend."
            except Exception as e2:
                detailed_error = f"Original error: {str(e)}\nSecond attempt error: {str(e2)}"
                return None, False, f"Could not import matplotlib. Some features will be limited. Errors: {detailed_error}"

# Try to import scipy with expanded error handling
@st.cache_resource
def load_scipy():
    try:
        import scipy
        from scipy.signal import savgol_filter
        return savgol_filter, True, None
    except ImportError as e:
        return None, False, f"Could not import scipy. Smoothing features will be disabled. Error: {str(e)}"

# Import scienceplots for publication-quality plots
@st.cache_resource
def load_scienceplots():
    try:
        import scienceplots
        return {"scienceplots": scienceplots}
    except ImportError:
        return {"scienceplots": None}

# Fallback plotting function using HTML/CSS when matplotlib is not available
def create_basic_plot_html(x_data, y_data, labels, colors, label_positions=None, title="XRD Data"):
    """Create a basic plot using HTML/CSS when matplotlib is not available"""
    html = f"""
    <style>
    .plot-container {{
        height: 400px;
        width: 100%;
        position: relative;
        border: 1px solid #ddd;
        padding: 20px;
        background-color: white;
    }}
    .plot-line {{
        position: absolute;
        bottom: 50px;
        height: 300px;
        width: 100%;
        left: 50px;
    }}
    .plot-point {{
        position: absolute;
        width: 2px;
        background-color: var(--color);
        bottom: 50px;
    }}
    .plot-title {{
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }}
    .x-axis {{
        position: absolute;
        bottom: 20px;
        left: 50px;
        right: 20px;
        height: 1px;
        background-color: black;
    }}
    .y-axis {{
        position: absolute;
        bottom: 20px;
        left: 50px;
        width: 1px;
        height: 330px;
        background-color: black;
    }}
    .x-label {{
        position: absolute;
        bottom: 0;
        text-align: center;
        width: 100%;
    }}
    .y-label {{
        position: absolute;
        left: 0;
        bottom: 175px;
        transform: rotate(-90deg);
        transform-origin: center left;
    }}
    .legend {{
        position: absolute;
        top: 10px;
        right: 10px;
        border: 1px solid #ddd;
        padding: 5px;
        background-color: rgba(255,255,255,0.8);
    }}
    .legend-item {{
        margin: 5px;
        display: flex;
        align-items: center;
    }}
    .legend-color {{
        width: 20px;
        height: 3px;
        margin-right: 5px;
    }}
    .data-label {{
        position: absolute;
        font-weight: bold;
        white-space: nowrap;
        transform: translate(-50%, -100%);
        background-color: rgba(255, 255, 255, 0.7);
        padding: 2px 5px;
        border-radius: 3px;
        border: 1px solid #ddd;
    }}
    </style>
    
    <div class="plot-container">
        <div class="plot-title">{title}</div>
        <div class="x-axis"></div>
        <div class="y-axis"></div>
        <div class="x-label">2θ (degrees)</div>
        <div class="y-label">Intensity (a.u.)</div>
    """
    
    if len(x_data) > 0:
        # Find global min/max for scaling
        all_x = np.concatenate([d for d in x_data if len(d) > 0])
        all_y = np.concatenate([d for d in y_data if len(d) > 0])
        x_min = np.min(all_x)
        x_max = np.max(all_x)
        y_min = np.min(all_y)
        y_max = np.max(all_y)
        x_range = x_max - x_min
        y_range = y_max - y_min if y_max > y_min else 1
        
        # Add points for each dataset
        legend_html = '<div class="legend">'
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            if len(x) > 0 and len(y) > 0:
                color = colors[i] if i < len(colors) else f"hsl({(i * 50) % 360}, 70%, 50%)"
                label = labels[i] if i < len(labels) else f"Data {i+1}"
                
                # Only plot a subset of points for performance
                step = max(1, len(x) // 100)
                for j in range(0, len(x), step):
                    x_pos = 50 + ((x[j] - x_min) / x_range) * (100 - 70)  # 70% width
                    height = ((y[j] - y_min) / y_range) * 300  # Scale to 300px height
                    html += f'<div class="plot-point" style="left: {x_pos}%; height: {height}px; background-color: {color};"></div>'
                
                # Add to legend
                legend_html += f"""
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {color};"></div>
                    <div>{label}</div>
                </div>
                """
                
                # Add custom labels if provided
                if label_positions and i < len(label_positions) and label_positions[i]['show']:
                    x_pos_label = 50 + ((label_positions[i]['x'] - x_min) / x_range) * (100 - 70)
                    y_pos_label = 50 + 300 - ((label_positions[i]['y'] - y_min) / y_range) * 300
                    
                    html += f"""
                    <div class="data-label" style="left: {x_pos_label}%; top: {y_pos_label}px; color: {color};">
                        {label_positions[i]['text']}
                    </div>
                    """
        
        legend_html += '</div>'
        html += legend_html
    
    html += "</div>"
    return html

# Only initialize these when first needed
plt = None
savgol_filter = None
deps = {"scienceplots_loaded": False, "modules": None}

st.title("XRD Data Plotter")
st.write("Upload XRD files to visualize, compare, and analyze X-ray diffraction patterns")

st.sidebar.markdown("""
## Tips for First-time Users
- Upload one or more XRD files (.txt, .csv, .dat, .xy)
- Each file can be individually customized (color, normalization, etc.)
- The sidebar provides global settings for the plot
""")

# Function to read XRD data
def read_xrd_data(file):
    try:
        # Get file extension
        file_name = file.name.lower()
        
        # Handle .xy files specifically (typically space or tab-separated two-column data)
        if file_name.endswith('.xy'):
            content = file.getvalue().decode('utf-8')
            lines = content.split('\n')
            x_data = []
            y_data = []
            for line in lines:
                # Skip comment lines and empty lines
                if line.strip() and not line.strip().startswith('#'):
                    # Try to parse space or tab-separated values
                    fields = re.split(r'\s+', line.strip())
                    if len(fields) >= 2:
                        try:
                            x = float(fields[0])
                            y = float(fields[1])
                            x_data.append(x)
                            y_data.append(y)
                        except ValueError:
                            # Skip lines that don't have valid float values
                            pass
            if x_data and y_data:
                return np.array(x_data), np.array(y_data)
            return None, None
            
        # Try to read as CSV first for other formats
        data = pd.read_csv(file, header=None)
        if data.shape[1] >= 2:
            return data.iloc[:, 0], data.iloc[:, 1]
    except:
        # If CSV fails, try to read as text
        try:
            content = file.getvalue().decode('utf-8')
            # Try to find data patterns
            lines = content.split('\n')
            x_data = []
            y_data = []
            for line in lines:
                # Skip comment lines
                if line.strip() and not line.strip().startswith('#'):
                    # Look for pairs of numbers
                    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                    if len(numbers) >= 2:
                        x_data.append(float(numbers[0]))
                        y_data.append(float(numbers[1]))
            if x_data and y_data:
                return np.array(x_data), np.array(y_data)
        except:
            pass
    return None, None

# Function to normalize data
def normalize_data(y_data):
    return (y_data - np.min(y_data)) / (np.max(y_data) - np.min(y_data))

# File uploader - now including .xy format
uploaded_files = st.file_uploader("Upload XRD files", accept_multiple_files=True, type=['txt', 'csv', 'dat', 'xy'])

# Display app status in the sidebar
st.sidebar.title("App Status")
with st.sidebar.status("Checking system requirements...", expanded=False) as status:
    # Only load dependencies when they're actually needed (after files are uploaded)
    if uploaded_files:
        # Load matplotlib now, since we'll need it
        plt, MATPLOTLIB_AVAILABLE, matplotlib_error = load_matplotlib()
        savgol_filter, SCIPY_AVAILABLE, scipy_error = load_scipy()
        
        # Display any errors in the status
        if matplotlib_error:
            status.update(label="System Check - Issues Found", state="error", expanded=True)
            st.sidebar.error(matplotlib_error)
        elif scipy_error:
            status.update(label="System Check - Issues Found", state="warning", expanded=True)
            st.sidebar.warning(scipy_error)
        else:
            status.update(label="System Ready", state="complete")
    else:
        status.update(label="System Ready - Waiting for Upload", state="complete")

# Process uploaded files
if uploaded_files:
    # Main content area for processing files
    all_data = []
    all_processed_data = []
    all_labels = []
    all_colors = []
    all_label_positions = []
    
    # Global settings for fallback HTML plot
    min_x_all = float('inf')
    max_x_all = float('-inf')
    use_custom_range = False
    min_theta = 0
    max_theta = 100
    
    # Create sidebar for global settings (using whether matplotlib is available or not)
    with st.sidebar:
        st.header("Global Settings")
        
        # Create a container for plot range settings
        st.subheader("Plot Range")
        use_custom_range = st.checkbox("Use custom 2θ range", value=False)
        if use_custom_range:
            # Get the range of all uploaded files to set reasonable defaults
            for file in uploaded_files:
                x_data, _ = read_xrd_data(file)
                if x_data is not None and len(x_data) > 0:
                    min_x_all = min(min_x_all, np.min(x_data))
                    max_x_all = max(max_x_all, np.max(x_data))
            
            # If we couldn't get ranges from files, use defaults
            if min_x_all == float('inf'):
                min_x_all, max_x_all = 0, 100
            
            # Add some padding
            min_x_all = max(0, min_x_all - 5)
            max_x_all = max_x_all + 5
            
            # Create range sliders
            col1, col2 = st.columns(2)
            with col1:
                min_theta = st.number_input("Min 2θ", value=float(min_x_all), min_value=0.0)
            with col2:
                max_theta = st.number_input("Max 2θ", value=float(max_x_all), min_value=min_theta + 1.0)
    
    # Process each file
    for i, file in enumerate(uploaded_files):
        x_data, y_data = read_xrd_data(file)
        
        if x_data is not None and y_data is not None:
            # Store original data
            all_data.append({
                'x': x_data,
                'y': y_data,
                'filename': file.name
            })
            
            # Create a container for this file's controls
            with st.expander(f"Controls for {file.name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    normalize = st.checkbox("Normalize", key=f"norm_{i}")
                
                with col2:
                    label = st.text_input("Label", value=file.name, key=f"label_{i}")
                
                with col3:
                    # Fix the Y-offset issue by clearly separating it from other operations
                    y_offset = st.number_input("Y-offset", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key=f"offset_{i}")
                
                color = st.color_picker("Color", key=f"color_{i}")
                all_colors.append(color)
                
                # Apply smoothing if requested
                if SCIPY_AVAILABLE and st.checkbox("Apply smoothing", key=f"smooth_{i}"):
                    window = st.slider("Smoothing window", 3, 51, 5, 2, key=f"window_{i}")
                    # Apply smoothing to a copy of the data to avoid modifying the original
                    y_data_processed = savgol_filter(y_data.copy(), window, 2)
                else:
                    # Use a copy to avoid modifying the original
                    y_data_processed = y_data.copy()
                
                # Normalize if requested (after smoothing)
                if normalize:
                    y_data_processed = normalize_data(y_data_processed)
                
                # Apply offset AFTER all other processing
                y_data_processed = y_data_processed + y_offset
                
                # Add label positioning controls on right side of the plot
                st.subheader("Label Position")
                show_label = st.checkbox("Show label on plot", value=True, key=f"show_label_{i}")
                
                if show_label:
                    # Get data within the visible range for default positioning
                    if use_custom_range:
                        mask = (x_data >= min_theta) & (x_data <= max_theta)
                        x_range = x_data[mask] if any(mask) else x_data
                        y_range = y_data_processed[mask] if any(mask) else y_data_processed
                        if len(x_range) > 0:
                            max_x = max_theta - (max_theta - min_theta) * 0.05  # 95% of the way to the right edge
                        else:
                            max_x = max_theta
                    else:
                        if len(x_data) > 0:
                            x_max = np.max(x_data)
                            x_min = np.min(x_data)
                            max_x = x_max - (x_max - x_min) * 0.05  # 95% of the way to the right edge
                        else:
                            max_x = 0
                    
                    # Get the y-value at this x position or nearby
                    if len(x_data) > 0 and len(y_data_processed) > 0:
                        # Find the closest x-value to our desired position
                        if use_custom_range:
                            mask = (x_data >= min_theta) & (x_data <= max_theta)
                            x_visible = x_data[mask] if any(mask) else x_data
                            y_visible = y_data_processed[mask] if any(mask) else y_data_processed
                        else:
                            x_visible = x_data
                            y_visible = y_data_processed
                            
                        if len(x_visible) > 0:
                            # Find nearest x-value to our desired position
                            nearest_idx = np.abs(x_visible - max_x).argmin()
                            default_y = y_visible[nearest_idx]
                        else:
                            default_y = 0
                    else:
                        default_y = 0
                    
                    # Create columns for X and Y positioning
                    label_col1, label_col2 = st.columns(2)
                    
                    with label_col1:
                        x_pos = st.number_input(
                            "X Position", 
                            value=float(max_x),
                            min_value=float(min_theta) if use_custom_range else float(np.min(x_data)) if len(x_data) > 0 else 0.0,
                            max_value=float(max_theta) if use_custom_range else float(np.max(x_data)) if len(x_data) > 0 else 100.0,
                            key=f"label_x_{i}"
                        )
                    
                    with label_col2:
                        y_pos = st.number_input(
                            "Y Position", 
                            value=float(default_y),
                            key=f"label_y_{i}"
                        )
                    
                    # Store the label information
                    all_label_positions.append({
                        'x': x_pos,
                        'y': y_pos,
                        'text': label,
                        'color': color,
                        'show': True
                    })
                else:
                    all_label_positions.append({
                        'show': False,
                        'text': label,
                        'color': color
                    })
                
                # Store processed data for potential export
                all_processed_data.append({
                    'x': x_data,
                    'y': y_data_processed,
                    'label': label,
                    'color': color
                })
                
                all_labels.append(label)
    
    # Plot data
    if not MATPLOTLIB_AVAILABLE:
        st.warning("Matplotlib is not available. Using simplified plotting mode.")
        
        # Extract data for HTML plotting
        x_data_list = []
        y_data_list = []
        
        for data in all_processed_data:
            if use_custom_range:
                mask = (data['x'] >= min_theta) & (data['x'] <= max_theta)
                x_plot = data['x'][mask]
                y_plot = data['y'][mask]
            else:
                x_plot = data['x']
                y_plot = data['y']
            
            x_data_list.append(x_plot)
            y_data_list.append(y_plot)
        
        # Create HTML plot with custom labels
        html_plot = create_basic_plot_html(x_data_list, y_data_list, all_labels, all_colors, all_label_positions)
        st.components.v1.html(html_plot, height=500)
        
        # Export options
        if st.button("Export Data"):
            for data in all_processed_data:
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
    
    elif uploaded_files:
        # Full matplotlib plotting
        use_latex = False
        use_publication_style = False
        science_style = "science"
        legend_position = "best"
        legend_bbox_to_anchor = None
        
        # Continue with the additional settings if matplotlib is available
        with st.sidebar:
            # Legend settings
            st.subheader("Legend Settings")
            show_legend = st.checkbox("Show legend", value=True)
            
            if show_legend:
                legend_options = ["best", "upper right", "upper left", "lower left", "lower right", 
                                "right", "center left", "center right", "lower center", "upper center", 
                                "center", "outside"]
                legend_position = st.selectbox("Legend position", legend_options, index=0)
                
                # If 'outside' is selected, place it to the right of the plot
                if legend_position == "outside":
                    legend_position = "center left"
                    legend_bbox_to_anchor = (1.05, 0.5)
                else:
                    legend_bbox_to_anchor = None
                
            # Plot Style Settings
            st.subheader("Plot Style")
            use_publication_style = st.checkbox("Use publication quality style", value=False)
            
            if use_publication_style and not deps["scienceplots_loaded"]:
                # Load scienceplots only when needed
                deps["modules"] = load_scienceplots()
                deps["scienceplots_loaded"] = True
            
            if use_publication_style:
                if deps["modules"] and deps["modules"].get("scienceplots"):
                    science_style = st.selectbox("Science style", 
                                                ["science", "ieee", "nature", "science", "grid"], 
                                                index=0)
                    # Add an option to use LaTeX for text rendering
                    use_latex = st.checkbox("Use LaTeX for text rendering", value=False)
                else:
                    st.info("Publication quality plotting is disabled. The scienceplots package is not available.")
                    use_publication_style = False
                    use_latex = False
        
        # Set up the figure with the selected style
        if use_publication_style and deps["modules"] and deps["modules"].get("scienceplots"):
            try:
                plt.style.use(['science', science_style])
            except:
                st.warning("Could not apply science style. Using default style instead.")
            
            # Configure matplotlib to use LaTeX if requested
            if use_latex:
                try:
                    plt.rcParams.update({
                        "text.usetex": True,
                        "font.family": "serif",
                        "font.serif": ["Computer Modern Roman"],
                    })
                except:
                    st.warning("LaTeX configuration failed. Using default text rendering.")
                    use_latex = False
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Main content area - process each file for plotting
        for i, data in enumerate(all_processed_data):
            if use_custom_range:
                mask = (data['x'] >= min_theta) & (data['x'] <= max_theta)
                x_plot = data['x'][mask]
                y_plot = data['y'][mask]
            else:
                x_plot = data['x']
                y_plot = data['y']
            
            # Plot the data
            ax.plot(x_plot, y_plot, label=data['label'], color=data['color'])
        
        # Add custom positioned labels to the plot
        for label_info in all_label_positions:
            if label_info['show']:
                ax.text(
                    label_info['x'], 
                    label_info['y'], 
                    label_info['text'],
                    color=label_info['color'],
                    ha='right',  # Right-align the text
                    va='center',  # Vertically center the text
                    fontweight='bold'
                )
        
        # Customize plot - use LaTeX-compatible theta character for the x-axis label
        if use_latex:
            ax.set_xlabel(r"$2\theta$ (degrees)")
        else:
            ax.set_xlabel("2$\theta$ (degrees)")  # This works in non-LaTeX mode
        
        ax.set_ylabel("Intensity (a.u.)")
        
        # Set x-axis limits if custom range is specified
        if use_custom_range:
            ax.set_xlim(min_theta, max_theta)
        
        ax.grid(True, alpha=0.3)
        
        # Create the legend with the specified position if requested
        if show_legend:
            if legend_bbox_to_anchor:
                ax.legend(loc=legend_position, bbox_to_anchor=legend_bbox_to_anchor)
            else:
                ax.legend(loc=legend_position)
        
        # Display plot
        st.pyplot(fig)
        
        # Add option to save the figure with publication quality
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Figure"):
                # Save figure with high dpi for publication
                dpi = 300
                width = st.sidebar.number_input("Width (inches)", value=8.0, min_value=1.0, max_value=20.0)
                height = st.sidebar.number_input("Height (inches)", value=6.0, min_value=1.0, max_value=20.0)
                
                # Create a new figure with the specified dimensions for saving
                save_fig, save_ax = plt.subplots(figsize=(width, height))
                
                # Recreate the plot
                for i, data in enumerate(all_processed_data):
                    if use_custom_range:
                        mask = (data['x'] >= min_theta) & (data['x'] <= max_theta)
                        x_plot = data['x'][mask]
                        y_plot = data['y'][mask]
                    else:
                        x_plot = data['x']
                        y_plot = data['y']
                    
                    save_ax.plot(x_plot, y_plot, label=data['label'], color=data['color'])
                
                # Add custom positioned labels to the saved plot
                for label_info in all_label_positions:
                    if label_info['show']:
                        save_ax.text(
                            label_info['x'], 
                            label_info['y'], 
                            label_info['text'],
                            color=label_info['color'],
                            ha='right',  # Right-align the text
                            va='center',  # Vertically center the text
                            fontweight='bold'
                        )
                
                # Use LaTeX-compatible theta character for the x-axis label in the saved figure
                if use_latex:
                    save_ax.set_xlabel(r"$2\theta$ (degrees)")
                else:
                    save_ax.set_xlabel("2$\theta$ (degrees)")
                    
                save_ax.set_ylabel("Intensity (a.u.)")
                
                if use_custom_range:
                    save_ax.set_xlim(min_theta, max_theta)
                
                save_ax.grid(True, alpha=0.3)
                
                # Add legend to saved figure if requested
                if show_legend:
                    if legend_bbox_to_anchor:
                        save_ax.legend(loc=legend_position, bbox_to_anchor=legend_bbox_to_anchor)
                    else:
                        save_ax.legend(loc=legend_position)
                
                # Save the figure
                buf = BytesIO()
                save_fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
                buf.seek(0)
                
                # Create a download button
                btn = st.download_button(
                    label="Download PNG",
                    data=buf,
                    file_name="xrd_plot.png",
                    mime="image/png"
                )
                
                # Also try PDF if possible
                try:
                    buf_pdf = BytesIO()
                    save_fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
                    buf_pdf.seek(0)
                    
                    btn_pdf = st.download_button(
                        label="Download PDF",
                        data=buf_pdf,
                        file_name="xrd_plot.pdf",
                        mime="application/pdf"
                    )
                except:
                    st.warning("PDF export is not available in this environment.")
                
                plt.close(save_fig)
        
        # Export options
        with col2:
            if st.button("Export Data"):
                for data in all_processed_data:
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
else:
    st.info("Please upload XRD data files to begin plotting.")
    
    # Add example/tutorial area
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
        
# Add footer with attribution
st.markdown("""
---
Made with [Streamlit](https://streamlit.io) • [GitHub Repository](https://github.com/jrjfonseca/XRD_PLOTTER)
""") 
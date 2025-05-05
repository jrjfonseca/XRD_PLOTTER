import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
from io import StringIO
import re

# Import scienceplots for publication-quality plots
try:
    import scienceplots
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False
    st.warning("scienceplots not found. Please install with: pip install scienceplots")

st.set_page_config(page_title="XRD Plotter", layout="wide")

st.title("XRD Data Plotter")

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

if uploaded_files:
    # Create sidebar for global settings
    with st.sidebar:
        st.header("Global Settings")
        
        # Create a container for plot range settings
        st.subheader("Plot Range")
        use_custom_range = st.checkbox("Use custom 2θ range", value=False)
        if use_custom_range:
            # Get the range of all uploaded files to set reasonable defaults
            min_x_all = float('inf')
            max_x_all = float('-inf')
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
        
        # Legend position controls
        st.subheader("Legend Settings")
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
        if HAS_SCIENCEPLOTS:
            use_publication_style = st.checkbox("Use publication quality style", value=False)
            if use_publication_style:
                science_style = st.selectbox("Science style", 
                                            ["science", "ieee", "nature", "science", "grid"], 
                                            index=0)
                # Add an option to use LaTeX for text rendering
                use_latex = st.checkbox("Use LaTeX for text rendering", value=False)
        else:
            use_publication_style = False
            use_latex = False
            st.warning("scienceplots package not found. Install with: pip install scienceplots")
    
    # Set up the figure with the selected style
    if HAS_SCIENCEPLOTS and use_publication_style:
        plt.style.use(['science', science_style])
        
        # Configure matplotlib to use LaTeX if requested
        if use_latex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Store data for each file and their processed versions
    all_data = []
    all_processed_data = []
    
    # Main content area - process each file
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
                
                # Apply smoothing if requested
                if st.checkbox("Apply smoothing", key=f"smooth_{i}"):
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
            
            # Filter data if custom range is specified
            if use_custom_range:
                mask = (x_data >= min_theta) & (x_data <= max_theta)
                x_plot = x_data[mask]
                y_plot = y_data_processed[mask]
            else:
                x_plot = x_data
                y_plot = y_data_processed
            
            # Plot the data
            ax.plot(x_plot, y_plot, label=label, color=color)
            
            # Store processed data for potential export
            all_processed_data.append({
                'x': x_data,
                'y': y_data_processed,
                'label': label
            })
    
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
    
    # Create the legend with the specified position
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
                
                save_ax.plot(x_plot, y_plot, label=data['label'])
            
            # Use LaTeX-compatible theta character for the x-axis label in the saved figure
            if use_latex:
                save_ax.set_xlabel(r"$2\theta$ (degrees)")
            else:
                save_ax.set_xlabel("2$\theta$ (degrees)")
                
            save_ax.set_ylabel("Intensity (a.u.)")
            
            if use_custom_range:
                save_ax.set_xlim(min_theta, max_theta)
            
            save_ax.grid(True, alpha=0.3)
            
            if legend_bbox_to_anchor:
                save_ax.legend(loc=legend_position, bbox_to_anchor=legend_bbox_to_anchor)
            else:
                save_ax.legend(loc=legend_position)
            
            # Save the figure
            save_fig.savefig("xrd_plot.png", dpi=dpi, bbox_inches="tight")
            save_fig.savefig("xrd_plot.pdf", bbox_inches="tight")
            plt.close(save_fig)
            
            st.success("Figure saved as 'xrd_plot.png' and 'xrd_plot.pdf'")
    
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
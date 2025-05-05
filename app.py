import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
from io import StringIO
import re

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
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Store data for each file
    all_data = []
    
    # Process each file
    for i, file in enumerate(uploaded_files):
        x_data, y_data = read_xrd_data(file)
        
        if x_data is not None and y_data is not None:
            # Create a container for this file's controls
            with st.expander(f"Controls for {file.name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    normalize = st.checkbox("Normalize", key=f"norm_{i}")
                
                with col2:
                    label = st.text_input("Label", value=file.name, key=f"label_{i}")
                
                with col3:
                    offset = st.slider("Y-offset", 0.0, 10.0, 0.0, 0.5, key=f"offset_{i}")
                
                color = st.color_picker("Color", key=f"color_{i}")
                
                # Apply smoothing if requested
                if st.checkbox("Apply smoothing", key=f"smooth_{i}"):
                    window = st.slider("Smoothing window", 3, 51, 5, 2, key=f"window_{i}")
                    y_data = savgol_filter(y_data, window, 2)
            
            # Normalize if requested
            if normalize:
                y_data = normalize_data(y_data)
            
            # Apply offset
            y_data = y_data + offset
            
            # Plot the data
            line = ax.plot(x_data, y_data, label=label, color=color)
            
            # Store data for potential export
            all_data.append({
                'x': x_data,
                'y': y_data,
                'label': label
            })
    
    # Customize plot
    ax.set_xlabel("2θ (degrees)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Display plot
    st.pyplot(fig)
    
    # Export options
    if st.button("Export Data"):
        for data in all_data:
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
import streamlit as st
import subprocess
import sys

st.title("XRD Plotter Diagnostics")

# Show Python version
st.subheader("Python Information")
st.code(sys.version)

# Show installed packages
st.subheader("Installed Packages")
try:
    result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
    st.code(result.stdout)
except Exception as e:
    st.error(f"Error running pip list: {str(e)}")

# Try importing matplotlib
st.subheader("Matplotlib Import Test")
try:
    import matplotlib
    import matplotlib.pyplot as plt
    st.success(f"✅ Matplotlib imported successfully (version {matplotlib.__version__})")
    
    # Try a simple plot
    st.write("Testing matplotlib plot generation:")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    st.pyplot(fig)
    
except Exception as e:
    st.error(f"❌ Matplotlib import failed: {str(e)}")
    st.info("Troubleshooting: Make sure matplotlib is installed with 'pip install matplotlib'")

# Try importing scienceplots
st.subheader("SciencePlots Import Test")
try:
    import scienceplots
    st.success(f"✅ SciencePlots imported successfully")
    
    # Try using scienceplots
    st.write("Testing SciencePlots style:")
    with plt.style.context(['science']):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        st.pyplot(fig)
    
except Exception as e:
    st.error(f"❌ SciencePlots import failed: {str(e)}")
    st.info("Troubleshooting: Make sure scienceplots is installed with 'pip install scienceplots'")

# Try fixing common issues
st.subheader("Potential Solutions")
st.write("If you're experiencing issues, try the following commands in your terminal:")

st.code("""
# Update pip
pip install --upgrade pip

# Install specific versions known to work
pip install matplotlib==3.7.2
pip install SciencePlots==2.1.0
pip install cycler>=0.10.0 pyparsing>=2.0.1

# Restart the Streamlit app after installation
""")

st.write("You can also try modifying the requirements.txt file to include these specific versions.") 
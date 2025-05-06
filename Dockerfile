FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for matplotlib
RUN apt-get update && apt-get install -y \
    build-essential \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Streamlit Cloud environment variable
ENV STREAMLIT_CLOUD=true

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "streamlit_app.py"] 
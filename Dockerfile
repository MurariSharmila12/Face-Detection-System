# Use a full Python 3.9 image for better compatibility
FROM python:3.9

# Install a comprehensive set of build tools and CV/media libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv-python-headless
RUN pip install --no-cache-dir mediapipe

# Copy the rest of the app code
COPY . .

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]

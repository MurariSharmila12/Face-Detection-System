# Use a full Python 3.9 image for better compatibility
FROM python:3.9

# Install a comprehensive set of build tools and CV/media libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
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
# We will install mediapipe separately with a specific compatible version
RUN pip install --no-cache-dir opencv-python-headless
RUN pip install --no-cache-dir mediapipe==0.9.1

# Copy the rest of the app code
COPY . .

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]

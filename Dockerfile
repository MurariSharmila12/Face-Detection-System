# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies AND build tools needed by MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Command to run the application (make sure this matches your app)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]

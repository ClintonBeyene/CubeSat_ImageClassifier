# start from python base image

# Use official Python image
FROM python:3.10.12

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app code
COPY ./app ./app

# Copy TFLite model to the container root (if not already in app/)
COPY ../Cubenet_Best_Quantized.tflite ./Cubenet_Best_Quantized.tflite

# Expose port
EXPOSE 80

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

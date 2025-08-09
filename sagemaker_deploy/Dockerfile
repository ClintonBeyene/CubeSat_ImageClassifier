# start from python base image

# Use official Python image
FROM python:3.10.12

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
# Expect build context = repo root (docker build -f sagemaker_deploy/Dockerfile .)
COPY sagemaker_deploy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app code
COPY sagemaker_deploy/app ./app

# Copy TFLite model to container (if not already in app/)
COPY Cubenet_Best_Quantized.tflite ./Cubenet_Best_Quantized.tflite

# Expose port
EXPOSE 80

# Run FastAPI app with uvicorn; honor platform PORT if provided
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-80}"]

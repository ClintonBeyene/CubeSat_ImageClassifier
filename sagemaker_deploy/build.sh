#!/bin/bash
# Build script to create SageMaker-compatible Docker image

# Set variables
IMAGE_NAME="cubesat-tflite-inference"
ECR_REPO="456202168010.dkr.ecr.us-east-1.amazonaws.com/cubesat-tflite-inference"
REGION="us-east-1"

# Build with Docker format (not OCI)
echo "Building Docker image with Docker format..."
docker build --platform linux/amd64 --tag ${IMAGE_NAME}:latest .

# Tag for ECR
echo "Tagging image for ECR..."
docker tag ${IMAGE_NAME}:latest ${ECR_REPO}:latest

# Get ECR login token
echo "Logging into ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPO}

# Push to ECR
echo "Pushing image to ECR..."
docker push ${ECR_REPO}:latest

echo "Build and push complete!"

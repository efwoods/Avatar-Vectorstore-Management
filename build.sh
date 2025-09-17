#!/bin/bash
# build.sh - Build and push nn-avatar-adapter-management to Docker Hub

# Configuration
IMAGE_NAME="nn-avatar-vectorstore-management"
TAG="latest"
DOCKERHUB_USER="evdev3" # Replace with your Docker Hub username
FULL_IMAGE_NAME="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

# Check if Docker is installed
if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed."
    exit 1
fi

# Build the image
echo "Building Docker image: ${FULL_IMAGE_NAME}"
docker build -t "${FULL_IMAGE_NAME}" -f Dockerfile .

if [ $? -ne 0 ]; then
    echo "Error: Docker build failed."
    exit 1
fi

# Push to Docker Hub
echo "Pushing to Docker Hub: ${FULL_IMAGE_NAME}"
docker push "${FULL_IMAGE_NAME}"

if [ $? -ne 0 ]; then
    echo "Error: Docker push failed."
    exit 1
fi

echo "Build and push completed: ${FULL_IMAGE_NAME}"
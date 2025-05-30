#!/bin/bash
# Build and push Docker image to Azure Container Registry

# Docker is not installed or running in this environment.
# Use Azure CLI to build and push the image without Docker.

set -e

ACR_NAME="llmpacr01"
IMAGE_NAME="functionapp-image"
IMAGE_TAG="latest"
DOCKERFILE_PATH="."
ENV_FILE="/workspace/.azure/toolbox001/.env"

if [ -f "$ENV_FILE" ]; then
  export $(grep AZURE_RESOURCE_GROUP "$ENV_FILE" | xargs)
else
  echo "ERROR: $ENV_FILE not found"
  exit 1
fi

# Use az acr build to build and push the image in Azure (no local Docker required)
# The build is performed using 'az acr build', which uploads your source code to Azure Container Registry (ACR)
# and builds the Docker image in the cloud, so Docker does not need to be installed locally.
az acr build --registry $ACR_NAME --resource-group "$AZURE_RESOURCE_GROUP" --image $IMAGE_NAME:$IMAGE_TAG $DOCKERFILE_PATH

echo "Image built and pushed to $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG"

# The error means the base image 'mcr.microsoft.com/azure-functions/python:4-python311' does not exist.
# To fix this, update your Dockerfile to use a valid Azure Functions Python base image tag.
# For Python 3.11, use: 'mcr.microsoft.com/azure-functions/python:4.0-python3.11'

# Example: To pull a different Azure Functions Python base image (e.g., Python 3.12), use:
# docker pull mcr.microsoft.com/azure-functions/python:4.0-python3.12

# After updating your Dockerfile, rerun this script.

# Next steps to deploy your Azure Function App from the uploaded image:

# 1. Ensure your Function App is configured for a custom container (Linux, Docker).
# 2. Set the Function App's container settings to use your ACR image:
#    - IMAGE: llmpacr01.azurecr.io/functionapp-image:latest
#    - Registry: Azure Container Registry (enable managed identity or provide credentials if needed)
# 3. You can do this via Azure Portal, or with Azure CLI:

# Example Azure CLI command:
# az functionapp config container set \
#   --name <FUNCTION_APP_NAME> \
#   --resource-group "$AZURE_RESOURCE_GROUP" \
#   --docker-custom-image-name llmpacr01.azurecr.io/functionapp-image:latest \
#   --docker-registry-server-url https://llmpacr01.azurecr.io

# If using managed identity for ACR access:
# az functionapp identity assign --name <FUNCTION_APP_NAME> --resource-group "$AZURE_RESOURCE_GROUP"
# az acr update -n llmpacr01 --admin-enabled false

# After setting the container config, restart the Function App if needed:
# az functionapp restart --name <FUNCTION_APP_NAME> --resource-group "$AZURE_RESOURCE_GROUP"

FROM mcr.microsoft.com/azure-functions/python:4-python3.12

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy function app code into the container
COPY function_app/ /home/site/wwwroot

# Optional: install Python dependencies
COPY requirements.txt /home/site/wwwroot
RUN pip install -r /home/site/wwwroot/requirements.txt

FROM mcr.microsoft.com/azure-functions/python:4.0-python3.12

# Install ffmpeg for audio processing
RUN apt-get update && apt-get install -y ffmpeg

# Copy your function app code into the container
COPY . /home/site/wwwroot
WORKDIR /home/site/wwwroot
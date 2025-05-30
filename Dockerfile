FROM mcr.microsoft.com/azure-functions/python:4.0-python3.12

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy function app code
COPY function_app/ /home/site/wwwroot

# (Optional) If requirements.txt is in the root, copy it as well

COPY requirements.txt /home/site/wwwroot
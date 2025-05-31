import os
import logging
import tempfile
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI()

@app.get("/healthz")
def health_check():
    """Health check endpoint."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return {"status": "ok", "ffmpeg": "available"}
    except Exception:
        return {"status": "error", "ffmpeg": "not available"}

@app.on_event("startup")
def on_startup():
    logging.info("audiomono container app started.")
    audiomono_endpoint = os.getenv("AUDIOMONO_ENDPOINT")
    if audiomono_endpoint:
        logging.info(f"AUDIOMONO_ENDPOINT is set to: {audiomono_endpoint}")
    else:
        logging.warning("AUDIOMONO_ENDPOINT environment variable is NOT set. Set this in your main function app to call this service.")

class ConvertRequest(BaseModel):
    blob_url: str
    storage_account_name: str
    source_container: str = "audio-in"
    dest_container: str = "audio-preprocessed"

def convert_stereo_to_mono_ffmpeg(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    else:
        logging.error(f"[FFMPEG] Error: {result.stderr}")
        return False

@app.post("/convert-to-mono")
def convert_to_mono(req: ConvertRequest):
    try:
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=f"https://{req.storage_account_name}.blob.core.windows.net",
            credential=credential
        )
        container_client = blob_service_client.get_container_client(req.source_container)
        blob_name = req.blob_url.split("/")[-1]
        blob_client = container_client.get_blob_client(blob_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_input = os.path.join(tmpdir, blob_name)
            with open(local_input, "wb") as f:
                f.write(blob_client.download_blob().readall())
            local_output = os.path.join(tmpdir, f"mono_{blob_name}")
            if not convert_stereo_to_mono_ffmpeg(local_input, local_output):
                raise HTTPException(status_code=500, detail="FFmpeg conversion failed")
            # Upload to destination container
            dest_container_client = blob_service_client.get_container_client(req.dest_container)
            dest_blob_client = dest_container_client.get_blob_client(f"mono_{blob_name}")
            with open(local_output, "rb") as f:
                dest_blob_client.upload_blob(f, overwrite=True)
            mono_url = dest_blob_client.url
            return {"mono_url": mono_url}
    except Exception as e:
        logging.error(f"[audiomono] Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))

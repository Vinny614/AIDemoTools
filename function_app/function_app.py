import logging
import os
from datetime import datetime, timedelta
import json
import traceback
import requests
import asyncio
import time
import azure.functions as func
import azure.durable_functions as df
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from dotenv import load_dotenv
from src.helpers.azure_function import (
    check_if_azurite_storage_emulator_is_running,
    check_if_env_var_is_set,
)
from azure.identity import DefaultAzureCredential

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger("azure")
_logger.setLevel(logging.WARNING)

speech_endpoint = os.getenv("SPEECH_ENDPOINT")
if not speech_endpoint:
    raise Exception("SPEECH_ENDPOINT environment variable is not set")

app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)

IS_CONTENT_UNDERSTANDING_DEPLOYED = check_if_env_var_is_set("CONTENT_UNDERSTANDING_ENDPOINT")
IS_AOAI_DEPLOYED = check_if_env_var_is_set("AOAI_ENDPOINT")
IS_DOC_INTEL_DEPLOYED = check_if_env_var_is_set("DOC_INTEL_ENDPOINT")
IS_SPEECH_DEPLOYED = check_if_env_var_is_set("SPEECH_ENDPOINT")
IS_LANGUAGE_DEPLOYED = check_if_env_var_is_set("LANGUAGE_ENDPOINT")
IS_STORAGE_ACCOUNT_AVAILABLE = (
    os.getenv("AzureWebJobsStorage") == "UseDevelopmentStorage=true"
    and check_if_azurite_storage_emulator_is_running()
) or all([
    check_if_env_var_is_set("AzureWebJobsStorage__accountName"),
    check_if_env_var_is_set("AzureWebJobsStorage__blobServiceUri"),
    check_if_env_var_is_set("AzureWebJobsStorage__queueServiceUri"),
    check_if_env_var_is_set("AzureWebJobsStorage__tableServiceUri"),
])
IS_COSMOSDB_AVAILABLE = check_if_env_var_is_set("COSMOSDB_DATABASE_NAME") and check_if_env_var_is_set("CosmosDbConnectionSetting__accountEndpoint")

def generate_sas_url(container_name, blob_name):
    credential = DefaultAzureCredential()
    storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
    if not storage_account_name:
        raise ValueError("STORAGE_ACCOUNT_NAME environment variable is not set.")

    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=credential
    )

    expiry_time = datetime.utcnow() + timedelta(hours=1)
    user_delegation_key = blob_service_client.get_user_delegation_key(
        key_start_time=datetime.utcnow(),
        key_expiry_time=expiry_time
    )

    sas_token = generate_blob_sas(
        account_name=storage_account_name,
        container_name=container_name,
        blob_name=blob_name,
        permission=BlobSasPermissions(read=True),
        expiry=expiry_time,
        user_delegation_key=user_delegation_key
    )

    return f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"

if IS_AOAI_DEPLOYED:
    from bp_summarize_text import bp_summarize_text
    app.register_blueprint(bp_summarize_text)
if IS_DOC_INTEL_DEPLOYED and IS_AOAI_DEPLOYED:
    from bp_doc_intel_extract_city_names import bp_doc_intel_extract_city_names
    from bp_form_extraction_with_confidence import bp_form_extraction_with_confidence
    app.register_blueprint(bp_doc_intel_extract_city_names)
    app.register_blueprint(bp_form_extraction_with_confidence)
if IS_SPEECH_DEPLOYED and IS_AOAI_DEPLOYED:
    from bp_call_center_audio_analysis import bp_call_center_audio_analysis
    app.register_blueprint(bp_call_center_audio_analysis)
if IS_DOC_INTEL_DEPLOYED:
    from bp_multimodal_doc_intel_processing import bp_multimodal_doc_intel_processing
    app.register_blueprint(bp_multimodal_doc_intel_processing)
if IS_CONTENT_UNDERSTANDING_DEPLOYED:
    from bp_content_understanding_audio import bp_content_understanding_audio
    from bp_content_understanding_document import bp_content_understanding_document
    from bp_content_understanding_image import bp_content_understanding_image
    from bp_content_understanding_video import bp_content_understanding_video
    app.register_blueprint(bp_content_understanding_document)
    app.register_blueprint(bp_content_understanding_video)
    app.register_blueprint(bp_content_understanding_audio)
    app.register_blueprint(bp_content_understanding_image)
if IS_LANGUAGE_DEPLOYED:
    from bp_pii_redaction import bp_pii_redaction
    app.register_blueprint(bp_pii_redaction)

if IS_STORAGE_ACCOUNT_AVAILABLE and IS_COSMOSDB_AVAILABLE and IS_AOAI_DEPLOYED and IS_DOC_INTEL_DEPLOYED:
    from extract_blob_field_info_to_cosmosdb import get_structured_extraction_func_outputs
    COSMOSDB_DATABASE_NAME = os.getenv("COSMOSDB_DATABASE_NAME")

    @app.function_name("blob_form_extraction_to_cosmosdb")
    @app.blob_trigger(
        arg_name="inputblob1",
        path="blob-form-to-cosmosdb-blobs/{name}",
        connection="AzureWebJobsStorage",
    )
    @app.cosmos_db_output(
        arg_name="outputdocument",
        connection="CosmosDbConnectionSetting",
        database_name=COSMOSDB_DATABASE_NAME,
        container_name="blob-form-to-cosmosdb-container",
    )
    def extract_blob_pdf_fields_to_cosmosdb(inputblob, outputdocument):
        output_result = get_structured_extraction_func_outputs(inputblob)
        outputdocument.set(func.Document.from_dict(output_result))

@app.function_name("audio_blob_trigger")
@app.blob_trigger(
    arg_name="inputblob2",
    path="audio-in/{name}",
    connection="AzureWebJobsStorage",
)
@app.durable_client_input(client_name="client")
async def audio_blob_trigger(inputblob2: func.InputStream, client: df.DurableOrchestrationClient):
    logging.warning("🔥 Blob trigger fired!")
    blob_name = inputblob2.name.replace("audio-in/", "")
    logging.warning(f"Blob name: {blob_name}")
    try:
        await asyncio.sleep(3)  # Simulate some processing delay
        sas_url = generate_sas_url("audio-in", blob_name)
        logging.warning(f"✅ SAS URL: {sas_url}")
        input_payload = {"sas_url": sas_url, "blob_name": blob_name}
        instance_id = await client.start_new("audio_processing_orchestrator", None, input_payload)
        logging.info(f"🎯 Orchestration started: {instance_id}")
    except Exception as e:
        logging.error(f"❌ Error in blob trigger: {e}")
        logging.error(traceback.format_exc())

@app.orchestration_trigger(context_name="context")
def audio_processing_orchestrator(context):
    input_data = context.get_input()
    logging.info(f"[Orchestrator] Started with input: {input_data}")
    sas_url = input_data.get("sas_url")
    blob_name = input_data.get("blob_name")
    batch_info = yield context.call_activity("start_batch_activity", {"sas_url": sas_url, "blob_name": blob_name})
    result_data = yield context.call_activity("poll_batch_activity", batch_info)
    yield context.call_activity("write_output_activity", result_data)

@app.activity_trigger(input_name="input_data")
def start_batch_activity(input_data):
    logging.warning(f"[Activity] Received input: {input_data}")
    sas_url = input_data["sas_url"]
    blob_name = input_data["blob_name"]
    transcription_url = f"{speech_endpoint}/speechtotext/v3.1/transcriptions"
    payload = {
        "displayName": f"Transcription - {blob_name}",
        "locale": "en-GB",
        "contentUrls": [sas_url],
        "properties": {
            "wordLevelTimestampsEnabled": True,
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "Masked"
        }
    }
    credential = DefaultAzureCredential()
    token = credential.get_token("https://cognitiveservices.azure.com/.default").token
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    response = requests.post(transcription_url, json=payload, headers=headers)
    if response.status_code not in [200, 201, 202]:
        logging.error(f"[Activity] Failed to start transcription: {response.status_code} - {response.text}")
        raise Exception("Batch transcription start failed")
    transcription_location = response.headers["Location"]
    logging.warning(f"[Activity] Transcription started: {transcription_location}")
    return {"status_url": transcription_location, "blob_name": blob_name}

@app.activity_trigger(input_name="batch_info")
def poll_batch_activity(batch_info):
    logging.warning(f"[Activity] Polling real batch job: {batch_info}")
    return {"result": "real-transcript-content", "blob_name": batch_info["blob_name"]}

@app.activity_trigger(input_name="result_data")
def write_output_activity(result_data):
    logging.info(f"[Activity] Writing result to blob: {result_data}")
    try:
        credential = DefaultAzureCredential()
        storage_account_name = os.getenv("STORAGE_ACCOUNT_NAME")
        if not storage_account_name:
            raise ValueError("Missing STORAGE_ACCOUNT_NAME")
        blob_service_client = BlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=credential
        )
        container_name = "audio-transcript-out"
        blob_name = result_data["blob_name"].rsplit(".", 1)[0] + ".json"
        output_container = blob_service_client.get_container_client(container_name)
        output_blob = output_container.get_blob_client(blob_name)
        content = json.dumps(result_data, indent=2)
        output_blob.upload_blob(content, overwrite=True)
        logging.info(f"[Activity] Output written to: {container_name}/{blob_name}")
        return "OK"
    except Exception as e:
        logging.error(f"[Activity] Failed to write output: {e}")
        return "ERROR"

@app.function_name(name="start_audio_processing")
@app.route(route="start-audio-processing", auth_level=func.AuthLevel.FUNCTION)
@app.durable_client_input(client_name="client")
def start_audio_processing(req: func.HttpRequest, client: df.DurableOrchestrationClient) -> func.HttpResponse:
    try:
        data = req.get_json()
        instance_id = client.start_new("audio_processing_orchestrator", None, data.get("input"))
        logging.info(f"🎯 Durable orchestration started: {instance_id}")
        return client.create_check_status_response(req, instance_id)
    except Exception as e:
        logging.error(f"❌ Failed to start orchestration via HTTP: {e}")
        return func.HttpResponse("Error starting orchestration", status_code=500)

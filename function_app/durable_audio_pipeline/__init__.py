import azure.functions as func
import azure.durable_functions as df

app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.blob_trigger(
    arg_name="inputblob",
    path="audio-in/{name}",
    connection="AzureWebJobsStorage",
)
@app.durable_client_input(client_name="client")
def audio_blob_trigger(inputblob: func.InputStream, client):
    # Start the orchestrator with blob info
    instance_id = client.start_new(
        "audio_processing_orchestrator",
        None,
        {"blob_name": inputblob.name, "container": inputblob.blob_container}
    )
    return func.HttpResponse(f"Started orchestration with ID = '{instance_id}'.", status_code=202)

@app.orchestration_trigger(context_name="context")
def audio_processing_orchestrator(context):
    from .orchestrator import main as orchestrator_main # type: ignore
    return orchestrator_main(context)

@app.activity_trigger(input_name="input_data")
def start_batch_activity(input_data):
    from .activity_start_batch import main as start_main # type: ignore
    return start_main(input_data)

@app.activity_trigger(input_name="batch_info")
def poll_batch_activity(batch_info):
    from .activity_poll_batch import main as poll_main # type: ignore
    return poll_main(batch_info)

@app.activity_trigger(input_name="result_data")
def write_output_activity(result_data):
    from .activity_write_output import main as write_main # type: ignore
    return write_main(result_data)
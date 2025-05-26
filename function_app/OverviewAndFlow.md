# Azure Durable Functions: Audio Batch Transcription Workflow

This workflow enables automated audio transcription via Azure Speech Services, orchestrated using Durable Functions and Managed Identity.

---

## 🔁 Process Flow

### 1. Blob Trigger

- An audio file is uploaded to the `audio-in/` container.
- This triggers the `audio_blob_trigger` function.
- The function generates a SAS URL and starts the Durable Function orchestrator.

### 2. Generate SAS URL

- `generate_sas_url()` creates a read-only URL using Azure Blob Storage and a user delegation key.
- The URL expires in 1 hour and is only used by the Azure Speech Service.

### 3. Start Transcription Job

- `start_batch_activity()` sends a `POST` request to:

  ```http
  https://<region>.api.cognitive.microsoft.com/speechtotext/v3.1/transcriptions
  ```json

- Payload:

```json
{
  "displayName": "Transcription - filename.wav",
  "locale": "en-GB",
  "contentUrls": ["<sas_url>"],
  "properties": {
    "wordLevelTimestampsEnabled": true,
    "punctuationMode": "DictatedAndAutomatic",
    "profanityFilterMode": "Masked"
  }
}
```

- A `202 Accepted` response returns a `Location` header URL for polling.

### 4. Poll for Completion

- `poll_batch_activity()` repeatedly polls the `Location` URL via `GET`.
- It waits until the transcription status is `"Succeeded"`.
- It then downloads the JSON transcript from the result URL.

### 5. Write Output

- `write_output_activity()` saves the result into the `audio-transcript-out` container.
- The JSON file name is derived from the original blob, e.g. `meeting.wav → meeting.json`.

---

## 🧩 Function Summary

| Function                        | Description                                               |
|---------------------------------|-----------------------------------------------------------|
| `audio_blob_trigger`           | Blob-triggered entrypoint. Starts orchestration.          |
| `generate_sas_url`             | Generates secure read-only blob access URL.               |
| `start_batch_activity`         | Submits batch job to Azure Speech API.                    |
| `poll_batch_activity`          | Polls the batch transcription job until complete.         |
| `write_output_activity`        | Writes result to blob storage.                            |
| `audio_processing_orchestrator`| Coordinates the above functions via Durable Functions.    |

---

## 🔐 Security Notes

- **Auth**: Uses `DefaultAzureCredential` (Managed Identity preferred).
- **Storage**: SAS URLs are scoped, time-limited, and only for read access.
- **Isolation**: No user secrets or keys are exposed in code or logs.

---

## 🗺️ Mermaid Diagram

```mermaid
flowchart TD
    A[Blob uploaded to audio-in/] --> B[Trigger Function: audio_blob_trigger]
    B --> C[Generate SAS URL]
    C --> D[Start Orchestration: audio_processing_orchestrator]
    D --> E[start_batch_activity]
    E --> F[Azure Speech API (POST)]
    F --> G[Transcription Location URL]
    G --> H[poll_batch_activity]
    H --> I[Check Status Loop]
    I -->|Success| J[Download Transcript JSON]
    J --> K[write_output_activity]
    K --> L[Save to audio-transcript-out container]
```

---

## 📁 Example Output

Container: audio-transcript-out/
File: interview_2025-05-25.json
Contents: {
  "result": "real-transcript-content",
  "blob_name": "interview_2025-05-25.wav"
}

'###

Container: audio-transcript-out/
File: interview_2025-05-25.json
Contents: {
  "result": "real-transcript-content",
  "blob_name": "interview_2025-05-25.wav"
}

###

---

## ✅ Requirements

- Azure Blob Storage with input/output containers
- Azure Speech resource (v3.1 batch endpoint)
- Azure Function App with Durable Functions
- Managed Identity enabled for secure token acquisition

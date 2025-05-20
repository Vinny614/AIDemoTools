import json
import logging
import os
from enum import Enum
from typing import Optional

import azure.functions as func
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI
from pydantic import BaseModel, Field
from src.components.speech import (
    AOAI_WHISPER_MIME_TYPE_MAPPER,
    BATCH_TRANSCRIPTION_MIME_TYPE_MAPPER,
    AzureSpeechTranscriber,
    is_phrase_start_time_match,
)
from src.components.utils import base64_bytes_to_buffer, get_file_ext_and_mime_type
from src.helpers.common import MeasureRunTime
from src.result_enrichment.common import is_value_in_content
from src.schema import LLMResponseBaseModel

load_dotenv()

bp_call_center_audio_analysis = func.Blueprint()
FUNCTION_ROUTE = "call_center_audio_analysis"

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

SPEECH_ENDPOINT = os.getenv("SPEECH_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")
AOAI_WHISPER_DEPLOYMENT = os.getenv("AOAI_WHISPER_DEPLOYMENT")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")

# Setup components
aoai_whisper_async_client = AsyncAzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_WHISPER_DEPLOYMENT,
    azure_ad_token_provider=token_provider,
    api_version="2024-06-01",
)
transcriber = AzureSpeechTranscriber(
    speech_endpoint=SPEECH_ENDPOINT,
    azure_ad_token_provider=token_provider,
    aoai_whisper_async_client=aoai_whisper_async_client,
)
fast_transcription_definition = {
    "locales": ["en-US"],
    "profanityFilterMode": "Masked",
    "diarizationEnabled": False,
    "wordLevelTimestampsEnabled": True,
}
aoai_whisper_kwargs = {
    "language": "en",
    "prompt": None,
    "temperature": None,
    "timeout": 60,
}

aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    azure_ad_token_provider=token_provider,
    api_version="2024-06-01",
    timeout=30,
    max_retries=0,
)

TRANSCRIPTION_METHOD_TO_MIME_MAPPER = {
    "fast": BATCH_TRANSCRIPTION_MIME_TYPE_MAPPER,
    "aoai_whisper": AOAI_WHISPER_MIME_TYPE_MAPPER,
}

# General enums for audio analysis
class ParticipantCooperationEnum(Enum):
    Cooperative = "Cooperative"
    Uncooperative = "Uncooperative"
    Unknown = "Unknown"

PARTICIPANT_COOPERATION_VALUES = [e.value for e in ParticipantCooperationEnum]

class ParticipantSentimentEnum(Enum):
    Calm = "Calm"
    Neutral = "Neutral"
    Distressed = "Distressed"
    Unknown = "Unknown"

PARTICIPANT_SENTIMENT_VALUES = [e.value for e in ParticipantSentimentEnum]

class RawKeyword(LLMResponseBaseModel):
    keyword: str = Field(
        description="A keyword or phrase extracted from the audio footage. This should match the wording in the transcription.",
        examples=["car alarm", "male voice", "glass breaking"],
    )
    timestamp: str = Field(
        description="The timestamp of the sentence from which the keyword was uttered.",
        examples=["0:18"],
    )

class ProcessedKeyWord(RawKeyword):
    keyword_matched_to_transcription_sentence: bool = Field(
        description="Whether the keyword was matched to a single sentence in the transcription.",
    )
    full_sentence_text: Optional[str] = Field(
        default=None,
        description="The full text of the sentence in which the keyword was uttered.",
    )
    sentence_confidence: Optional[float] = Field(
        default=None,
        description="The confidence score of the sentence from which the keyword was extracted.",
    )
    sentence_start_time_secs: Optional[float] = Field(
        default=None,
        description="The start time of the sentence in the audio recording.",
    )
    sentence_end_time_secs: Optional[float] = Field(
        default=None,
        description="The end time of the sentence in the audio recording.",
    )

class LLMRawResponseModel(LLMResponseBaseModel):
    """
    JSON schema for the LLM to follow for analysis of investigative or legal audio recordings.
    """

    audio_summary: str = Field(
        description="A summary of the audio footage, including main events, participants, and key details. No more than 20 words.",
        examples=[
            "Two people discuss the incident; a car alarm is heard in the background.",
        ],
    )
    participant_cooperation: ParticipantCooperationEnum = Field(
        description=f"Were participants cooperative? Options: {PARTICIPANT_COOPERATION_VALUES}.",
        examples=[PARTICIPANT_COOPERATION_VALUES[0]],
    )
    participant_sentiment: ParticipantSentimentEnum = Field(
        description=f"The general mood or emotional state of participants. Options: {PARTICIPANT_SENTIMENT_VALUES}.",
        examples=[PARTICIPANT_SENTIMENT_VALUES[0]],
    )
    next_action: Optional[str] = Field(
        description="Recommended next step after reviewing this audio. No more than 20 words. If none, return null.",
        examples=["Review security camera footage from nearby store."],
    )
    next_action_sentence_timestamp: Optional[str] = Field(
        description="The timestamp where the next action is mentioned.",
        examples=["6:12"],
    )
    keywords: list[RawKeyword] = Field(
        description=(
            "A list of keywords or key phrases from the audio, such as names, locations, objects, or events. "
            "Each should include the keyword and the timestamp of the sentence where it was said."
        ),
        examples=[
            [
                {"keyword": "car alarm", "timestamp": "0:18"},
                {"keyword": "glass breaking", "timestamp": "1:42"},
                {"keyword": "police siren", "timestamp": "4:29"},
            ]
        ],
    )

class ProcessedResultModel(LLMRawResponseModel):
    keywords: list[ProcessedKeyWord] = Field(
        description=(
            "A list of key phrases from the audio with metadata, including sentence match, confidence, and timing."
        ),
    )

class FunctionReponseModel(BaseModel):
    success: bool = Field(
        default=False, description="Indicates whether the pipeline was successful."
    )
    result: Optional[ProcessedResultModel] = Field(
        default=None, description="The final result of the pipeline."
    )
    error_text: Optional[str] = Field(
        default=None,
        description="If an error occurred, this field will contain the error message.",
    )
    speech_extracted_text: Optional[str] = Field(
        default=None,
        description="The raw & formatted text content extracted by Azure AI Speech.",
    )
    speech_raw_response: Optional[list | dict] = Field(
        default=None, description="The raw API response from Azure AI Speech."
    )
    speech_time_taken_secs: Optional[float] = Field(
        default=None,
        description="The time taken to transcribe the text using Azure AI Speech.",
    )
    llm_input_messages: Optional[list[dict]] = Field(
        default=None, description="The messages that were sent to the LLM."
    )
    llm_reply_message: Optional[dict] = Field(
        default=None, description="The message that was received from the LLM."
    )
    llm_raw_response: Optional[str] = Field(
        default=None, description="The raw text response from the LLM."
    )
    llm_time_taken_secs: Optional[float] = Field(
        default=None, description="The time taken to receive a response from the LLM."
    )
    func_time_taken_secs: Optional[float] = Field(
        default=None, description="The total time taken to process the request."
    )

LLM_SYSTEM_PROMPT = (
    "You are an assistant specializing in summarizing and analyzing investigative, legal, or incident-related audio footage.\n"
    "Your task is to review any kind of audio recording (statements, interviews, on-scene or ambient audio, etc) and extract all key information, including main events, participants, sounds, and more.\n"
    "Respond ONLY in the following JSON format. Do not include any commentary or explanation.\n"
    "{\n"
    '  "audio_summary": "A summary of the audio footage, including main events, participants, and key details. No more than 20 words.",\n'
    '  "participant_cooperation": "Cooperative",\n'
    '  "participant_sentiment": "Calm",\n'
    '  "next_action": "Recommended next step after reviewing this audio. No more than 20 words. If none, return null.",\n'
    '  "next_action_sentence_timestamp": "6:12",\n'
    '  "keywords": [\n'
    '    {"keyword": "car alarm", "timestamp": "0:18"},\n'
    '    {"keyword": "glass breaking", "timestamp": "1:42"},\n'
    '    {"keyword": "police siren", "timestamp": "4:29"}\n'
    "  ]\n"
    "}"
)

@bp_call_center_audio_analysis.route(route=FUNCTION_ROUTE)
async def call_center_audio_analysis(
    req: func.HttpRequest,
) -> func.HttpResponse:
    logging.info(f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request.")
    output_model = FunctionReponseModel(success=False)
    try:
        error_text = "An error occurred during processing."
        error_code = 422

        func_timer = MeasureRunTime()
        func_timer.start()

        # Check the request body
        logging.info("Step 1: Checking request body")
        request_json_content = json.loads(req.files["json"].read().decode("utf-8"))

        logging.info("Step 2: Validating transcription method")
        transcription_method = request_json_content["method"]
        logging.info(f"Step 3: Got transcription method: {transcription_method}")
        if transcription_method not in TRANSCRIPTION_METHOD_TO_MIME_MAPPER:
            output_model.error_text = f"Invalid transcription method `{transcription_method}`. Please use one of {list(TRANSCRIPTION_METHOD_TO_MIME_MAPPER.keys())}"
            logging.exception(output_model.error_text)
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=error_code,
            )
        # Audio file & type
        valid_mime_to_filetype_mapper = TRANSCRIPTION_METHOD_TO_MIME_MAPPER[
            transcription_method
        ]
        error_text = "Invalid audio file. Please submit a file with a valid filename and content type."
        audio_file = req.files["audio"]
        audio_file_b64 = audio_file.read()
        audio_file_ext, _audio_file_content_type = get_file_ext_and_mime_type(
            valid_mimes_to_file_ext_mapper=valid_mime_to_filetype_mapper,
            filename=audio_file.filename,
            content_type=audio_file.content_type,
        )
        audio_filename = (
            audio_file.filename if audio_file.filename else f"file.{audio_file_ext}"
        )
        # Get the transcription result
        error_text = "An error occurred during audio transcription."
        error_code = 500
        logging.info("Step 4: Starting audio transcription")
        with MeasureRunTime() as speech_timer:
            if transcription_method == "fast":
                transcription, raw_transcription_api_response = (
                    await transcriber.get_fast_transcription_async(
                        audio_file=audio_file_b64,
                        definition=fast_transcription_definition,
                    )
                )
            else:
                audio_file = base64_bytes_to_buffer(
                    b64_str=audio_file_b64, name=audio_filename
                )
                transcription, raw_transcription_api_response = (
                    await transcriber.get_aoai_whisper_transcription_async(
                        audio_file=audio_file,
                        **aoai_whisper_kwargs,
                    )
                )
        logging.info("Step 5: Audio transcription complete")
        logging.debug(f"Raw transcription API response: {raw_transcription_api_response}")

        logging.info("Step 6: Formatting transcription text")
        formatted_transcription_text = transcription.to_formatted_str(
            transcription_prefix_format="Language: {language}\nDuration: {formatted_duration} minutes\n\nAudio:\n",
            phrase_format="[{start_min}:{start_sub_sec}] {auto_phrase_source_name} {auto_phrase_source_id}: {display_text}",
        )
        output_model.speech_extracted_text = formatted_transcription_text
        output_model.speech_raw_response = raw_transcription_api_response
        output_model.speech_time_taken_secs = speech_timer.time_taken
        # LLM input
        error_text = "An error occurred while creating the LLM input messages."
        logging.info("Step 7: Preparing LLM input messages")
        input_messages = [
            {
                "role": "system",
                "content": LLM_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": formatted_transcription_text,
            },
        ]
        output_model.llm_input_messages = input_messages
        logging.debug(f"LLM input messages: {input_messages}")
        # Send request to LLM
        error_text = "An error occurred when sending the LLM request."
        logging.info("Step 8: Sending request to LLM")
        with MeasureRunTime() as llm_timer:
            llm_result = aoai_client.chat.completions.create(
                messages=input_messages,
                model=AOAI_LLM_DEPLOYMENT,
                response_format={"type": "json_object"},  # Ensure we get JSON responses
            )
        output_model.llm_time_taken_secs = llm_timer.time_taken
        logging.info("Step 9: LLM response received")
        logging.debug(f"Raw LLM response: {llm_result.choices[0].message.content}")
        # Validate response schema
        error_text = "An error occurred when validating the LLM's returned response into the expected schema."
        output_model.llm_reply_message = llm_result.choices[0].to_dict()
        output_model.llm_raw_response = llm_result.choices[0].message.content
        logging.info("Step 10: Parsing LLM response JSON")
        try:
            llm_structured_response = LLMRawResponseModel(
                **json.loads(llm_result.choices[0].message.content)
            )
        except Exception as e:
            logging.error(f"Failed to parse LLM response: {e}")
            logging.error(f"Raw LLM response: {llm_result.choices[0].message.content}")
            output_model.error_text = f"Failed to parse LLM response: {e}"
            output_model.func_time_taken_secs = func_timer.stop()
            return func.HttpResponse(
                body=output_model.model_dump_json(),
                mimetype="application/json",
                status_code=500,
            )
        # Post-process keywords
        error_text = "An error occurred when post-processing the keywords."
        processed_keywords = []
        for keyword in llm_structured_response.keywords:
            # Find the sentence in the transcription that contains the keyword.
            keyword_sentence_start_time_secs = int(
                keyword.timestamp.split(":")[0]
            ) * 60 + int(keyword.timestamp.split(":")[1])
            matching_phrases = [
                phrase
                for phrase in transcription.phrases
                if is_value_in_content(
                    keyword.keyword.lower(), phrase.display_text.lower()
                )
                and is_phrase_start_time_match(
                    expected_start_time_secs=keyword_sentence_start_time_secs,
                    phrase=phrase,
                    start_time_tolerance_secs=1,
                )
            ]
            if len(matching_phrases) == 1:
                processed_keywords.append(
                    ProcessedKeyWord(
                        **keyword.dict(),
                        keyword_matched_to_transcription_sentence=True,
                        full_sentence_text=matching_phrases[0].display_text,
                        sentence_confidence=matching_phrases[0].confidence,
                        sentence_start_time_secs=matching_phrases[0].start_secs,
                        sentence_end_time_secs=matching_phrases[0].end_secs,
                    )
                )
            else:
                processed_keywords.append(
                    ProcessedKeyWord(
                        **keyword.dict(),
                        keyword_matched_to_transcription_sentence=False,
                    )
                )
        # Construct processed model, replacing the raw keywords with the processed keywords
        llm_structured_response_dict = llm_structured_response.dict()
        llm_structured_response_dict.pop("keywords")
        output_model.result = ProcessedResultModel(
            **llm_structured_response_dict,
            keywords=processed_keywords,
        )
        output_model.success = True
        output_model.func_time_taken_secs = func_timer.stop()
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as _e:
        logging.exception(f"An error occurred: {_e}")
        output_model.error_text = f"{error_text} Details: {_e}"
        output_model.func_time_taken_secs = func_timer.stop()
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=error_code,
        )

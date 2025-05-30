import json
import logging
import os
import time
import random
from typing import Optional

import azure.functions as func
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError
from src.components.doc_intelligence import (
    VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES,
    DefaultDocumentFigureProcessor,
    DefaultDocumentPageProcessor,
    DocumentIntelligenceProcessor,
    convert_processed_di_docs_to_openai_message,
)
from src.helpers.common import MeasureRunTime
from src.helpers.data_loading import load_visual_obj_bytes_to_pil_imgs_dict
from src.schema import LLMResponseBaseModel

load_dotenv()

bp_doc_intel_extract_city_names = func.Blueprint()
FUNCTION_ROUTE = "doc_intel_extract_city_names"

aoai_token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_LLM_DEPLOYMENT = os.getenv("AOAI_LLM_DEPLOYMENT")

DOC_INTEL_MODEL_ID = "prebuilt-read"

MAX_LLM_RETRIES = 5

# Clients

di_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=DefaultAzureCredential(),
    api_version="2024-11-30",
)
aoai_client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    azure_deployment=AOAI_LLM_DEPLOYMENT,
    azure_ad_token_provider=aoai_token_provider,
    api_version="2024-06-01",
    timeout=90,
    max_retries=2,
)

# Document Intelligence Processor

doc_intel_result_processor = DocumentIntelligenceProcessor(
    page_processor=DefaultDocumentPageProcessor(page_img_order="after"),
    figure_processor=DefaultDocumentFigureProcessor(output_figure_img=False),
)

# Pydantic Models

class StructuredPerson(BaseModel):
    name: str
    role: Optional[str] = None
    affiliations: list[str] = Field(default_factory=list)
    vehicles: list[dict] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    contact_info: list[str] = Field(default_factory=list)

class StructuredEvent(BaseModel):
    type: str
    datetime: str
    location: Optional[str] = None
    suspects: list[str] = Field(default_factory=list)
    officers: list[str] = Field(default_factory=list)
    vehicles: list[dict] = Field(default_factory=list)
    witness_description: Optional[str] = ""

class LLMInvestigationExtractionModel(LLMResponseBaseModel):
    people: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)
    roles: dict[str, str] = Field(default_factory=dict)
    times: list[str] = Field(default_factory=list)
    time_event_map: dict[str, str] = Field(default_factory=dict)
    locations: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    vehicles: list[dict] = Field(default_factory=list)
    weapons: list[str] = Field(default_factory=list)
    descriptions: list[str] = Field(default_factory=list)
    contact_info: list[str] = Field(default_factory=list)
    summary: str = ""
    structured_people: list[StructuredPerson] = Field(default_factory=list)
    structured_events: list[StructuredEvent] = Field(default_factory=list)

class FunctionResponseModel(BaseModel):
    success: bool = Field(False)
    result: Optional[LLMInvestigationExtractionModel] = None
    func_time_taken_secs: Optional[float] = None
    error_text: Optional[str] = None
    di_extracted_text: Optional[str] = None
    di_raw_response: Optional[dict] = None
    di_time_taken_secs: Optional[float] = None
    llm_input_messages: Optional[list[dict]] = None
    llm_reply_message: Optional[dict] = None
    llm_raw_response: Optional[str] = None
    llm_time_taken_secs: Optional[float] = None

def normalize_llm_response(raw_json: dict) -> dict:
    fields = [
        ("people", list),
        ("relationships", list),
        ("roles", dict),
        ("time_event_map", dict),
        ("locations", list),
        ("events", list),
        ("vehicles", list),
        ("weapons", list),
        ("descriptions", list),
        ("contact_info", list),
        ("summary", str),
        ("structured_people", list),
        ("structured_events", list),
    ]

    for key, expected_type in fields:
        if key not in raw_json or not isinstance(raw_json[key], expected_type):
            raw_json[key] = [] if expected_type in [list, dict] else ""

    # Patch contact_info
    if isinstance(raw_json["contact_info"], dict):
        raw_json["contact_info"] = [
            str(v) for v in raw_json["contact_info"].values() if isinstance(v, str)
        ]

    # Vehicles cleanup
    raw_json["vehicles"] = [
        v for v in raw_json["vehicles"]
        if isinstance(v, dict) and "description" in v and "reg_number" in v
    ]

    # Witness descriptions
    for event in raw_json["structured_events"]:
        if isinstance(event.get("witness_description"), list):
            event["witness_description"] = "; ".join(map(str, event["witness_description"]))

        if not raw_json.get("times"):
        raw_json["times"] = list(raw_json.get("time_event_map", {}).keys())

    return raw_json

LLM_INVESTIGATION_PROMPT = (
    "Return a valid JSON object that matches this schema. "
    "You are a digital case analyst reviewing a police report.

"
    "Return a structured JSON object that follows this schema and captures key information and relationships.

"
    "Flat fields:
"
    "- people: List all individuals mentioned in the report.
"
    "- roles: Map each person's name to their role (e.g., 'Daniel Smith': 'suspect').
"
    "- relationships: Describe how individuals are connected (e.g., 'X is married to Y', 'X is solicitor for Y').
"
    "- locations: Include addresses, cities, or institutions.
"
    "- events: Major incidents (e.g., robbery, assault).
"
    "- vehicles: Each entry must be an object with keys 'description', 'reg_number', and optionally 'owner'.
"
    "- weapons: Descriptions of any weapons or offensive items.
"
    "- descriptions: A list of clothing or physical descriptions as plain strings.
"
    "- contact_info: Phone numbers, emails, etc.
"
    "- time_event_map: A dictionary mapping each timestamp to what happened.
"
    "- summary: A brief, readable summary of the incident.

"
    "Structured fields:
"
    "- structured_people: Each object must have name, role, affiliations, vehicles (as objects), contact_info, and locations.
"
    "- structured_events: Each object must include type, datetime, location, suspects, officers, vehicles (as objects), and witness_description (as a single string).

"
    "Guidelines:
"
    "- Use exact field names from this schema.
"
    "- Do not rename keys (e.g., do not use 'vehicles_involved' instead of 'vehicles').
"
    "- Ensure relationships is a list of strings.
"
    "- Ensure descriptions are a list of plain strings, not objects.
"
    "- witness_description must be a single string.
"
    "- Output must be valid JSON matching this structure.
"
    "- Ensure contact_info is a list of plain strings, (never a dictionary or object).
"
)

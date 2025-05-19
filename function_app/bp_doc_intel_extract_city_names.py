import json
import logging
import os
from typing import Optional

import azure.functions as func
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field
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
    timeout=60,
    max_retries=2,
)

doc_intel_result_processor = DocumentIntelligenceProcessor(
    page_processor=DefaultDocumentPageProcessor(
        page_img_order="after",
    ),
    figure_processor=DefaultDocumentFigureProcessor(
        output_figure_img=False,
    ),
)


class StructuredPerson(BaseModel):
    name: str
    role: Optional[str] = None
    affiliations: Optional[list[str]] = None
    vehicles: Optional[list[dict]] = None
    locations: Optional[list[str]] = None
    contact_info: Optional[list[str]] = None


class StructuredEvent(BaseModel):
    type: str
    datetime: str
    location: Optional[str] = None
    suspects: Optional[list[str]] = None
    officers: Optional[list[str]] = None
    vehicles: Optional[list[dict]] = None
    witness_description: Optional[str] = None


class LLMInvestigationExtractionModel(LLMResponseBaseModel):
    people: list[str]
    relationships: list[str]
    roles: dict[str, str]
    times: list[str]
    time_event_map: dict[str, str]
    locations: list[str]
    events: list[str]
    vehicles: list[dict]
    weapons: list[str]
    descriptions: list[str]
    contact_info: list[str]
    summary: str
    structured_people: Optional[list[StructuredPerson]] = None
    structured_events: Optional[list[StructuredEvent]] = None


class FunctionResponseModel(BaseModel):  # Typo fixed here
    """
    Response model for the doc_intel_extract_city_names function.
    """

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
    """
    Normalize and sanitize the LLM response to match the expected schema.
    """
    # Always set defaults for ALL expected fields
    raw_json.setdefault("people", [])
    raw_json.setdefault("relationships", [])
    raw_json.setdefault("roles", {})
    raw_json.setdefault("times", list(raw_json.get("time_event_map", {}).keys()))
    raw_json.setdefault("time_event_map", {})
    raw_json.setdefault("locations", [])
    raw_json.setdefault("events", [])
    raw_json.setdefault("vehicles", [])
    raw_json.setdefault("weapons", [])
    raw_json.setdefault("descriptions", [])
    raw_json.setdefault("contact_info", [])
    raw_json.setdefault("summary", "")
    raw_json.setdefault("structured_people", [])
    raw_json.setdefault("structured_events", [])

    # Vehicles must be a list of dicts with specific keys

    if "vehicles" in raw_json:
        raw_json["vehicles"] = [
            v
            for v in raw_json["vehicles"]
            if isinstance(v, dict) and "description" in v and "reg_number" in v
        ]
    # Descriptions must be a list of strings

    if "descriptions" in raw_json:
        raw_json["descriptions"] = [
            d for d in raw_json["descriptions"] if isinstance(d, str)
        ]

    # contact_info must be a list of strings, never a dict
    if "contact_info" in raw_json:
        if isinstance(raw_json["contact_info"], dict):
            raw_json["contact_info"] = [
                str(v) for v in raw_json["contact_info"].values() if isinstance(v, str)
            ]
        elif not isinstance(raw_json["contact_info"], list):
            raw_json["contact_info"] = []

    # Fix structured_people entries.

    if "structured_people" in raw_json:
        for p in raw_json["structured_people"]:
            # contact_info: must be a list
            if "contact_info" in p:
                if isinstance(p["contact_info"], dict):
                    p["contact_info"] = [
                        str(v) for v in p["contact_info"].values() if isinstance(v, str)
                    ]
                elif not isinstance(p["contact_info"], list):
                    p["contact_info"] = []

            # affiliations: must be a list.
            if "affiliations" in p:
                if isinstance(p["affiliations"], str):
                    p["affiliations"] = (
                        [] if not p["affiliations"].strip() else [p["affiliations"]]
                    )
                elif not isinstance(p["affiliations"], list):
                    p["affiliations"] = []

            # vehicles: must be a list of dicts

            if "vehicles" in p and isinstance(p["vehicles"], list):
                p["vehicles"] = [v for v in p["vehicles"] if isinstance(v, dict)]

    # Fix witness_description in structured_events..

    if "structured_events" in raw_json:
        for e in raw_json["structured_events"]:
            if isinstance(e.get("witness_description"), list):
                e["witness_description"] = "; ".join(map(str, e["witness_description"]))
            if "type" in e and not isinstance(e["type"], str):
                e["type"] = ""

    return raw_json


LLM_INVESTIGATION_PROMPT = (
    "You are a digital case analyst reviewing a police report.\n\n"
    "Return a structured JSON object that follows this schema and captures key information and relationships.\n\n"
    "Flat fields:\n"
    "- people: List all individuals mentioned in the report.\n"
    "- roles: Map each person's name to their role (e.g., 'Daniel Smith': 'suspect').\n"
    "- relationships: Describe how individuals are connected (e.g., 'X is married to Y', 'X is solicitor for Y').\n"
    "- locations: Include addresses, cities, or institutions.\n"
    "- events: Major incidents (e.g., robbery, assault).\n"
    "- vehicles: Each entry must be an object with keys 'description', 'reg_number', and optionally 'owner'.\n"
    "- weapons: Descriptions of any weapons or offensive items.\n"
    "- descriptions: A list of clothing or physical descriptions as plain strings.\n"
    "- contact_info: Phone numbers, emails, etc.\n"
    "- time_event_map: A dictionary mapping each timestamp to what happened.\n"
    "- summary: A brief, readable summary of the incident.\n\n"
    "Structured fields:\n"
    "- structured_people: Each object must have name, role, affiliations, vehicles (as objects), contact_info, and locations.\n"
    "- structured_events: Each object must include type, datetime, location, suspects, officers, vehicles (as objects), and witness_description (as a single string).\n\n"
    "Guidelines:\n"
    "- Use exact field names from this schema.\n"
    "- Do not rename keys (e.g., do not use 'vehicles_involved' instead of 'vehicles').\n"
    "- Ensure relationships is a list of strings.\n"
    "- Ensure descriptions are a list of plain strings, not objects.\n"
    "- witness_description must be a single string.\n"
    "- Output must be valid JSON matching this structure.\n"
    "- Ensure contact_info is a list of plain strings, (never a dictionary or object).\n"
)


@bp_doc_intel_extract_city_names.route(route=FUNCTION_ROUTE)
def doc_intel_extract_city_names(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function to extract structured city name and incident information from a police report.
    """
    logging.info(f"Python HTTP trigger function `{FUNCTION_ROUTE}` received a request.")
    output_model = FunctionResponseModel(success=False)
    try:
        error_text = "An error occurred during processing."
        error_code = 422

        func_timer = MeasureRunTime()
        func_timer.start()

        mime_type = req.headers.get("Content-Type")
        if mime_type not in VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES:
            return func.HttpResponse(
                f"This function only supports a Content-Type of {', '.join(VALID_DI_PREBUILT_READ_LAYOUT_MIME_TYPES)}. Supplied file is of type {mime_type}",
                status_code=422,
            )

        req_body = req.get_body()
        if len(req_body) == 0:
            return func.HttpResponse(
                "Please provide a base64 encoded PDF in the request body.",
                status_code=422,
            )

        error_text = "An error occurred during image extraction."
        error_code = 500
        logging.info("Extracting images from document...")
        doc_page_imgs = load_visual_obj_bytes_to_pil_imgs_dict(
            req_body, mime_type, starting_idx=1, pdf_img_dpi=100
        )

        error_text = "An error occurred during Document Intelligence extraction."
        with MeasureRunTime() as di_timer:
            poller = di_client.begin_analyze_document(
                model_id=DOC_INTEL_MODEL_ID,
                analyze_request=AnalyzeDocumentRequest(bytes_source=req_body),
            )
            di_result = poller.result()
            output_model.di_raw_response = di_result.as_dict()
            processed_content_docs = doc_intel_result_processor.process_analyze_result(
                analyze_result=di_result,
                doc_page_imgs=doc_page_imgs,
                on_error="raise",
            )
            merged_processed_content_docs = (
                doc_intel_result_processor.merge_adjacent_text_content_docs(
                    processed_content_docs
                )
            )
        output_model.di_extracted_text = "\n".join(
            doc.content for doc in processed_content_docs if doc.content is not None
        )
        output_model.di_time_taken_secs = di_timer.time_taken

        error_text = "An error occurred while creating the LLM input messages."
        content_openai_message = convert_processed_di_docs_to_openai_message(
            merged_processed_content_docs, role="user"
        )
        input_messages = [
            {"role": "system", "content": LLM_INVESTIGATION_PROMPT},
            content_openai_message,
        ]
        output_model.llm_input_messages = input_messages

        error_text = "An error occurred when sending the LLM request."
        with MeasureRunTime() as llm_timer:
            llm_result = aoai_client.chat.completions.create(
                messages=input_messages,
                model=AOAI_LLM_DEPLOYMENT,
                response_format={"type": "json_object"},
            )
        output_model.llm_time_taken_secs = llm_timer.time_taken

        error_text = "An error occurred when validating the LLM's returned response into the expected schema."
        output_model.llm_reply_message = llm_result.choices[0].to_dict()
        output_model.llm_raw_response = llm_result.choices[0].message.content

        raw_json = json.loads(llm_result.choices[0].message.content)
        raw_json = normalize_llm_response(raw_json)
        llm_structured_response = LLMInvestigationExtractionModel(**raw_json)
        output_model.result = llm_structured_response

        output_model.success = True
        output_model.func_time_taken_secs = func_timer.stop()
        logging.info("Function completed successfully.")
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=200,
        )
    except Exception as _e:
        output_model.success = False
        output_model.error_text = error_text
        output_model.func_time_taken_secs = func_timer.stop()
        logging.exception(output_model.error_text)
        return func.HttpResponse(
            body=output_model.model_dump_json(),
            mimetype="application/json",
            status_code=error_code,
        )

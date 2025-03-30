# /mnt/LAB/rag-chat-poc/backend/main.py

import os
import logging
from fastapi import FastAPI, HTTPException, Path as FastAPIPath # Rename Path to avoid conflict
from dotenv import load_dotenv
import google.generativeai as genai
import httpx # Added for OpenRouter REST API calls
from pydantic import BaseModel, Field # Import Pydantic
from typing import List, Optional, Dict, Literal
from pathlib import Path # Use pathlib for path operations
import pdfplumber # PDF parsing
from docx import Document as DocxDocument # DOCX parsing (renamed)
import markdown as md_parser # Markdown parsing
import pandas as pd # CSV/Excel parsing
import traceback # For detailed error logging
import asyncio # For potential async file operations later
import json # For OpenRouter payload
import tiktoken # Added for token counting
from cachetools import cached, TTLCache # For caching OpenRouter model data

# --- Configuration & Setup ---
# Set logging level to DEBUG to capture detailed info for pricing issue
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)
load_dotenv()
logger.info(".env file loaded.")

# --- Project Directory Setup ---
PROJECTS_DIR = Path(__file__).parent.parent / "projects"
logger.info(f"Projects directory set to: {PROJECTS_DIR}")
if not PROJECTS_DIR.is_dir():
    logger.error(f"Projects directory not found at {PROJECTS_DIR}")
    raise RuntimeError(f"Projects directory not found: {PROJECTS_DIR}")

# --- API Key Retrieval ---
google_api_key = os.getenv("GOOGLE_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

google_configured = False
if google_api_key and google_api_key != "YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE":
    logger.info("Google API Key loaded.")
    try:
        genai.configure(api_key=google_api_key)
        logger.info("Google Generative AI client configured.")
        google_configured = True
    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI client: {e}")
else:
    logger.warning("GOOGLE_API_KEY not found or is placeholder in .env file. Google models will be unavailable.")

openrouter_configured = False
if openrouter_api_key and openrouter_api_key != "YOUR_OPENROUTER_API_KEY_HERE":
    logger.info("OpenRouter API Key loaded.")
    openrouter_configured = True
else:
    logger.warning("OPENROUTER_API_KEY not found or is placeholder in .env file. OpenRouter models will be unavailable.")

if not google_configured and not openrouter_configured:
     raise RuntimeError("No valid API keys found for either Google or OpenRouter. Application cannot function.")

# --- Model Loading Function ---
MODELS_FILE_PATH = Path(__file__).parent / "models.txt"

def _load_available_models() -> Dict[str, List[str]]:
    """Loads available models from the models.txt file."""
    global google_configured, openrouter_configured # Allow modification if needed on refresh
    logger.info(f"Attempting to load models from {MODELS_FILE_PATH}...")

    # Check API keys status (still needed to know if provider is usable)
    google_api_key = os.getenv("GOOGLE_API_KEY") # Keep checking keys
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    google_configured = False
    if google_api_key and google_api_key != "YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE":
        try:
            # Re-configure in case key changed, though genai might handle this internally
            genai.configure(api_key=google_api_key)
            logger.info("Google Generative AI client re-configured (or confirmed).")
            google_configured = True
        except Exception as e:
            logger.error(f"Failed to re-configure Google Generative AI client during refresh: {e}")
    else:
        logger.warning("GOOGLE_API_KEY not found or is placeholder during refresh.")

    openrouter_configured = False
    if openrouter_api_key and openrouter_api_key != "YOUR_OPENROUTER_API_KEY_HERE":
        logger.info("OpenRouter API Key confirmed during refresh.")
        openrouter_configured = True
    else:
        logger.warning("OPENROUTER_API_KEY not found or is placeholder.")

    # Load models from models.txt
    loaded_models = {}
    if not MODELS_FILE_PATH.is_file():
        logger.error(f"Models file not found at {MODELS_FILE_PATH}. No models loaded.")
        return {}

    try:
        with open(MODELS_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(':', 1)
                if len(parts) == 2:
                    provider, models_str = parts
                    provider = provider.strip().lower()
                    models_list = [model.strip() for model in models_str.split(',') if model.strip()]
                    if not models_list:
                         logger.warning(f"No models listed for provider '{provider}' in {MODELS_FILE_PATH.name}")
                         continue

                    # Only add models if the corresponding provider is configured
                    if provider == "google" and google_configured:
                        loaded_models["google"] = models_list
                        logger.info(f"Loaded Google models from file: {models_list}")
                    elif provider == "openrouter" and openrouter_configured:
                        loaded_models["openrouter"] = models_list
                        logger.info(f"Loaded OpenRouter models from file: {models_list}")
                    else:
                         logger.warning(f"Provider '{provider}' found in {MODELS_FILE_PATH.name}, but its API key is not configured or valid. Models skipped.")
                else:
                    logger.warning(f"Skipping malformed line in {MODELS_FILE_PATH.name}: {line}")

    except Exception as e:
        logger.error(f"Error reading or parsing {MODELS_FILE_PATH.name}: {e}", exc_info=True)
        return {} # Return empty if file reading fails

    if not loaded_models:
         logger.warning(f"No valid and configured models loaded from {MODELS_FILE_PATH.name}.")

    return loaded_models

# --- Initial Model Load ---
AVAILABLE_MODELS = _load_available_models()


# --- Configuration Refresh Logic ---
def refresh_configuration():
    """Reloads API keys from .env and models from models.txt."""
    global AVAILABLE_MODELS
    logger.info("Refreshing configuration...")
    try:
        # Force reload of .env file for API keys
        load_dotenv(override=True)
        logger.info(".env file reloaded for API keys.")
        # Update the global model list by re-reading models.txt
        AVAILABLE_MODELS = _load_available_models()
        logger.info("Available models updated from models.txt.")
        return True
    except Exception as e:
        logger.error(f"Error during configuration refresh: {e}", exc_info=True)
        return False


# --- Pydantic Models ---
class ProjectInfo(BaseModel):
    name: str
    description: Optional[str] = "No description available."
    file_count: int

class ChatMessage(BaseModel):
    role: str # e.g., "user", "model"
    content: str

class ChatRequest(BaseModel):
    # Added example values using Field for better FastAPI /docs UI experience
    query: str = Field(..., example="Summarize the key points of the provided documents.")
    provider: Literal["google", "openrouter"] = Field(..., example="openrouter") # Added provider selection
    model_name: str = Field(..., example="mistralai/mistral-7b-instruct") # Added model name selection
    # history: Optional[List[ChatMessage]] = None # Keep commented for now

class ChatResponse(BaseModel):
    response: str
    sources_consulted: List[str] # List of filenames used in context
    model_used: str # Added to confirm which model responded
    input_tokens: Optional[int] = None # Added token counts
    output_tokens: Optional[int] = None # Added token counts

class DocumentContent(BaseModel):
    filename: str
    content: str
    error: Optional[str] = None # To report if parsing failed

class ModelInfo(BaseModel):
    provider: str
    models: List[str]

class AvailableModelsResponse(BaseModel):
    providers: List[ModelInfo]

class ModelDetails(BaseModel):
    context_window: Optional[int] = None
    input_token_limit: Optional[int] = None # Specific to Google
    output_token_limit: Optional[int] = None # Specific to Google
    input_cost_per_million_tokens: Optional[float] = None
    output_cost_per_million_tokens: Optional[float] = None
    notes: Optional[str] = None # For additional info like pricing source

class ModelDetailsResponse(BaseModel):
    provider: str
    model_name: str
    details: ModelDetails


# --- File Parsing Utilities ---
def parse_pdf(file_path: Path) -> DocumentContent:
    """Parses text content from a PDF file using pdfplumber."""
    logger.info(f"Parsing PDF: {file_path.name}")
    full_text = ""
    error_msg = None
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += f"\n--- Page {i+1} ---\n{page_text}"
                else:
                    logger.warning(f"No text extracted from page {i+1} of {file_path.name}")
        if not full_text:
             logger.warning(f"No text extracted from the entire PDF: {file_path.name}")
    except Exception as e:
        logger.error(f"Error parsing PDF {file_path.name}: {e}\n{traceback.format_exc()}")
        error_msg = f"Failed to parse PDF: {e}"
    return DocumentContent(filename=file_path.name, content=full_text.strip(), error=error_msg)

def parse_docx(file_path: Path) -> DocumentContent:
    """Parses text content from a DOCX file using python-docx."""
    logger.info(f"Parsing DOCX: {file_path.name}")
    full_text = ""
    error_msg = None
    try:
        doc = DocxDocument(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text])
    except Exception as e:
        logger.error(f"Error parsing DOCX {file_path.name}: {e}\n{traceback.format_exc()}")
        error_msg = f"Failed to parse DOCX: {e}"
    return DocumentContent(filename=file_path.name, content=full_text.strip(), error=error_msg)

def parse_markdown(file_path: Path) -> DocumentContent:
    """Parses text content from a Markdown file."""
    logger.info(f"Parsing Markdown: {file_path.name}")
    plain_text = ""
    error_msg = None
    try:
        md_content = file_path.read_text(encoding="utf-8")
        import re
        from bs4 import BeautifulSoup # Requires: pip install beautifulsoup4
        html = md_parser.markdown(md_content)
        html = re.sub(r'<pre>(.*?)</pre>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<code>(.*?)</code>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
        soup = BeautifulSoup(html, "html.parser")
        plain_text = soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.error(f"Error parsing Markdown {file_path.name}: {e}\n{traceback.format_exc()}")
        error_msg = f"Failed to parse Markdown: {e}"
    return DocumentContent(filename=file_path.name, content=plain_text.strip(), error=error_msg)

def parse_csv(file_path: Path) -> DocumentContent:
    """Parses content from a CSV file using pandas, converting to string."""
    logger.info(f"Parsing CSV: {file_path.name}")
    content_str = ""
    error_msg = None
    try:
        try:
            df = pd.read_csv(file_path, sep=',')
            content_str = df.to_string(index=False, na_rep='NULL')
        except pd.errors.ParserError as pe:
             logger.error(f"Pandas ParserError for CSV {file_path.name}: {pe}")
             error_msg = f"CSV Parsing Error: {pe}"
        except Exception as e:
            raise e
    except Exception as e:
        logger.error(f"Error processing CSV {file_path.name}: {e}\n{traceback.format_exc()}")
        error_msg = f"Failed to process CSV: {e}"

    if error_msg and not content_str:
         return DocumentContent(filename=file_path.name, content="", error=error_msg)
    else:
        return DocumentContent(filename=file_path.name, content=content_str.strip(), error=error_msg)


# --- Project Loading Logic ---
def get_project_path(project_name: str) -> Optional[Path]:
    """Gets the validated path for a project."""
    project_path = (PROJECTS_DIR / project_name).resolve()
    if PROJECTS_DIR in project_path.parents and project_path.is_dir():
        return project_path
    logger.warning(f"Invalid or non-existent project requested: {project_name}")
    return None

def load_project_data(project_path: Path) -> Dict[str, List[DocumentContent] | str]:
    """Loads system prompt and parses all documents listed in filelist.csv."""
    project_name = project_path.name
    logger.info(f"Loading data for project: {project_name}")
    data = {"system_prompt": "", "documents": []}

    prompt_file = project_path / "system_prompt.txt"
    if prompt_file.is_file():
        try:
            data["system_prompt"] = prompt_file.read_text(encoding="utf-8").strip()
            logger.info(f"Loaded system prompt for {project_name}.")
        except Exception as e:
            logger.error(f"Error reading system prompt for {project_name}: {e}")
            data["system_prompt"] = f"Error: Could not load system prompt. {e}"
    else:
        logger.warning(f"system_prompt.txt not found for project {project_name}.")
        data["system_prompt"] = "Default System Prompt: You are a helpful assistant."

    filelist_csv = project_path / "filelist.csv"
    if not filelist_csv.is_file():
        logger.error(f"filelist.csv not found for project {project_name}. Cannot load documents.")
        return data

    try:
        try:
             df = pd.read_csv(filelist_csv, usecols=["file_name"], skipinitialspace=True) # Changed "file name" to "file_name"
        except ValueError as ve:
             logger.error(f"Column 'file_name' not found in {filelist_csv}: {ve}") # Changed "file name" to "file_name"
             raise FileNotFoundError(f"'file_name' column missing in {filelist_csv}") from ve # Changed "file name" to "file_name"
        except pd.errors.EmptyDataError:
             logger.warning(f"{filelist_csv} is empty. No documents to load.")
             return data
        except Exception as e:
             logger.error(f"Error reading {filelist_csv}: {e}")
             raise

        parsed_docs = []
        filenames_to_parse = df["file_name"].dropna().unique().tolist() # Changed "file name" to "file_name"
        logger.info(f"Found {len(filenames_to_parse)} unique filenames to parse in {filelist_csv.name}.")

        for filename in filenames_to_parse:
            logger.debug(f"Processing filename from CSV: '{filename}'") # Added log
            file_path = (project_path / filename).resolve()
            logger.debug(f"Resolved path: {file_path}") # Added log

            if project_path not in file_path.parents:
                logger.warning(f"Skipping file outside project directory: {filename}")
                parsed_docs.append(DocumentContent(filename=filename, content="", error="Access Denied: File is outside project scope."))
                continue

            if not file_path.is_file():
                logger.warning(f"File listed in CSV not found: {filename} at resolved path {file_path}") # Enhanced log
                parsed_docs.append(DocumentContent(filename=filename, content="", error="File not found."))
                continue

            suffix = file_path.suffix.lower()
            logger.debug(f"File suffix: '{suffix}' for {filename}") # Added log
            doc_content = None
            if suffix == ".pdf":
                logger.debug(f"Calling parse_pdf for {filename}") # Added log
                doc_content = parse_pdf(file_path)
            elif suffix == ".docx":
                logger.debug(f"Calling parse_docx for {filename}") # Added log
                doc_content = parse_docx(file_path)
            elif suffix == ".md":
                logger.debug(f"Calling parse_markdown for {filename}") # Added log
                doc_content = parse_markdown(file_path)
            elif suffix == ".csv":
                 if file_path.name != filelist_csv.name:
                     logger.debug(f"Calling parse_csv for {filename}") # Added log
                     doc_content = parse_csv(file_path)
                 else:
                      logger.info(f"Skipping parsing of the filelist CSV itself: {filename}") # Kept info level
                      doc_content = DocumentContent(filename=filename, content="[Metadata File: List of project documents]", error=None)
            else:
                logger.warning(f"Unsupported file type listed: {filename} ({suffix})")
                doc_content = DocumentContent(filename=filename, content="", error=f"Unsupported file type: {suffix}")

            if doc_content:
                logger.debug(f"Appending parsed content for {filename}. Error: {doc_content.error}") # Added log
                parsed_docs.append(doc_content)
            else:
                # This case should ideally not happen if all branches create a DocumentContent object
                logger.warning(f"No DocumentContent object created for {filename} after parsing attempt.") # Added log

        data["documents"] = parsed_docs

    except FileNotFoundError as fnf:
         logger.error(f"Project loading failed for {project_name}: {fnf}")
    except Exception as e:
        logger.error(f"Unexpected error loading project {project_name}: {e}\n{traceback.format_exc()}")

    return data

# --- Token Counting Utility ---
# Cache encodings to avoid re-fetching them repeatedly
_encodings_cache = {}

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """Estimates token count for text using tiktoken."""
    if not text:
        return 0
    encoding_name = "cl100k_base" # Default encoding
    try:
        # Attempt to get model-specific encoding
        if model_name not in _encodings_cache:
            try:
                _encodings_cache[model_name] = tiktoken.encoding_for_model(model_name)
                logger.debug(f"Cached tiktoken encoding for model: {model_name}")
            except KeyError:
                # Fallback to default if model-specific encoding not found
                logger.warning(f"Tiktoken encoding not found for model '{model_name}'. Falling back to '{encoding_name}'.")
                if encoding_name not in _encodings_cache:
                     _encodings_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
                _encodings_cache[model_name] = _encodings_cache[encoding_name] # Cache the fallback for this model

        encoding = _encodings_cache[model_name]
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens for model {model_name}: {e}. Text length: {len(text)}", exc_info=True)
        # Fallback: rough estimate (e.g., chars / 4) if tiktoken fails unexpectedly
        return len(text) // 4


# --- FastAPI App Instance ---
app = FastAPI(
    title="RAG Chat PoC Backend",
    description="Proof-of-Concept API for interacting with LLMs using project-specific documents as context.",
    version="0.2.0", # Incremented version
)
logger.info("FastAPI app instance created.")


# --- API Endpoints ---

@app.get("/")
async def read_root():
    """ Basic endpoint to check if the server is running. """
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Welcome to the RAG Chat PoC Backend!"}

@app.get("/ping")
async def ping():
    """ Simple health check endpoint. """
    logger.info("Ping endpoint '/ping' accessed.")
    return {"status": "ok", "message": "pong"}

@app.get("/models", response_model=AvailableModelsResponse)
async def get_available_models():
    """ Returns a list of available models grouped by provider. """
    logger.info("'/models' endpoint accessed.")
    providers_list = [
        ModelInfo(provider=provider, models=models)
        for provider, models in AVAILABLE_MODELS.items()
    ]
    return AvailableModelsResponse(providers=providers_list)


@app.post("/refresh")
async def refresh_models():
    """
    Reloads configuration from the .env file, primarily updating the list of available models.
    """
    logger.info("'/refresh' endpoint accessed.")
    success = refresh_configuration()
    if success:
        return {"message": "Configuration refreshed successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to refresh configuration. Check backend logs.")


@app.get("/projects", response_model=List[ProjectInfo])
async def list_projects():
    """ Lists available projects found in the projects directory. """
    logger.info("'/projects' endpoint accessed.")
    projects = []
    if not PROJECTS_DIR.is_dir():
        logger.error(f"Projects directory {PROJECTS_DIR} does not exist.")
        return []

    try:
        for item in PROJECTS_DIR.iterdir():
            # Skip directories starting with an underscore
            if item.is_dir() and not item.name.startswith('_'):
                file_count = 0
                desc = "Default project description."
                filelist_csv = item / "filelist.csv"
                prompt_file = item / "system_prompt.txt"

                if filelist_csv.is_file():
                    try:
                        # Ensure we use the correct column name 'file_name' as updated previously
                        df = pd.read_csv(filelist_csv, usecols=["file_name"], skipinitialspace=True)
                        file_count = len(df["file_name"].dropna())
                        logger.debug(f"Counted {file_count} files for project '{item.name}' from {filelist_csv.name}")
                    except ValueError as ve:
                        # Log specific error if 'file_name' column is missing
                        logger.warning(f"Could not count files for project '{item.name}': Column 'file_name' not found in {filelist_csv.name}. Error: {ve}")
                        file_count = 0 # Default to 0 on error
                    except pd.errors.EmptyDataError:
                        logger.warning(f"Could not count files for project '{item.name}': {filelist_csv.name} is empty.")
                        file_count = 0 # Default to 0 if empty
                    except Exception as e:
                        # Log other potential errors during counting
                        logger.error(f"Error counting files for project '{item.name}' from {filelist_csv.name}: {e}", exc_info=True)
                        file_count = 0 # Default to 0 on error
                else:
                    logger.warning(f"filelist.csv not found for project '{item.name}', cannot count files.")
                    file_count = 0

                # Description logic remains unchanged (currently just default)
                if prompt_file.is_file():
                     try: pass # Keep default desc
                     except Exception: pass # Keep this pass for description part

                projects.append(ProjectInfo(name=item.name, description=desc, file_count=file_count))
        logger.info(f"Found {len(projects)} projects.")
        return projects
    except Exception as e:
        logger.error(f"Error listing projects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing projects: {e}")


@app.post("/chat/{project_name}", response_model=ChatResponse)
async def chat_with_project(
    request: ChatRequest,
    project_name: str = FastAPIPath(..., title="The name of the project to use", min_length=1)
):
    """
    Handles a chat query for a specific project, using the selected provider and model.
    """
    logger.info(f"'/chat/{project_name}' endpoint accessed. Provider: {request.provider}, Model: {request.model_name}, Query: '{request.query[:50]}...'")

    # 1. Validate Provider and Model
    if request.provider not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid provider specified: {request.provider}. Available: {list(AVAILABLE_MODELS.keys())}")
    # Ensure the model name provided exactly matches one in our list for that provider
    if request.model_name not in AVAILABLE_MODELS[request.provider]:
         raise HTTPException(status_code=400, detail=f"Invalid model specified for provider {request.provider}: {request.model_name}. Available: {AVAILABLE_MODELS[request.provider]}")

    # 2. Check API Key for selected provider
    if request.provider == "google" and not google_configured:
         raise HTTPException(status_code=400, detail="Google provider selected, but GOOGLE_API_KEY is not configured correctly in the backend.")
    if request.provider == "openrouter" and not openrouter_configured:
         raise HTTPException(status_code=400, detail="OpenRouter provider selected, but OPENROUTER_API_KEY is not configured correctly in the backend.")

    # 3. Validate and get project path
    project_path = get_project_path(project_name)
    if not project_path:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found or invalid.")

    # 4. Load project data (system prompt + parsed documents)
    try:
        project_data = load_project_data(project_path)
    except Exception as e:
        logger.error(f"Failed to load data for project {project_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load project data: {e}")

    system_prompt = project_data.get("system_prompt", "Error: System prompt missing.")
    documents: List[DocumentContent] = project_data.get("documents", [])
    source_filenames = [doc.filename for doc in documents]

    # 5. Construct the full context (same logic as before)
    context_parts = [system_prompt]
    context_parts.append("\n<DOCUMENT_CONTEXT>")
    if not documents:
         context_parts.append("\n[No documents loaded for this project.]")
    else:
        logger.info(f"Constructing context with {len(documents)} documents for project {project_name}.")
        # Simple length check - NEEDS PROPER TOKENIZATION AND STRATEGY
        # Calculate rough character limit per doc based on a total context goal (e.g., 1M chars)
        total_char_limit = 1_000_000 # Target total characters for context (adjust as needed)
        base_prompt_len = len(system_prompt) + len(request.query) + 500 # Estimate overhead
        remaining_chars = total_char_limit - base_prompt_len
        chars_per_doc = remaining_chars // len(documents) if len(documents) > 0 else remaining_chars
        if chars_per_doc < 100: # Ensure a minimum reasonable limit
             chars_per_doc = 100
             logger.warning(f"Calculated low character limit per doc ({chars_per_doc}), potentially too many docs or too long base prompt.")

        for i, doc in enumerate(documents):
            context_parts.append(f"\n\n--- START DOCUMENT {i+1}: {doc.filename} ---")
            if doc.error:
                context_parts.append(f"\n[Parsing Error: {doc.error}]")
            elif doc.content:
                if len(doc.content) > chars_per_doc:
                     truncated_content = doc.content[:chars_per_doc]
                     context_parts.append(f"\n{truncated_content}")
                     context_parts.append(f"\n[... Content truncated due to length ...]")
                     logger.warning(f"Truncated content for {doc.filename} to {chars_per_doc} chars.")
                else:
                     context_parts.append(f"\n{doc.content}")
            else:
                context_parts.append("\n[Document loaded but contains no text content]")
            context_parts.append(f"\n--- END DOCUMENT {i+1}: {doc.filename} ---")
    context_parts.append("\n</DOCUMENT_CONTEXT>\n")
    context_parts.append(f"\nUser Query: {request.query}")
    full_prompt = "".join(context_parts)
    input_token_count = count_tokens(full_prompt, request.model_name)
    logger.info(f"Approximate prompt length: {len(full_prompt)} characters. Estimated input tokens: {input_token_count}")

    # 6. Send to the selected LLM Provider/Model
    llm_response_text = ""
    output_token_count = 0
    try:
        if request.provider == "google":
            logger.info(f"Sending prompt to Google model: {request.model_name}...")
            # Use the exact model name from the request
            # The genai library expects names like 'gemini-1.5-pro-latest' or 'models/gemini-1.5-pro-latest'
            # Let's assume the names in GOOGLE_MODELS are directly usable.
            model_instance = genai.GenerativeModel(request.model_name)
            response = await model_instance.generate_content_async(full_prompt)
            logger.info("Received response from Google LLM.")

            # Google-specific response handling
            if response and hasattr(response, 'text'):
                llm_response_text = response.text
            elif response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                llm_response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            else:
                # Handle blocked prompts or other issues for Google
                block_reason = getattr(getattr(response, 'prompt_feedback', None), 'block_reason', None)
                finish_reason = getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None
                safety_ratings = getattr(response.candidates[0], 'safety_ratings', None) if response.candidates else None
                logger.warning(f"Google LLM response issue. BlockReason: {block_reason}, FinishReason: {finish_reason}, Safety: {safety_ratings}, Response: {response}")
                if block_reason:
                    raise HTTPException(status_code=400, detail=f"Request blocked by Google safety settings: {block_reason}")
                elif finish_reason != 'STOP':
                     safety_msg = f" Safety Ratings: {safety_ratings}" if safety_ratings else ""
                     raise HTTPException(status_code=500, detail=f"Google LLM generation failed: {finish_reason}.{safety_msg}")
                else:
                     raise HTTPException(status_code=500, detail="Failed to interpret Google LLM response.")
            # Estimate output tokens after getting response
            output_token_count = count_tokens(llm_response_text, request.model_name)

        elif request.provider == "openrouter":
            logger.info(f"Sending prompt to OpenRouter model: {request.model_name}...")
            openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501", # Adjust if needed
                "X-Title": "RAG Chat PoC",
            }
            # Use the full prompt as the user message content
            payload = {
                "model": request.model_name, # Use the exact model identifier from the request
                "messages": [
                    {"role": "user", "content": full_prompt}
                ],
                # "temperature": 0.7, # Optional parameters
                # "max_tokens": 2048,
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(openrouter_url, headers=headers, json=payload)

            logger.info(f"Received response from OpenRouter (Status: {response.status_code}).")
            response.raise_for_status() # Raise exception for 4xx/5xx errors

            response_data = response.json()

            # Extract response text (OpenAI format)
            if response_data and "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    llm_response_text = choice["message"]["content"]
                elif "text" in choice: # Fallback for older formats?
                     llm_response_text = choice["text"]

            if not llm_response_text:
                 logger.warning(f"OpenRouter response received but no text found. Response: {response_data}")
                 raise HTTPException(status_code=500, detail="Failed to interpret OpenRouter LLM response.")
            # Estimate output tokens after getting response
            output_token_count = count_tokens(llm_response_text, request.model_name)

        # If we got here, we should have llm_response_text
        logger.info(f"Estimated output tokens: {output_token_count}")
        return ChatResponse(
            response=llm_response_text,
            sources_consulted=source_filenames,
            model_used=f"{request.provider}/{request.model_name}", # Include provider in response
            input_tokens=input_token_count,
            output_tokens=output_token_count
        )

    except httpx.HTTPStatusError as hse:
         error_body = hse.response.text
         logger.error(f"HTTP error calling OpenRouter ({hse.response.status_code}): {error_body}", exc_info=True)
         detail = f"OpenRouter API error ({hse.response.status_code})"
         try:
              error_json = hse.response.json()
              if "error" in error_json and "message" in error_json["error"]:
                   detail += f": {error_json['error']['message']}"
         except Exception: pass
         raise HTTPException(status_code=hse.response.status_code, detail=detail)

    except Exception as e:
        logger.error(f"Error during LLM interaction ({request.provider}/{request.model_name}): {e}", exc_info=True)
        if isinstance(e, HTTPException):
             raise e
        # Add specific exception handling for genai if needed
        # e.g., except google.api_core.exceptions.PermissionDenied: ...
        raise HTTPException(status_code=500, detail=f"An error occurred during LLM interaction: {e}")


# --- Model Details Endpoint ---

# Cache OpenRouter model data for 1 hour
# openrouter_model_cache = TTLCache(maxsize=1, ttl=3600) # Removed cache due to async issues

# @cached(openrouter_model_cache) # Removed cache decorator - caused "cannot reuse already awaited coroutine" error
async def get_openrouter_models_data() -> Optional[Dict]:
    """Fetches the full model list from OpenRouter. (Caching removed due to async issues)."""
    logger.info("Fetching full model list from OpenRouter API...")
    openrouter_models_url = "https://openrouter.ai/api/v1/models"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(openrouter_models_url)
            response.raise_for_status()
            logger.info("Successfully fetched OpenRouter model list.")
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"HTTP error fetching OpenRouter models: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Error parsing OpenRouter models response: {e}", exc_info=True)
        return None

# Use :path converter for model_name to allow slashes (e.g., in OpenRouter model IDs like 'google/gemini...').
# Without :path, FastAPI treats slashes in the variable part as path separators, leading to 404s.
@app.get("/model-details/{provider}/{model_name:path}", response_model=ModelDetailsResponse)
async def get_model_details(
    provider: Literal["google", "openrouter"] = FastAPIPath(..., title="The LLM provider"),
    model_name: str = FastAPIPath(..., title="The specific model name including any slashes")
):
    """Gets details like context window and pricing for a specific model."""
    logger.info(f"'/model-details' endpoint accessed for {provider}/{model_name}")

    # Validate provider and model against AVAILABLE_MODELS first
    if provider not in AVAILABLE_MODELS or model_name not in AVAILABLE_MODELS[provider]:
         raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or not configured for provider '{provider}'.")

    details = ModelDetails() # Initialize empty details

    try:
        if provider == "google":
            if not google_configured:
                 raise HTTPException(status_code=400, detail="Google provider selected, but GOOGLE_API_KEY is not configured.")
            try:
                # Google model names might need 'models/' prefix for the API call
                api_model_name = model_name if model_name.startswith("models/") else f"models/{model_name}"
                logger.info(f"Querying Google API for model details: {api_model_name}")
                model_info = genai.get_model(api_model_name)
                details.input_token_limit = getattr(model_info, 'input_token_limit', None)
                details.output_token_limit = getattr(model_info, 'output_token_limit', None)
                # Pricing is generally not available via this API
                details.notes = "Pricing information not available via Google API. Refer to Google Cloud/AI Studio documentation."
                logger.info(f"Google model details retrieved: Input Limit={details.input_token_limit}, Output Limit={details.output_token_limit}")

            except Exception as e:
                logger.error(f"Error fetching Google model details for {model_name}: {e}", exc_info=True)
                details.notes = f"Error fetching Google model details: {e}"


        elif provider == "openrouter":
            if not openrouter_configured:
                 raise HTTPException(status_code=400, detail="OpenRouter provider selected, but OPENROUTER_API_KEY is not configured.")

            all_models_data = await get_openrouter_models_data()
            if not all_models_data or "data" not in all_models_data:
                 raise HTTPException(status_code=503, detail="Could not retrieve model list from OpenRouter.")

            found_model = None
            for model_data in all_models_data["data"]:
                if model_data.get("id") == model_name:
                    found_model = model_data
                    break

            if found_model:
                logger.info(f"Found OpenRouter model details for: {model_name}")
                details.context_window = found_model.get("context_length")
                pricing = found_model.get("pricing", {})
                # OpenRouter provides cost per single token under 'prompt' and 'completion' keys
                prompt_cost_str = pricing.get("prompt") # Use 'prompt' key for input cost
                completion_cost_str = pricing.get("completion") # Use 'completion' key for output cost

                # Convert per-token cost to per-million-token cost
                try:
                    prompt_cost_float = float(prompt_cost_str) if prompt_cost_str is not None else None
                    details.input_cost_per_million_tokens = prompt_cost_float * 1_000_000 if prompt_cost_float is not None else None
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert OpenRouter prompt cost '{prompt_cost_str}' to float for model {model_name}.")
                    details.input_cost_per_million_tokens = None

                try:
                    completion_cost_float = float(completion_cost_str) if completion_cost_str is not None else None
                    details.output_cost_per_million_tokens = completion_cost_float * 1_000_000 if completion_cost_float is not None else None
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert OpenRouter completion cost '{completion_cost_str}' to float for model {model_name}.")
                    details.output_cost_per_million_tokens = None

                details.notes = "Pricing shown is per 1 million tokens (calculated from API data)." # Updated note
            else:
                logger.warning(f"Model '{model_name}' not found in OpenRouter's /models response.")
                details.notes = "Model details not found in OpenRouter's current list."


        return ModelDetailsResponse(
            provider=provider,
            model_name=model_name,
            details=details
        )

    except HTTPException as he:
        raise he # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error fetching model details for {provider}/{model_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error fetching model details: {e}")


# --- Uvicorn runner ---
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting Uvicorn server directly from main.py")
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# /mnt/LAB/rag-chat-poc/backend/main.py

import os
import logging
from fastapi import FastAPI, HTTPException, Path as FastAPIPath # Rename Path to avoid conflict
from dotenv import load_dotenv
import google.generativeai as genai
import httpx # Added for OpenRouter REST API calls
from pydantic import BaseModel, Field # Import Pydantic
from typing import List, Optional, Dict, Literal, Tuple
from pathlib import Path # Use pathlib for path operations
import pdfplumber # PDF parsing
from docx import Document as DocxDocument # DOCX parsing (renamed)
import markdown as md_parser # Markdown parsing
import pandas as pd # CSV/Excel parsing
import traceback # For detailed error logging
# import asyncio # No longer needed for async file ops here
import json # For OpenRouter payload and disk cache
import time # For cache validation
import os # For file modification times
import tiktoken # Added for token counting
# import concurrent.futures # No longer needed for runtime parsing
# from functools import partial # No longer needed for runtime parsing
# from cachetools import cached, TTLCache # Not using cachetools for doc cache
# Import parsing functions using absolute import from backend perspective
# These are still needed by the processing script, but not directly by main.py anymore
# from parsing_utils import parse_pdf, parse_docx, parse_markdown, parse_csv, DocumentContent
import subprocess # For running the processing script
from fastapi.responses import JSONResponse # For script execution response

# --- Configuration & Setup ---
# Set logging level to DEBUG to capture detailed info for pricing issue
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)
# Set pdfplumber logger to INFO to reduce verbosity
logging.getLogger("pdfplumber").setLevel(logging.INFO)
# Set pdfminer logger to INFO as well
logging.getLogger("pdfminer").setLevel(logging.INFO)
load_dotenv()
logger.info(".env file loaded.")

# --- Project Directory Setup ---
PROJECT_ROOT = Path(__file__).parent.parent # Define project root relative to this file
PROJECTS_DIR = PROJECT_ROOT / "projects"
logger.info(f"Projects directory set to: {PROJECTS_DIR}")
if not PROJECTS_DIR.is_dir():
    logger.error(f"Projects directory not found at {PROJECTS_DIR}")
    raise RuntimeError(f"Projects directory not found: {PROJECTS_DIR}")

# --- Cache Configuration ---
BACKEND_DIR = Path(__file__).parent
CACHE_DIR = BACKEND_DIR / ".cache" / "parsed_docs"
CACHE_VERSION = 1
IN_MEMORY_DOC_CACHE = {} # Simple dictionary for in-memory cache

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
    total_document_tokens: Optional[int] = None # Add field for pre-calculated token sum

class ChatMessage(BaseModel):
    role: str # e.g., "user", "model"
    content: str

class ChatRequest(BaseModel):
    # Added example values using Field for better FastAPI /docs UI experience
    query: str = Field(..., example="Summarize the key points of the provided documents.")
    provider: Literal["google", "openrouter"] = Field(..., example="openrouter") # Added provider selection
    model_name: str = Field(..., example="mistralai/mistral-7b-instruct") # Added model name selection
    max_context_tokens: Optional[int] = Field(None, example=900000, description="Maximum tokens allowed for the context (documents + prompt). If None, uses a default.") # Added max tokens
    # history: Optional[List[ChatMessage]] = None # Keep commented for now

class ChatResponse(BaseModel):
    response: str
    sources_consulted: List[str] # List of filenames used in context
    skipped_sources: List[str] = Field([], description="List of document filenames skipped because adding them would exceed the token limit.") # Added skipped sources
    model_used: str # Added to confirm which model responded
    input_tokens: Optional[int] = None # Added token counts
    output_tokens: Optional[int] = None # Added token counts
    # runtime_parsing_occurred: bool = Field(False, description="Flag indicating whether documents were parsed at runtime rather than loaded from cache.") # Removed

class DocumentContent(BaseModel):
    filename: str
    content: Optional[str] # Content can be None if parsing fails or file is skipped
    token_count: Optional[int] = None # Added pre-calculated token count
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
# Parsing functions moved to parsing_utils.py and are no longer called directly by main.py

# --- Cache Helper Functions ---

def get_cache_file_path(project_name: str) -> Path:
    """Constructs the path to the project's disk cache file."""
    return CACHE_DIR / f"{project_name}.json"

def get_modification_time(file_path: Path) -> Optional[float]:
    """Gets the last modification time of a file."""
    try:
        # Use os.path.getmtime for potentially better compatibility across OS
        return os.path.getmtime(file_path)
    except FileNotFoundError:
        logger.warning(f"File not found when checking modification time: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error getting modification time for {file_path}: {e}", exc_info=True)
        return None

def load_from_disk_cache(project_path: Path, cache_file_path: Path) -> Optional[List[DocumentContent]]:
    """Loads and validates data from the disk cache file."""
    project_name = project_path.name
    if not cache_file_path.is_file():
        logger.info(f"Disk cache file not found for project '{project_name}': {cache_file_path}")
        return None

    logger.info(f"Attempting to load from disk cache: {cache_file_path}")
    try:
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # Basic validation
        if cache_data.get("cache_version") != CACHE_VERSION:
            logger.warning(f"Cache version mismatch for {project_name}. Expected {CACHE_VERSION}, found {cache_data.get('cache_version')}. Invalidating cache.")
            return None
        if cache_data.get("project_name") != project_name:
            logger.warning(f"Project name mismatch in cache file {cache_file_path}. Expected '{project_name}', found '{cache_data.get('project_name')}'. Invalidating cache.")
            return None

        cached_docs_dict = cache_data.get("documents", {})
        validated_docs = []
        is_cache_valid = True

        # Validate modification times
        for filename, doc_cache_info in cached_docs_dict.items():
            source_file_path = (project_path / filename).resolve()
            cached_mod_time = doc_cache_info.get("source_mod_time")

            # Check if file is still within project dir (important if filelist.csv was manually edited)
            if project_path.resolve() not in source_file_path.resolve().parents:
                 logger.warning(f"Cached file '{filename}' is outside project directory '{project_path}'. Invalidating cache.")
                 is_cache_valid = False
                 break # Invalidate whole cache if one file is bad

            current_mod_time = get_modification_time(source_file_path)

            if current_mod_time is None: # Source file deleted or inaccessible
                logger.warning(f"Source file '{filename}' for cached entry not found or inaccessible. Invalidating cache.")
                is_cache_valid = False
                break
            if cached_mod_time is None or current_mod_time > cached_mod_time:
                logger.info(f"Source file '{filename}' has been modified since cache generation (Current: {current_mod_time}, Cached: {cached_mod_time}). Invalidating cache.")
                is_cache_valid = False
                break

            # If valid so far, reconstruct DocumentContent, including token count if available
            validated_docs.append(DocumentContent(
                filename=filename,
                content=doc_cache_info.get("parsed_content"),
                token_count=doc_cache_info.get("token_count"), # Load token count from cache
                error=doc_cache_info.get("parsing_error")
            ))

        if is_cache_valid:
            logger.info(f"Disk cache for project '{project_name}' is valid.")
            return validated_docs
        else:
            logger.info(f"Disk cache for project '{project_name}' is invalid or stale.")
            return None

    except json.JSONDecodeError as jde:
        logger.error(f"Error decoding JSON from cache file {cache_file_path}: {jde}. Invalidating cache.")
        return None
    except Exception as e:
        logger.error(f"Error loading or validating disk cache {cache_file_path}: {e}", exc_info=True)
        return None

def save_to_disk_cache(project_name: str, documents_data: Dict[str, dict]):
    """Saves parsed document data (including mod times and token counts) to the disk cache."""
    cache_file_path = get_cache_file_path(project_name)
    logger.info(f"Saving freshly parsed data to disk cache: {cache_file_path}")
    try:
        cache_data = {
            "cache_version": CACHE_VERSION,
            "project_name": project_name,
            "generation_timestamp": time.time(),
            "documents": documents_data # Expects dict {filename: {"parsed_content":..., "source_mod_time":..., "token_count":..., "parsing_error":...}}
        }
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Successfully saved cache for project '{project_name}'.")
    except Exception as e:
        logger.error(f"Error saving disk cache for project '{project_name}': {e}", exc_info=True)

def save_document_to_cache(project_name: str, filename: str, doc_data: dict):
    """Saves a single document to the cache, creating or updating the cache file."""
    cache_file_path = get_cache_file_path(project_name)
    
    # Create cache directory if it doesn't exist
    cache_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing cache or create new
    if cache_file_path.exists():
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
                # Validate cache format
                if cache_data.get("cache_version") != CACHE_VERSION or cache_data.get("project_name") != project_name:
                    logger.warning(f"Existing cache file for {project_name} has incorrect version or project name. Creating new cache.")
                    cache_data = {
                        "cache_version": CACHE_VERSION,
                        "project_name": project_name,
                        "generation_timestamp": time.time(),
                        "documents": {}
                    }
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Existing cache file for {project_name} is corrupted: {e}. Creating new.")
            cache_data = {
                "cache_version": CACHE_VERSION,
                "project_name": project_name,
                "generation_timestamp": time.time(),
                "documents": {}
            }
    else:
        # Create new cache
        cache_data = {
            "cache_version": CACHE_VERSION,
            "project_name": project_name,
            "generation_timestamp": time.time(),
            "documents": {}
        }
    
    # Update the document in the cache
    cache_data["documents"][filename] = doc_data
    cache_data["generation_timestamp"] = time.time()  # Update timestamp
    
    # Write updated cache back to file
    try:
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Updated cache for '{project_name}' with document: {filename}")
    except Exception as e:
        logger.error(f"Error updating cache file for {project_name} with document {filename}: {e}")


# --- Project Loading Logic ---
def get_project_path(project_name: str) -> Optional[Path]:
    """Gets the validated path for a project."""
    project_path = (PROJECTS_DIR / project_name).resolve()
    # Ensure project_path is actually inside PROJECTS_DIR and is a directory
    if project_path.is_dir() and PROJECTS_DIR.resolve() in project_path.resolve().parents:
        return project_path
    logger.warning(f"Invalid or non-existent project requested: {project_name}")
    return None

def load_project_data(project_path: Path) -> Dict[str, List[DocumentContent] | str]:
    """
    Loads system prompt and document content for a project, utilizing in-memory
    and disk caches. Parses fresh if cache is missing or stale, and updates disk cache.
    """
    project_name = project_path.name
    logger.info(f"Loading data for project: {project_name}")

    # 1. Check In-Memory Cache
    if project_name in IN_MEMORY_DOC_CACHE:
        logger.info(f"Using in-memory cache for project '{project_name}'.")
        return IN_MEMORY_DOC_CACHE[project_name]

    # --- Load System Prompt (Always load fresh for simplicity) ---
    system_prompt = "Default System Prompt: You are a helpful assistant."
    prompt_file = project_path / "system_prompt.txt"
    if prompt_file.is_file():
        try:
            system_prompt = prompt_file.read_text(encoding="utf-8").strip()
            logger.info(f"Loaded system prompt for {project_name}.")
        except Exception as e:
            logger.error(f"Error reading system prompt for {project_name}: {e}")
            system_prompt = f"Error: Could not load system prompt. {e}"
    else:
        logger.warning(f"system_prompt.txt not found for project {project_name}.")

    # 2. Check Disk Cache
    cache_file_path = get_cache_file_path(project_name)
    validated_docs_from_cache = load_from_disk_cache(project_path, cache_file_path)

    if validated_docs_from_cache is not None:
        # Disk cache is valid, store in memory and return
        project_data = {"system_prompt": system_prompt, "documents": validated_docs_from_cache}
        IN_MEMORY_DOC_CACHE[project_name] = project_data
        # Disk cache is valid, store in memory and return
        project_data = {"system_prompt": system_prompt, "documents": validated_docs_from_cache}
        IN_MEMORY_DOC_CACHE[project_name] = project_data
        return project_data

    # 3. Cache Miss or Invalid - Return empty documents
    logger.warning(f"No valid cache found for project '{project_name}'. Returning empty document list. Run processing script to generate cache.")
    project_data = {"system_prompt": system_prompt, "documents": []}
    # Optionally cache the empty result to avoid repeated disk checks? For now, let's not.
    # IN_MEMORY_DOC_CACHE[project_name] = project_data
    return project_data

# --- Token Counting Utility ---
# Cache encodings to avoid re-fetching them repeatedly
_encodings_cache = {}

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

# Removed /check-parsing endpoint

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
                total_tokens = None # Initialize total tokens
                desc = "Default project description."
                filelist_csv = item / "filelist.csv"
                prompt_file = item / "system_prompt.txt"

                if filelist_csv.is_file():
                    try:
                        # Read filelist, attempting to get both file_name and token_count
                        df = pd.read_csv(filelist_csv, skipinitialspace=True)
                        if "file_name" in df.columns:
                            file_count = len(df["file_name"].dropna())
                            logger.debug(f"Counted {file_count} files for project '{item.name}' from {filelist_csv.name}")
                        else:
                             logger.warning(f"Column 'file_name' not found in {filelist_csv.name} for project '{item.name}'. File count set to 0.")
                             file_count = 0

                        # Calculate total tokens if column exists and has valid numbers
                        if "token_count" in df.columns:
                             # Convert to numeric, coercing errors to NaN, then sum valid numbers
                             valid_tokens = pd.to_numeric(df["token_count"], errors='coerce').dropna()
                             if not valid_tokens.empty:
                                 total_tokens = int(valid_tokens.sum())
                                 logger.debug(f"Calculated total document tokens for project '{item.name}': {total_tokens}")
                             else:
                                 logger.warning(f"'token_count' column found but contains no valid numbers for project '{item.name}'.")
                        else:
                             logger.warning(f"'token_count' column not found in {filelist_csv.name} for project '{item.name}'. Cannot calculate total document tokens.")

                    except ValueError as ve:
                        # Log specific error if 'file_name' column is missing (though handled above now)
                        logger.warning(f"ValueError reading {filelist_csv.name} for project '{item.name}': {ve}")
                        file_count = 0
                        total_tokens = None
                    except pd.errors.EmptyDataError:
                        logger.warning(f"{filelist_csv.name} is empty for project '{item.name}'.")
                        file_count = 0
                        total_tokens = None
                    except Exception as e:
                        logger.error(f"Error reading or processing {filelist_csv.name} for project '{item.name}': {e}", exc_info=True)
                        file_count = 0
                        total_tokens = None
                else:
                    logger.warning(f"filelist.csv not found for project '{item.name}'. Cannot count files or tokens.")
                    file_count = 0
                    total_tokens = None

                # Description logic remains unchanged
                if prompt_file.is_file():
                     try: pass
                     except Exception: pass

                projects.append(ProjectInfo(
                    name=item.name,
                    description=desc,
                    file_count=file_count,
                    total_document_tokens=total_tokens # Add the calculated sum
                ))
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
    if request.provider == "google" and (not google_configured or google_api_key == "YOUR_GOOGLE_AI_STUDIO_API_KEY_HERE"):
         raise HTTPException(status_code=400, detail="Google provider selected, but GOOGLE_API_KEY is not configured correctly in the backend.")
    if request.provider == "openrouter" and (not openrouter_configured or openrouter_api_key == "YOUR_OPENROUTER_API_KEY_HERE"):
         raise HTTPException(status_code=400, detail="OpenRouter provider selected, but OPENROUTER_API_KEY is not configured correctly in the backend.")

    # 3. Validate and get project path
    project_path = get_project_path(project_name)
    if not project_path:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found or invalid.")

    # 4. Load project data (system prompt + parsed documents from cache ONLY)
    try:
        project_data = load_project_data(project_path)
        # Check if documents were loaded (i.e., cache was valid)
        if not project_data.get("documents"):
            logger.warning(f"No documents loaded for project '{project_name}'. Cache might be missing or invalid. Chatting without document context.")
            # Optionally raise an error or return a specific message if context is crucial
            # raise HTTPException(status_code=404, detail=f"Document cache for project '{project_name}' not found. Please process the project first.")
    except Exception as e:
        logger.error(f"Failed to load data for project {project_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load project data: {e}")

    system_prompt = project_data.get("system_prompt", "Error: System prompt missing.")
    documents: List[DocumentContent] = project_data.get("documents", [])
    # source_filenames = [doc.filename for doc in documents] # We'll build this list based on included docs

    # 5. Construct Context based on Token Limit
    DEFAULT_MAX_CONTEXT_TOKENS = 900_000 # Default limit if not provided
    effective_max_tokens = request.max_context_tokens if request.max_context_tokens and request.max_context_tokens > 0 else DEFAULT_MAX_CONTEXT_TOKENS
    logger.info(f"Using effective max context tokens: {effective_max_tokens}")

    # Calculate base prompt tokens (system prompt + query + separators/structure)
    # Estimate tokens for separators and structure conservatively
    base_prompt_structure = f"{system_prompt}\n<DOCUMENT_CONTEXT>\n</DOCUMENT_CONTEXT>\n\nUser Query: {request.query}"
    base_prompt_tokens = count_tokens(base_prompt_structure, request.model_name)
    # Add estimated tokens per document for separators like "--- START DOC ---", etc.
    tokens_per_doc_overhead = count_tokens("\n\n--- START DOCUMENT X: filename.ext ---\n\n--- END DOCUMENT X: filename.ext ---", request.model_name)

    context_parts = [system_prompt, "\n<DOCUMENT_CONTEXT>"]
    included_sources: List[str] = []
    skipped_sources: List[str] = []
    current_token_usage = base_prompt_tokens
    remaining_docs = list(documents) # Create a mutable list to track remaining docs

    if not documents:
        context_parts.append("\n[No documents loaded for this project.]")
    else:
        logger.info(f"Attempting to construct context with {len(documents)} documents for project {project_name} within {effective_max_tokens} token limit.")
        for i, doc in enumerate(documents):
            # Check if document has valid precalculated token count and *some* content (even if error is flagged)
            if doc.token_count is None or doc.token_count <= 0:
                logger.warning(f"Skipping document {doc.filename} due to missing or invalid precalculated token count ({doc.token_count}).")
                skipped_sources.append(f"{doc.filename} (invalid token count)")
                remaining_docs.pop(0) # Remove from remaining list
                continue
            # Also skip if content is truly None (e.g., file not found during parse)
            if doc.content is None:
                logger.warning(f"Skipping document {doc.filename} because content is None (likely file not found or inaccessible during parse).")
                skipped_sources.append(f"{doc.filename} (no content)")
                remaining_docs.pop(0) # Remove from remaining list
                continue

            # Calculate potential usage if this doc is added
            potential_usage = current_token_usage + doc.token_count + tokens_per_doc_overhead

            # --- DEBUG LOGGING REMOVED ---

            if potential_usage <= effective_max_tokens:
                # Add the document
                context_parts.append(f"\n\n--- START DOCUMENT {i+1}: {doc.filename} ---")
                context_parts.append(f"\n{doc.content}")
                context_parts.append(f"\n--- END DOCUMENT {i+1}: {doc.filename} ---")
                current_token_usage = potential_usage
                included_sources.append(doc.filename)
                remaining_docs.pop(0) # Remove from remaining list
                logger.debug(f"Included document: {doc.filename} (Tokens: {doc.token_count}). Current total tokens: {current_token_usage}")
            else:
                # Document exceeds limit, stop adding documents
                logger.warning(f"Skipping document {doc.filename} (Tokens: {doc.token_count}) as adding it would exceed the limit ({potential_usage} > {effective_max_tokens}).")
                # Add this and all subsequent documents to skipped_sources
                skipped_sources.append(f"{doc.filename} (limit reached)")
                skipped_sources.extend([f"{rem_doc.filename} (limit reached)" for rem_doc in remaining_docs[1:]]) # Add remaining filenames
                break # Stop iterating

    context_parts.append("\n</DOCUMENT_CONTEXT>\n")
    context_parts.append(f"\nUser Query: {request.query}")
    full_prompt = "".join(context_parts)

    # Calculate final input token count based on the actual assembled prompt
    final_input_token_count = count_tokens(full_prompt, request.model_name)
    logger.info(f"Final prompt length: {len(full_prompt)} characters. Final input tokens: {final_input_token_count}")
    if skipped_sources:
         logger.warning(f"Skipped {len(skipped_sources)} sources due to token limit: {skipped_sources}")

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
            sources_consulted=included_sources, # Return list of included sources
            skipped_sources=skipped_sources, # Return list of skipped sources
            model_used=f"{request.provider}/{request.model_name}", # Include provider in response
            input_tokens=final_input_token_count, # Use final calculated input tokens
            output_tokens=output_token_count
            # runtime_parsing_occurred removed
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


# --- Script Execution Endpoints ---

@app.post("/projects/{project_name}/process")
async def process_project(
    project_name: str = FastAPIPath(..., title="The name of the project to process", min_length=1)
):
    """Triggers the document processing script for a specific project."""
    logger.info(f"'/projects/{project_name}/process' endpoint accessed.")

    # Validate project exists (optional but good practice)
    project_path = get_project_path(project_name)
    if not project_path:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found or invalid.")

    script_path = (Path(__file__).parent / "scripts" / "run_process_docs.sh").resolve()
    # Use the absolute path to the project directory
    project_dir_abs_path = project_path.resolve()

    command = ["/bin/sh", str(script_path), str(project_dir_abs_path)]
    logger.info(f"Executing command: {' '.join(map(str, command))}") # Ensure all parts are strings for join

    try:
        # Run synchronously
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600) # 10 min timeout

        logger.info(f"Script execution finished for {project_name}. Return code: {result.returncode}")
        logger.debug(f"Script stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Script stderr:\n{result.stderr}")

        # Clear in-memory cache for this project after processing
        if project_name in IN_MEMORY_DOC_CACHE:
            del IN_MEMORY_DOC_CACHE[project_name]
            logger.info(f"Cleared in-memory cache for project '{project_name}' after processing.")

        return JSONResponse(content={
            "success": result.returncode == 0,
            "message": f"Processing script for '{project_name}' finished.",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        })

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout expired while running processing script for {project_name}.")
        raise HTTPException(status_code=504, detail=f"Processing script for '{project_name}' timed out.")
    except Exception as e:
        logger.error(f"Error executing processing script for {project_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error executing processing script: {e}")


@app.post("/projects/process-all")
async def process_all_projects():
    """Triggers the document processing script for all projects."""
    logger.info("'/projects/process-all' endpoint accessed.")

    script_path = (Path(__file__).parent / "scripts" / "run_process_docs.sh").resolve()
    command = ["/bin/sh", str(script_path), "--all-projects"]
    logger.info(f"Executing command: {' '.join(map(str, command))}") # Ensure all parts are strings for join

    try:
        # Run synchronously - might take a long time! Increase timeout significantly.
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=1800) # 30 min timeout

        logger.info(f"Script execution finished for all projects. Return code: {result.returncode}")
        logger.debug(f"Script stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Script stderr:\n{result.stderr}")

        # Clear entire in-memory cache after processing all
        IN_MEMORY_DOC_CACHE.clear()
        logger.info("Cleared entire in-memory cache after processing all projects.")

        return JSONResponse(content={
            "success": result.returncode == 0,
            "message": "Processing script for all projects finished.",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        })

    except subprocess.TimeoutExpired:
        logger.error("Timeout expired while running processing script for all projects.")
        raise HTTPException(status_code=504, detail="Processing script for all projects timed out.")
    except Exception as e:
        logger.error(f"Error executing processing script for all projects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error executing processing script for all projects: {e}")


# --- Uvicorn runner ---
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting Uvicorn server directly from main.py")
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

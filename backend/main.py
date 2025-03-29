# /mnt/LAB/rag-chat-poc/backend/main.py

import os
import logging
from fastapi import FastAPI, HTTPException, Path as FastAPIPath # Rename Path to avoid conflict
from dotenv import load_dotenv
import google.generativeai as genai
import httpx
from pydantic import BaseModel, Field # Import Pydantic
from typing import List, Optional, Dict
from pathlib import Path # Use pathlib for path operations
import pdfplumber # PDF parsing
from docx import Document as DocxDocument # DOCX parsing (renamed)
import markdown as md_parser # Markdown parsing
import pandas as pd # CSV/Excel parsing
import traceback # For detailed error logging
import asyncio # For potential async file operations later

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()
logger.info(".env file loaded.")

# --- Project Directory Setup ---
# Assuming the script runs from the 'backend' directory
# Go up one level to the project root, then into 'projects'
PROJECTS_DIR = Path(__file__).parent.parent / "projects"
logger.info(f"Projects directory set to: {PROJECTS_DIR}")
if not PROJECTS_DIR.is_dir():
    logger.error(f"Projects directory not found at {PROJECTS_DIR}")
    # Handle error appropriately - maybe raise startup error

# --- API Key Retrieval & Gemini Config ---
google_api_key = os.getenv("GOOGLE_API_KEY")
# openrouter_api_key = os.getenv("OPENROUTER_API_KEY") # Keep for potential future use

if not google_api_key:
    logger.error("GOOGLE_API_KEY not found in .env file. The application requires this key.")
    # Exit or raise a configuration error for clarity in PoC
    raise RuntimeError("GOOGLE_API_KEY is missing in .env")
else:
    logger.info("Google API Key loaded successfully.")
    try:
        genai.configure(api_key=google_api_key)
        logger.info("Google Generative AI client configured.")
        # --- Model Selection ---
        # Confirmed working in previous step, keep using it.
        # If switching to OpenRouter, this client/model name would change.
        LLM_MODEL_NAME = 'models/rag-2.5-pro-exp-03-25'
        # Initialize the model instance once (can be reused)
        llm_model = genai.GenerativeModel(LLM_MODEL_NAME)
        logger.info(f"Using LLM model: {LLM_MODEL_NAME}")

    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI client or model: {e}")
        raise RuntimeError(f"LLM configuration failed: {e}")


# --- Pydantic Models ---
# (Sources [7], [16], [21], [38], [40])

class ProjectInfo(BaseModel):
    name: str
    description: Optional[str] = "No description available."
    file_count: int

class ChatMessage(BaseModel):
    role: str # e.g., "user", "model"
    content: str

class ChatRequest(BaseModel):
    query: str
    # We might add history later: history: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    response: str
    sources_consulted: List[str] # List of filenames used in context

class DocumentContent(BaseModel):
    filename: str
    content: str
    error: Optional[str] = None # To report if parsing failed

# --- File Parsing Utilities ---

# Consider making file operations async if they become a bottleneck,
# but for ~50 docs, sync might be okay for the PoC.
# Example async reading: `async with aiofiles.open(file_path, mode='r') as f: content = await f.read()`
# Requires `pip install aiofiles`

def parse_pdf(file_path: Path) -> DocumentContent:
    """Parses text content from a PDF file using pdfplumber."""
    logger.info(f"Parsing PDF: {file_path.name}")
    full_text = ""
    error_msg = None
    try:
        # pdfplumber is good for text extraction (Sources [8], [26], [27], [33])
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    # Add page breaks or markers for clarity
                    full_text += f"\n--- Page {i+1} ---\n{page_text}"
                else:
                    logger.warning(f"No text extracted from page {i+1} of {file_path.name}")
        if not full_text:
             logger.warning(f"No text extracted from the entire PDF: {file_path.name}")
             # Consider if this should be an error or just return empty content
             # error_msg = "No text could be extracted from this PDF."
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
        # python-docx reads paragraphs (Sources [14], [25], [32], [37], [46])
        doc = DocxDocument(file_path)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text])
        # Note: This basic approach might miss text in tables, headers, footers.
        # Source [46] mentions limitations and potential workarounds if needed later.
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
        # Read the file content (Source [1], [15], [22], [41])
        md_content = file_path.read_text(encoding="utf-8")
        # Convert Markdown to plain text (simpler than HTML for context)
        # A common approach is to convert to HTML then strip tags,
        # but let's try a simpler direct text extraction if possible.
        # The 'markdown' library primarily converts to HTML (Sources [17], [28], [45]).
        # Alternative: Use regex or a dedicated lib for MD -> text if needed.
        # Simple approach: treat it mostly as plain text for now.
        # For PoC, maybe just include the raw markdown? Or basic HTML->Text conversion.
        # Let's use the HTML conversion + basic stripping (Source [42])
        import re
        from bs4 import BeautifulSoup # Requires: pip install beautifulsoup4
        html = md_parser.markdown(md_content)
        # Remove code blocks before parsing text
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
        # Read CSV, convert to a string representation (e.g., markdown table or just string dump)
        # (Sources [4], [11], [18], [19], [23])
        # Handle potential parsing errors within pandas read_csv (Source [3])
        try:
            df = pd.read_csv(file_path, sep=',') # Adjust separator if needed
             # Convert DataFrame to a string format suitable for LLM context
            content_str = df.to_string(index=False, na_rep='NULL') # Simple string dump
            # Alternative: df.to_markdown(index=False) # Nicer formatting if needed
        except pd.errors.ParserError as pe:
             logger.error(f"Pandas ParserError for CSV {file_path.name}: {pe}")
             error_msg = f"CSV Parsing Error: {pe}"
        except Exception as e: # Catch other potential file reading issues
            raise e # Re-raise unexpected errors
    except Exception as e:
        logger.error(f"Error processing CSV {file_path.name}: {e}\n{traceback.format_exc()}")
        error_msg = f"Failed to process CSV: {e}"

    if error_msg and not content_str: # If only error occurred
         return DocumentContent(filename=file_path.name, content="", error=error_msg)
    else: # Return content even if there was a parsing warning/error but some data read
        return DocumentContent(filename=file_path.name, content=content_str.strip(), error=error_msg)

# Add parse_xlsx similarly if needed, using pd.read_excel()

# --- Project Loading Logic ---

def get_project_path(project_name: str) -> Optional[Path]:
    """Gets the validated path for a project."""
    project_path = (PROJECTS_DIR / project_name).resolve()
    # Security check: ensure path is within PROJECTS_DIR
    if PROJECTS_DIR in project_path.parents and project_path.is_dir():
        return project_path
    logger.warning(f"Invalid or non-existent project requested: {project_name}")
    return None

def load_project_data(project_path: Path) -> Dict[str, List[DocumentContent] | str]:
    """Loads system prompt and parses all documents listed in filelist.csv."""
    project_name = project_path.name
    logger.info(f"Loading data for project: {project_name}")
    data = {"system_prompt": "", "documents": []}

    # 1. Load System Prompt
    prompt_file = project_path / "system_prompt.txt"
    if prompt_file.is_file():
        try:
            data["system_prompt"] = prompt_file.read_text(encoding="utf-8").strip()
            logger.info(f"Loaded system prompt for {project_name}.")
        except Exception as e:
            logger.error(f"Error reading system prompt for {project_name}: {e}")
            data["system_prompt"] = f"Error: Could not load system prompt. {e}" # Include error in prompt
    else:
        logger.warning(f"system_prompt.txt not found for project {project_name}.")
        data["system_prompt"] = "Default System Prompt: You are a helpful assistant." # Fallback

    # 2. Load and Parse filelist.csv
    filelist_csv = project_path / "filelist.csv"
    if not filelist_csv.is_file():
        logger.error(f"filelist.csv not found for project {project_name}. Cannot load documents.")
        # Return early or with an error indicator? For PoC, maybe continue without docs.
        return data # Return with just system prompt

    try:
        # Use pandas to read the CSV (Sources [4], [11], [18], [19], [23])
        # Specify the 'file name' column is needed (Source [11] `usecols`)
        # Handle potential read errors (Source [3])
        try:
             df = pd.read_csv(filelist_csv, usecols=["file name"], skipinitialspace=True)
        except ValueError as ve: # Happens if 'file name' column doesn't exist
             logger.error(f"Column 'file name' not found in {filelist_csv}: {ve}")
             raise FileNotFoundError(f"'file name' column missing in {filelist_csv}") from ve
        except pd.errors.EmptyDataError:
             logger.warning(f"{filelist_csv} is empty. No documents to load.")
             return data # No files listed
        except Exception as e:
             logger.error(f"Error reading {filelist_csv}: {e}")
             raise # Re-raise other pandas/read errors

        # 3. Parse each document listed
        parsed_docs = []
        filenames_to_parse = df["file name"].dropna().unique().tolist()
        logger.info(f"Found {len(filenames_to_parse)} unique filenames to parse in {filelist_csv.name}.")

        for filename in filenames_to_parse:
            file_path = (project_path / filename).resolve()

            # Security check: Ensure file is within the project directory
            if project_path not in file_path.parents:
                logger.warning(f"Skipping file outside project directory: {filename}")
                parsed_docs.append(DocumentContent(filename=filename, content="", error="Access Denied: File is outside project scope."))
                continue

            if not file_path.is_file():
                logger.warning(f"File listed in CSV not found: {filename} at {file_path}")
                parsed_docs.append(DocumentContent(filename=filename, content="", error="File not found."))
                continue

            # Determine file type and parse
            suffix = file_path.suffix.lower()
            doc_content = None
            if suffix == ".pdf":
                doc_content = parse_pdf(file_path)
            elif suffix == ".docx":
                doc_content = parse_docx(file_path)
            elif suffix == ".md":
                doc_content = parse_markdown(file_path)
            elif suffix == ".csv":
                 # Avoid parsing filelist.csv itself recursively if listed
                 if file_path.name != filelist_csv.name:
                     doc_content = parse_csv(file_path)
                 else:
                      logger.info(f"Skipping parsing of the filelist CSV itself: {filename}")
                      # Optionally add metadata about it instead of full content
                      doc_content = DocumentContent(filename=filename, content="[Metadata File: List of project documents]", error=None)
            # Add elif for .xlsx using parse_xlsx if needed
            else:
                logger.warning(f"Unsupported file type listed: {filename} ({suffix})")
                doc_content = DocumentContent(filename=filename, content="", error=f"Unsupported file type: {suffix}")

            if doc_content:
                parsed_docs.append(doc_content)

        data["documents"] = parsed_docs

    except FileNotFoundError as fnf: # Catch missing filelist or missing 'file name' column
         logger.error(f"Project loading failed for {project_name}: {fnf}")
         # Maybe add error info to be returned?
    except Exception as e:
        logger.error(f"Unexpected error loading project {project_name}: {e}\n{traceback.format_exc()}")
        # Add error info

    return data


# --- FastAPI App Instance ---
app = FastAPI(
    title="RAG Chat PoC Backend",
    description="Proof-of-Concept API for interacting with Gemini using project-specific documents as context.",
    version="0.1.0",
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

@app.get("/projects", response_model=List[ProjectInfo])
async def list_projects():
    """ Lists available projects found in the projects directory. """
    logger.info("'/projects' endpoint accessed.")
    projects = []
    if not PROJECTS_DIR.is_dir():
        logger.error(f"Projects directory {PROJECTS_DIR} does not exist.")
        return [] # Or raise 500 error

    try:
        # Iterate through items in PROJECTS_DIR (Sources [1], [2], [6], [9], [13], [20])
        for item in PROJECTS_DIR.iterdir():
            if item.is_dir():
                # Basic info: count files listed in filelist.csv if it exists
                file_count = 0
                desc = "Default project description."
                filelist_csv = item / "filelist.csv"
                prompt_file = item / "system_prompt.txt"

                if filelist_csv.is_file():
                    try:
                        df = pd.read_csv(filelist_csv, usecols=["file name"], skipinitialspace=True)
                        file_count = len(df["file name"].dropna())
                    except Exception: # Ignore errors here, just report count 0
                        pass
                if prompt_file.is_file():
                     try:
                          # Maybe read first line as description?
                          # desc = prompt_file.read_text(encoding='utf-8').splitlines()[0][:100] + "..."
                          pass # Keep default desc for now
                     except Exception:
                          pass

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
    Handles a chat query for a specific project, loading all its documents into context.
    """
    logger.info(f"'/chat/{project_name}' endpoint accessed with query: '{request.query[:50]}...'")

    # 1. Validate and get project path
    project_path = get_project_path(project_name)
    if not project_path:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found or invalid.")

    # 2. Load project data (system prompt + parsed documents)
    # This is synchronous for now. Could be made async if parsing takes too long.
    try:
        project_data = load_project_data(project_path)
    except Exception as e:
        logger.error(f"Failed to load data for project {project_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load project data: {e}")

    system_prompt = project_data.get("system_prompt", "Error: System prompt missing.")
    documents: List[DocumentContent] = project_data.get("documents", [])
    source_filenames = [doc.filename for doc in documents] # For the response

    # 3. Construct the full context for the LLM
    # Google recommends structuring context, e.g., with XML tags (Source [10], [35])
    context_parts = [system_prompt]
    context_parts.append("\n<DOCUMENT_CONTEXT>") # Start of document context block

    if not documents:
         context_parts.append("\n[No documents loaded for this project.]")
    else:
        logger.info(f"Constructing context with {len(documents)} documents for project {project_name}.")
        for i, doc in enumerate(documents):
            context_parts.append(f"\n\n--- START DOCUMENT {i+1}: {doc.filename} ---")
            if doc.error:
                context_parts.append(f"\n[Parsing Error: {doc.error}]")
                logger.warning(f"Including parsing error in context for {doc.filename}: {doc.error}")
            elif doc.content:
                # Basic truncation check (very rough estimate, not token-based)
                # A real implementation needs proper token counting.
                MAX_CHARS_PER_DOC = 1_500_000 // len(documents) if len(documents) > 0 else 1_500_000 # Arbitrary split
                if len(doc.content) > MAX_CHARS_PER_DOC:
                     truncated_content = doc.content[:MAX_CHARS_PER_DOC]
                     context_parts.append(f"\n{truncated_content}")
                     context_parts.append(f"\n[... Content truncated due to length ...]")
                     logger.warning(f"Truncated content for {doc.filename} due to estimated size limits.")
                else:
                     context_parts.append(f"\n{doc.content}")
            else:
                context_parts.append("\n[Document loaded but contains no text content]")
            context_parts.append(f"\n--- END DOCUMENT {i+1}: {doc.filename} ---")

    context_parts.append("\n</DOCUMENT_CONTEXT>\n") # End of document context block

    # Add the user's current query
    context_parts.append(f"\nUser Query: {request.query}")

    full_prompt = "".join(context_parts)

    # Optional: Log the approximate size for debugging
    logger.info(f"Approximate prompt length: {len(full_prompt)} characters.")
    # WARNING: Character count != token count. 1M tokens is roughly 4M chars for English.
    # This PoC might hit limits easily. A production system needs token counting.
    # Example using tiktoken (needs `pip install tiktoken`):
    # import tiktoken
    # enc = tiktoken.get_encoding("cl100k_base") # Or appropriate encoding for Gemini
    # token_count = len(enc.encode(full_prompt))
    # logger.info(f"Estimated token count: {token_count}")


    # 4. Send to LLM
    try:
        logger.info(f"Sending prompt to model {LLM_MODEL_NAME}...")
        # Use async generation within FastAPI endpoint
        response = await llm_model.generate_content_async(full_prompt)
        logger.info("Received response from LLM.")

        # Check for safety ratings or blocks if necessary
        # print(response.prompt_feedback) # Check if prompt was blocked
        # print(response.candidates[0].finish_reason)
        # print(response.candidates[0].safety_ratings)

        llm_response_text = ""
        if response and hasattr(response, 'text'):
            llm_response_text = response.text
        elif response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             # Handle potential multi-part responses if 'text' isn't directly available
             llm_response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
        else:
             logger.warning(f"LLM response received but no text found. Response: {response}")
             # Check for blocked content
             try:
                 if response.prompt_feedback.block_reason:
                      block_reason = response.prompt_feedback.block_reason
                      logger.error(f"Prompt was blocked by API. Reason: {block_reason}")
                      raise HTTPException(status_code=400, detail=f"Request blocked due to safety settings: {block_reason}")
                 if response.candidates and response.candidates[0].finish_reason != 'STOP':
                      finish_reason = response.candidates[0].finish_reason
                      logger.error(f"LLM generation finished unexpectedly. Reason: {finish_reason}")
                      # Check safety ratings for clues
                      safety_msg = ""
                      if response.candidates[0].safety_ratings:
                           safety_msg = f" Safety Ratings: {response.candidates[0].safety_ratings}"
                      raise HTTPException(status_code=500, detail=f"LLM generation failed: {finish_reason}.{safety_msg}")
             except Exception as e:
                 if isinstance(e, HTTPException): raise e # Re-raise if we already created one
                 logger.error(f"Error interpreting LLM response structure: {e}", exc_info=True)
                 raise HTTPException(status_code=500, detail="Failed to interpret LLM response.")


        return ChatResponse(response=llm_response_text, sources_consulted=source_filenames)

    except Exception as e:
        logger.error(f"Error during LLM interaction for project {project_name}: {e}", exc_info=True)
        # Check if it's an HTTPException we raised earlier (e.g., blocked prompt)
        if isinstance(e, HTTPException):
             raise e
        # Handle potential API errors from google-generativeai library
        # (Specific error types depend on the library, check its docs)
        # Example pseudo-code:
        # except genai.types.BlockedPromptException as bpe:
        #    logger.error(f"Prompt blocked: {bpe}")
        #    raise HTTPException(status_code=400, detail=f"Request blocked by safety filter: {bpe}")
        # except genai.types.StopCandidateException as sce:
        #    logger.error(f"Content generation stopped: {sce}")
        #    # Might contain partial response or safety info
        #    raise HTTPException(status_code=500, detail=f"LLM generation stopped unexpectedly: {sce}")
        # except Exception as e: # Generic fallback
        raise HTTPException(status_code=500, detail=f"An error occurred during LLM interaction: {e}")


# --- Test LLM Endpoint (Keep for basic checks) ---
@app.get("/test-llm")
async def test_llm_connection():
    """ Endpoint to test basic interaction with the configured LLM. """
    logger.info("LLM test endpoint '/test-llm' accessed.")
    if not llm_model:
         raise HTTPException(status_code=503, detail="LLM service unavailable: Model not initialized.")
    try:
        prompt = "Hello Gemini! Respond with a short confirmation."
        logger.info(f"Sending test prompt: '{prompt}'")
        response = await llm_model.generate_content_async(prompt)
        logger.info("Received response from LLM.")
        if response and hasattr(response, 'text'):
             return {"status": "ok", "model_response": response.text}
        else:
             logger.warning(f"Received unexpected LLM response format: {response}")
             raise HTTPException(status_code=500, detail="Received unexpected response format from LLM.")
    except Exception as e:
        logger.error(f"Error during LLM test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to communicate with LLM: {e}")


# --- Uvicorn runner (for `python main.py`) ---
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting Uvicorn server directly from main.py")
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Added reload=True here too

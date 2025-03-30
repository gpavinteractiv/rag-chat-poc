#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd
import tiktoken

# Add backend directory to sys.path to allow importing parsing_utils
# This assumes the script is run from the project root directory
project_root = Path(__file__).parent.parent.resolve()
backend_dir = project_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from parsing_utils import parse_pdf, parse_docx, parse_markdown, parse_csv, DocumentContent
except ImportError as e:
    print(f"Error: Could not import parsing utilities from {backend_dir}. Ensure the backend directory is correct and dependencies are installed.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    sys.exit(1)

# --- Constants ---
TOKENIZER_MODEL = "cl100k_base" # Using cl100k_base (common for GPT-4) for estimation
FILELIST_NAME = "filelist.csv"
FILENAME_COLUMN = "file_name" # Expected column name for filenames
TOKEN_COUNT_COLUMN = "token_count" # Column to add/update

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Token Counting Utility ---
# Cache the tokenizer encoding
try:
    tokenizer = tiktoken.get_encoding(TOKENIZER_MODEL)
    logger.info(f"Using tokenizer: {TOKENIZER_MODEL}")
except Exception as e:
    logger.error(f"Failed to load tokenizer '{TOKENIZER_MODEL}': {e}", exc_info=True)
    sys.exit(1)

def count_tokens(text: str) -> int:
    """Counts tokens in the text using the pre-loaded tokenizer."""
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        logger.error(f"Error encoding text for token count: {e}. Text length: {len(text)}", exc_info=True)
        return -1 # Indicate error

# --- Main Script Logic ---
def process_project(project_dir: Path):
    """Parses documents and updates token counts in the project's filelist.csv."""
    logger.info(f"Processing project directory: {project_dir}")

    filelist_path = project_dir / FILELIST_NAME
    if not filelist_path.is_file():
        logger.error(f"{FILELIST_NAME} not found in {project_dir}. Skipping.")
        return

    logger.info(f"Reading {filelist_path}...")
    try:
        # Read CSV, keeping existing data
        df = pd.read_csv(filelist_path)
        if FILENAME_COLUMN not in df.columns:
             logger.error(f"Required column '{FILENAME_COLUMN}' not found in {filelist_path}. Skipping.")
             return

        # Ensure token_count column exists, initialize with NaN or keep existing
        if TOKEN_COUNT_COLUMN not in df.columns:
            df[TOKEN_COUNT_COLUMN] = pd.NA
        # Convert to a numeric type that supports NA (like float), then maybe Int64 later
        df[TOKEN_COUNT_COLUMN] = pd.to_numeric(df[TOKEN_COUNT_COLUMN], errors='coerce')


    except Exception as e:
        logger.error(f"Error reading or processing {filelist_path}: {e}", exc_info=True)
        return

    logger.info(f"Calculating token counts for {len(df)} files listed...")
    new_counts = []
    processed_count = 0
    error_count = 0

    for index, row in df.iterrows():
        filename = row[FILENAME_COLUMN]
        if pd.isna(filename):
            logger.warning(f"Skipping row {index+2} due to missing filename.")
            new_counts.append(row[TOKEN_COUNT_COLUMN]) # Keep existing value if filename missing
            continue

        file_path = (project_dir / filename).resolve()
        logger.info(f"Processing file: {filename} ({file_path})")

        if project_dir not in file_path.parents:
            logger.warning(f"Skipping file outside project directory: {filename}")
            new_counts.append(pd.NA) # Or set to -1 to indicate skipped? Using NA for now.
            continue

        if not file_path.is_file():
            logger.warning(f"File not found: {filename} at {file_path}. Setting token count to 0.")
            new_counts.append(0)
            continue

        # Parse the document based on suffix
        suffix = file_path.suffix.lower()
        doc_content: Optional[DocumentContent] = None
        try:
            if suffix == ".pdf": doc_content = parse_pdf(file_path)
            elif suffix == ".docx": doc_content = parse_docx(file_path)
            elif suffix == ".md": doc_content = parse_markdown(file_path)
            elif suffix == ".csv" and file_path.name != FILELIST_NAME:
                 doc_content = parse_csv(file_path)
            # Add other parsers here if needed
            else:
                logger.warning(f"Unsupported file type for token counting: {filename} ({suffix}). Setting count to 0.")
                new_counts.append(0)
                continue # Skip token counting for unsupported types

            # Count tokens
            if doc_content and not doc_content.error and doc_content.content:
                token_count = count_tokens(doc_content.content)
                logger.info(f" -> Estimated tokens: {token_count}")
                new_counts.append(token_count)
                processed_count += 1
            elif doc_content and doc_content.error:
                 logger.error(f" -> Parsing error: {doc_content.error}. Setting token count to -1.")
                 new_counts.append(-1) # Indicate error with -1
                 error_count += 1
            else: # No content or other issue
                 logger.warning(f" -> No content parsed or unexpected issue. Setting token count to 0.")
                 new_counts.append(0)

        except Exception as e:
             logger.error(f" -> Unexpected error processing file {filename}: {e}", exc_info=True)
             new_counts.append(-1) # Indicate error
             error_count += 1

    # Update the DataFrame column
    # Ensure the length matches before assigning
    if len(new_counts) == len(df):
        df[TOKEN_COUNT_COLUMN] = new_counts
        # Attempt to convert to nullable Int64 if possible (requires pandas >= 1.0)
        try:
            df[TOKEN_COUNT_COLUMN] = df[TOKEN_COUNT_COLUMN].astype('Int64')
        except Exception:
            logger.debug("Could not convert token_count column to Int64.")
            pass # Keep as float if conversion fails

        # Save the updated DataFrame back to CSV
        try:
            logger.info(f"Saving updated {filelist_path}...")
            df.to_csv(filelist_path, index=False, encoding='utf-8')
            logger.info(f"Successfully updated {filelist_path}. Processed: {processed_count}, Errors/Skipped: {error_count + df[FILENAME_COLUMN].isna().sum()}")
        except Exception as e:
            logger.error(f"Error writing updated {filelist_path}: {e}", exc_info=True)
    else:
         logger.error(f"Mismatch between number of rows processed ({len(new_counts)}) and original DataFrame rows ({len(df)}). Aborting save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Calculate estimated token counts for documents listed in a project's {FILELIST_NAME} and update the file.")
    parser.add_argument("project_directory", type=str, help="Path to the project directory containing the filelist.csv.")
    args = parser.parse_args()

    project_path = Path(args.project_directory).resolve()

    if not project_path.is_dir():
        logger.error(f"Provided path is not a valid directory: {project_path}")
        sys.exit(1)

    process_project(project_path)

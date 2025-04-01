#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified script to process project documents:
1. Pre-generate disk cache of parsed document content for faster loading
2. Calculate token counts for documents and update filelist.csv

This script combines and improves functionality from:
- calculate_token_counts.py 
- pregenerate_doc_cache.py
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import os
import time
import datetime
import traceback
import concurrent.futures
import shutil
from typing import Optional, Dict, List, Tuple, Set

# --- Add backend directory to sys.path ---
# This script is now located in backend/scripts/
try:
    script_dir = Path(__file__).parent.resolve()
    backend_dir = script_dir.parent.resolve() # backend directory is one level up
    project_root = backend_dir.parent.resolve() # project root is two levels up
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from parsing_utils import parse_pdf, parse_docx, parse_markdown, parse_csv, DocumentContent
except ImportError as e:
    print(f"Error: Could not import parsing utilities from {backend_dir}. Ensure the backend directory is correct and dependencies are installed.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import setup: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# --- Constants ---
CACHE_DIR_NAME = ".cache"
PARSED_DOCS_SUBDIR = "parsed_docs"
CACHE_VERSION = 1 # Increment if cache structure changes
FILELIST_NAME = "filelist.csv"
FILENAME_COLUMN = "file_name" # Expected column name for filenames
TOKEN_COUNT_COLUMN = "token_count" # Column to add/update
FILE_NUMBER_COLUMN = "file_number" # Column for file numbering
DATE_COLUMN = "date" # Column for date 
DESCRIPTION_COLUMN = "description" # Column for description
TEMPLATE_DIR_NAME = "_template" # Name of the template directory
TOKENIZER_MODEL = "cl100k_base" # Using cl100k_base (common for GPT-4) for estimation

# List of supported file extensions
SUPPORTED_FILE_EXTENSIONS = [".pdf", ".docx", ".md", ".csv"]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def get_modification_time(file_path: Path) -> Optional[float]:
    """Gets the last modification time of a file."""
    try:
        return file_path.stat().st_mtime
    except FileNotFoundError:
        logger.warning(f"File not found when checking modification time: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error getting modification time for {file_path}: {e}", exc_info=True)
        return None

def write_cache_file(cache_file_path: Path, data: dict):
    """Writes data to a JSON cache file."""
    try:
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully wrote cache file: {cache_file_path}")
    except Exception as e:
        logger.error(f"Error writing cache file {cache_file_path}: {e}", exc_info=True)

def find_project_files(project_dir: Path) -> List[str]:
    """Find all supported files in the project directory (excluding filelist.csv itself)."""
    supported_files = []
    for ext in SUPPORTED_FILE_EXTENSIONS:
        for file_path in project_dir.glob(f"**/*{ext}"):
            if file_path.name != FILELIST_NAME and project_dir in file_path.parents:
                # Get relative path from project_dir
                rel_path = file_path.relative_to(project_dir)
                supported_files.append(str(rel_path))
    return supported_files

def ensure_filelist_exists(project_dir: Path, projects_base_dir: Path) -> bool:
    """
    Ensure the filelist.csv exists for a project. If it doesn't, create it from the template.
    Returns True if filelist was created or already exists, False on error.
    """
    filelist_path = project_dir / FILELIST_NAME
    
    # If filelist already exists, we're good
    if filelist_path.is_file():
        return True
        
    # If not, look for the template
    template_dir = projects_base_dir / TEMPLATE_DIR_NAME
    template_filelist = template_dir / FILELIST_NAME
    
    if not template_filelist.is_file():
        logger.error(f"Template filelist not found at {template_filelist}. Cannot create filelist.")
        return False
    
    try:
        # Copy the template
        logger.info(f"Creating filelist.csv from template for project {project_dir.name}")
        shutil.copy(template_filelist, filelist_path)
        
        # Now populate it with files from the project
        try:
            import pandas as pd
            
            # Read the template file - preserve exact structure
            with open(template_filelist, 'r') as f:
                header_line = f.readline().strip()
            
            columns = header_line.split(',')
            
            # Find all supported files in the project
            project_files = find_project_files(project_dir)
            if not project_files:
                logger.warning(f"No supported files found in {project_dir}")
                return True  # Return True even though no files were found - filelist exists
            
            # Create new file with exact same structure
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            
            # Check if columns exist in the template
            file_number_col = next((col for col in columns if col.lower().replace(' ', '').startswith('file') and col.lower().replace(' ', '').endswith('number')), FILE_NUMBER_COLUMN)
            filename_col = next((col for col in columns if col.lower().replace(' ', '').startswith('file') and col.lower().replace(' ', '').endswith('name')), FILENAME_COLUMN)
            date_col = next((col for col in columns if col.lower().replace(' ', '') == 'date'), DATE_COLUMN)
            desc_col = next((col for col in columns if col.lower().replace(' ', '').startswith('desc')), DESCRIPTION_COLUMN)
            token_col = next((col for col in columns if col.lower().replace(' ', '').startswith('token')), TOKEN_COUNT_COLUMN)
            
            # Create rows with exact column structure
            with open(filelist_path, 'w') as f:
                f.write(header_line + '\n')  # Write the exact header
                
                # Write each file
                for i, file_path in enumerate(project_files):
                    row_values = []
                    for col in columns:
                        if col.lower().replace(' ', '') == file_number_col.lower().replace(' ', ''):
                            row_values.append(str(i+1))
                        elif col.lower().replace(' ', '') == filename_col.lower().replace(' ', ''):
                            row_values.append(file_path)
                        elif col.lower().replace(' ', '') == date_col.lower().replace(' ', ''):
                            row_values.append(today)
                        elif col.lower().replace(' ', '') == desc_col.lower().replace(' ', ''):
                            row_values.append('')
                        elif col.lower().replace(' ', '') == token_col.lower().replace(' ', ''):
                            row_values.append('')
                        else:
                            row_values.append('')
                    
                    f.write(','.join(row_values) + '\n')
            
            logger.info(f"Successfully created and populated {filelist_path} with {len(project_files)} files")
            return True
            
        except ImportError:
            logger.error("Pandas not found. Filelist was created from template but not populated with files.")
            return True  # Return True because at least we copied the template
        except Exception as e:
            logger.error(f"Error populating filelist with project files: {e}")
            return True  # Return True because at least we copied the template
            
    except Exception as e:
        logger.error(f"Error creating filelist from template: {e}")
        return False

# --- Token Counting Utility ---
# Cache the tokenizer encoding
try:
    import tiktoken
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

# --- Main Processing Logic ---
def process_project(
    project_dir: Path, 
    cache_base_dir: Path,  # This parameter is retained for backward compatibility
    projects_base_dir: Path,
    token_counts_only: bool = False,
    cache_only: bool = False,
    use_parallel: bool = False,
    force_regenerate: bool = False
):
    """
    Process a project: parse documents, update token counts, and/or generate disk cache.
    
    Args:
        project_dir: Path to the project directory
        cache_base_dir: Path to the cache directory (deprecated - cache now stored in project dir)
        projects_base_dir: Path to the base projects directory
        token_counts_only: If True, only update token counts in filelist.csv (no cache)
        cache_only: If True, only generate disk cache (no CSV update)
        use_parallel: If True, use parallel processing for document parsing
        force_regenerate: If True, regenerate cache and update filelist even if they exist
    """
    project_name = project_dir.name
    logger.info(f"Processing project: {project_name}")
    logger.info(f"  - Token counts only: {token_counts_only}")
    logger.info(f"  - Cache only: {cache_only}")
    logger.info(f"  - Use parallel: {use_parallel}")
    logger.info(f"  - Force regenerate: {force_regenerate}")

    # Ensure filelist.csv exists by creating it from template if needed
    if not ensure_filelist_exists(project_dir, projects_base_dir):
        logger.error(f"Failed to ensure filelist.csv exists for {project_name}. Skipping.")
        return

    filelist_path = project_dir / FILELIST_NAME
    
    # Create .cache directory in the project directory
    project_cache_dir = project_dir / ".cache"
    project_cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file_path = project_cache_dir / "docs.json"
    
    logger.info(f"Using project-local cache at: {cache_file_path}")
    
    # If force_regenerate is True, delete existing cache file
    if force_regenerate and cache_file_path.exists() and not token_counts_only:
        try:
            logger.info(f"Deleting existing cache file: {cache_file_path}")
            cache_file_path.unlink()
        except Exception as e:
            logger.error(f"Error deleting cache file {cache_file_path}: {e}")
    
    # Storage for results
    parsed_documents = {}  # filename -> DocumentContent
    document_cache_entries = {} # filename -> cache entry
    
    try:
        # Use pandas to read the file list robustly
        import pandas as pd  # Import pandas here to avoid making it a hard dep
        
        try:
            # If force_regenerate is True, get all files again
            if force_regenerate and not cache_only:
                logger.info(f"Force regenerating filelist.csv for {project_name}")
                
                # Back up the existing filelist
                backup_path = filelist_path.with_suffix(f".csv.bak")
                try:
                    shutil.copy(filelist_path, backup_path)
                    logger.info(f"Created backup of filelist at {backup_path}")
                except Exception as e:
                    logger.error(f"Error creating filelist backup: {e}")
                
                # Find all project files
                project_files = find_project_files(project_dir)
                if not project_files:
                    logger.warning(f"No supported files found in {project_dir}")
                
                # Read existing filelist to preserve any manual entries
                try:
                    existing_df = pd.read_csv(filelist_path)
                    has_existing = True
                except Exception:
                    # If we can't read it, create a new one
                    existing_df = pd.DataFrame()
                    has_existing = False
                
                # Create new dataframe with the files
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                rows = []
                
                for i, file_path in enumerate(project_files):
                    # Try to preserve existing data for this file
                    if has_existing and FILENAME_COLUMN in existing_df.columns:
                        existing_row = existing_df[existing_df[FILENAME_COLUMN] == file_path]
                        if not existing_row.empty:
                            # Preserve existing values
                            row = {col: existing_row[col].iloc[0] 
                                  for col in existing_df.columns 
                                  if col in existing_row}
                            row[FILENAME_COLUMN] = file_path  # Ensure filename is set
                            rows.append(row)
                            continue
                    
                    # Create new row if no existing data
                    rows.append({
                        FILE_NUMBER_COLUMN: i+1,
                        FILENAME_COLUMN: file_path,
                        DATE_COLUMN: today,
                        DESCRIPTION_COLUMN: "",
                        TOKEN_COUNT_COLUMN: None
                    })
                
                # Get template structure
                template_dir = projects_base_dir / TEMPLATE_DIR_NAME
                template_filelist = template_dir / FILELIST_NAME
                
                # Instead of using pandas to save, manually preserve the exact format
                if template_filelist.is_file():
                    # Read template header
                    with open(template_filelist, 'r') as f:
                        header_line = f.readline().strip()
                    
                    columns = header_line.split(',')
                    
                    # Create a structured CSV with exact same header format
                    with open(filelist_path, 'w') as f:
                        f.write(header_line + '\n')  # Write the exact header
                        
                        # Write each file, preserving data from existing rows where possible
                        for i, row_data in enumerate(rows):
                            row_values = []
                            for col in columns:
                                col_clean = col.lower().replace(' ', '')
                                # Try to find matching column
                                if col_clean.startswith('file') and col_clean.endswith('number'):
                                    row_values.append(str(row_data.get(FILE_NUMBER_COLUMN, i+1)))
                                elif col_clean.startswith('file') and col_clean.endswith('name'):
                                    row_values.append(str(row_data.get(FILENAME_COLUMN, '')))
                                elif col_clean == 'date':
                                    row_values.append(str(row_data.get(DATE_COLUMN, '')))
                                elif col_clean.startswith('desc'):
                                    row_values.append(str(row_data.get(DESCRIPTION_COLUMN, '')))
                                elif col_clean.startswith('token'):
                                    row_values.append(str(row_data.get(TOKEN_COUNT_COLUMN, '')))
                                else:
                                    # For any other columns in template, add empty value or preserve
                                    row_values.append(str(row_data.get(col, '')))
                            
                            f.write(','.join(row_values) + '\n')
                else:
                    # Fallback to pandas if template not found
                    df = pd.DataFrame(rows)
                    df.to_csv(filelist_path, index=False)
                
                logger.info(f"Updated {filelist_path} with {len(rows)} files (preserving template structure)")
                
                # After manually writing the file, read it back with pandas for the rest of the processing
                df = pd.read_csv(filelist_path)
            else:
                # Normal operation - just read the existing filelist
                # Figure out which columns we need
                df = pd.read_csv(filelist_path)
                
                # Ensure token_count column exists if needed
                if not cache_only and TOKEN_COUNT_COLUMN not in df.columns:
                    logger.info(f"Adding '{TOKEN_COUNT_COLUMN}' column to {filelist_path}")
                    df[TOKEN_COUNT_COLUMN] = pd.NA
                
                # Convert token count to numeric if it exists
                if TOKEN_COUNT_COLUMN in df.columns:
                    df[TOKEN_COUNT_COLUMN] = pd.to_numeric(df[TOKEN_COUNT_COLUMN], errors='coerce')
            
            # Verify we have the filename column
            if FILENAME_COLUMN not in df.columns:
                logger.error(f"Required column '{FILENAME_COLUMN}' not found in {filelist_path}. Skipping.")
                return
                
        except ValueError as ve:
            logger.error(f"Required column(s) not found in {filelist_path}. Skipping. Error: {ve}")
            return
        except pd.errors.EmptyDataError:
            logger.warning(f"{filelist_path} is empty. No documents to process.")
            # Since we're not processing anything, write empty outputs if needed
            if not token_counts_only:
                write_cache_file(cache_file_path, {
                    "cache_version": CACHE_VERSION,
                    "project_name": project_name,
                    "generation_timestamp": time.time(),
                    "documents": {}
                })
            return
        except Exception as e:
            logger.error(f"Error reading {filelist_path}: {e}. Skipping.", exc_info=True)
            return

        # Get list of valid filenames to process
        df_valid_filenames = df.dropna(subset=[FILENAME_COLUMN])
        filenames_to_process = df_valid_filenames[FILENAME_COLUMN].dropna().unique().tolist()
        logger.info(f"Found {len(filenames_to_process)} unique filenames to process.")
        
        # Check for existing cache to potentially skip unchanged files
        existing_cache = {}
        if not token_counts_only and not force_regenerate and cache_file_path.exists():
            try:
                with open(cache_file_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # Check if cache is valid format 
                if cache_data.get("cache_version") == CACHE_VERSION and cache_data.get("project_name") == project_name:
                    existing_cache = cache_data.get("documents", {})
                    if existing_cache:
                        logger.info(f"Found existing cache with {len(existing_cache)} document(s).")
            except Exception as e:
                logger.warning(f"Could not read existing cache: {e}")
                existing_cache = {}

        # Define function to process a single document
        def process_single_document(filename: str) -> Tuple[str, Dict, Optional[int]]:
            """Process a single document, returns (filename, cache_entry, token_count)"""
            file_path = (project_dir / filename).resolve()
            logger.info(f"Processing: {filename} ({file_path})")
            
            # Get precalculated token count if needed and available
            precalculated_token_count = None
            if TOKEN_COUNT_COLUMN in df.columns:
                row = df[df[FILENAME_COLUMN] == filename]
                if not row.empty:
                    count_value = row[TOKEN_COUNT_COLUMN].iloc[0]
                    if pd.notna(count_value):
                        precalculated_token_count = int(count_value)
            
            # Skip unchanged files if they exist in cache
            if not token_counts_only and not force_regenerate and filename in existing_cache:
                current_mod_time = get_modification_time(file_path)
                cached_mod_time = existing_cache[filename].get("source_mod_time")
                
                if (current_mod_time is not None and cached_mod_time is not None and 
                    current_mod_time <= cached_mod_time):
                    logger.info(f"Using cached version of {filename} (unchanged).")
                    
                    # Return cached info
                    cache_entry = existing_cache[filename]
                    token_count = cache_entry.get("token_count")
                    return filename, cache_entry, token_count
            
            # Check if file exists and is accessible
            if project_dir.resolve() not in file_path.resolve().parents:
                logger.warning(f"Skipping file outside project directory: {filename}")
                error_msg = "Access Denied: File is outside project scope."
                
                cache_entry = {
                    "parsed_content": None,
                    "source_mod_time": None,
                    "parsing_error": error_msg,
                    "token_count": None
                }
                return filename, cache_entry, None
                
            if not file_path.is_file():
                logger.warning(f"File not found: {filename} at {file_path}")
                error_msg = "File not found or inaccessible."
                
                cache_entry = {
                    "parsed_content": None, 
                    "source_mod_time": None,
                    "parsing_error": error_msg,
                    "token_count": None
                }
                return filename, cache_entry, None
            
            # Get file modification time
            mod_time = get_modification_time(file_path)
            if mod_time is None:
                logger.warning(f"Could not get modification time for {filename}")
                error_msg = "File inaccessible or error getting modification time."
                
                cache_entry = {
                    "parsed_content": None,
                    "source_mod_time": None, 
                    "parsing_error": error_msg,
                    "token_count": None
                }
                return filename, cache_entry, None
            
            # Parse the document based on suffix
            suffix = file_path.suffix.lower()
            doc_content: Optional[DocumentContent] = None
            token_count = None
            
            try:
                if suffix == ".pdf":
                    # Use sequential processing by default for reliability
                    doc_content = parse_pdf(file_path, precalculated_token_count=precalculated_token_count, use_parallel=use_parallel)
                elif suffix == ".docx":
                    doc_content = parse_docx(file_path)
                elif suffix == ".md":
                    doc_content = parse_markdown(file_path)
                elif suffix == ".csv" and file_path.name != FILELIST_NAME:
                    doc_content = parse_csv(file_path)
                else:
                    logger.warning(f"Unsupported file type for processing: {filename} ({suffix})")
                    doc_content = DocumentContent(
                        filename=filename, 
                        content=None, 
                        error=f"Unsupported file type: {suffix}"
                    )
                
                # Count tokens if needed and we have content
                if doc_content and doc_content.content and (doc_content.token_count is None or token_counts_only or force_regenerate):
                    token_count = count_tokens(doc_content.content)
                    logger.info(f" -> Calculated token count: {token_count}")
                    
                    # Update DocumentContent object
                    doc_content.token_count = token_count
                
                # Create cache entry
                if doc_content:
                    cache_entry = {
                        "parsed_content": doc_content.content,
                        "source_mod_time": mod_time,
                        "token_count": doc_content.token_count,
                        "parsing_error": doc_content.error
                    }
                    
                    # Log success or errors
                    if doc_content.error:
                        logger.error(f" -> Parsing error for {filename}: {doc_content.error}")
                    else:
                        logger.info(f" -> Successfully processed {filename}")
                        
                    return filename, cache_entry, doc_content.token_count
                else:
                    # This shouldn't happen if all branches create DocumentContent
                    logger.error(f" -> Failed to get DocumentContent for {filename}")
                    cache_entry = {
                        "parsed_content": None,
                        "source_mod_time": mod_time,
                        "parsing_error": "Unknown processing failure",
                        "token_count": None
                    }
                    return filename, cache_entry, None
                    
            except Exception as e:
                logger.error(f" -> Unexpected error processing {filename}: {e}", exc_info=True)
                cache_entry = {
                    "parsed_content": None,
                    "source_mod_time": mod_time,
                    "parsing_error": f"Unexpected error: {e}",
                    "token_count": None
                }
                return filename, cache_entry, None
        
        # Process documents
        results = []
        if use_parallel and len(filenames_to_process) > 5:
            # Use parallel processing for larger document sets
            logger.info(f"Processing {len(filenames_to_process)} documents in parallel")
            max_workers = min(8, len(filenames_to_process))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all processing tasks
                future_to_filename = {
                    executor.submit(process_single_document, filename): filename 
                    for filename in filenames_to_process
                }
                
                # Process results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(future_to_filename)):
                    try:
                        filename, cache_entry, token_count = future.result()
                        results.append((filename, cache_entry, token_count))
                        logger.info(f"Progress: {i+1}/{len(filenames_to_process)} documents processed")
                    except Exception as e:
                        filename = future_to_filename[future]
                        logger.error(f"Error processing {filename}: {e}", exc_info=True)
        else:
            # Process sequentially
            logger.info(f"Processing {len(filenames_to_process)} documents sequentially")
            for i, filename in enumerate(filenames_to_process):
                try:
                    result = process_single_document(filename)
                    results.append(result)
                    logger.info(f"Progress: {i+1}/{len(filenames_to_process)} documents processed")
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}", exc_info=True)
        
        # Organize results
        for filename, cache_entry, token_count in results:
            # Store for the cache update
            document_cache_entries[filename] = cache_entry
            
            # Store token count for CSV update
            if not cache_only and token_count is not None:
                # Find row index for this filename
                mask = df[FILENAME_COLUMN] == filename
                if any(mask):
                    # Update token count
                    df.loc[mask, TOKEN_COUNT_COLUMN] = token_count
        
        # Update filelist.csv if needed
        if not cache_only and TOKEN_COUNT_COLUMN in df.columns:
            try:
                # Convert token count to Int64 if possible (better for CSV readability)
                try:
                    df[TOKEN_COUNT_COLUMN] = df[TOKEN_COUNT_COLUMN].astype('Int64')
                except:
                    # Keep as float if conversion fails
                    pass
                
                logger.info(f"Saving updated token counts to {filelist_path}")
                df.to_csv(filelist_path, index=False, encoding='utf-8')
                logger.info(f"Successfully updated {filelist_path}")
            except Exception as e:
                logger.error(f"Error saving updated CSV: {e}", exc_info=True)
        
        # Update cache file if needed
        if not token_counts_only:
            cache_data = {
                "cache_version": CACHE_VERSION,
                "project_name": project_name,
                "generation_timestamp": time.time(),
                "documents": document_cache_entries
            }
            write_cache_file(cache_file_path, cache_data)
            
        # Final summary
        logger.info(f"Successfully processed project {project_name}.")
        logger.info(f"  - Documents processed: {len(results)}")
        if not cache_only:
            token_counts = [count for _, _, count in results if count is not None]
            logger.info(f"  - Token counts updated: {len(token_counts)}")
        if not token_counts_only:
            cache_entries = [entry for _, entry, _ in results if entry.get("parsed_content") is not None]
            logger.info(f"  - Cache entries added/updated: {len(cache_entries)}")
            error_entries = [entry for _, entry, _ in results if entry.get("parsing_error") is not None]
            logger.info(f"  - Documents with parsing errors: {len(error_entries)}")
        
    except ImportError:
        logger.error("Pandas library not found. Please install it in the backend environment ('pip install pandas') to read filelist.csv.")
    except Exception as e:
        logger.error(f"Critical error processing project {project_name}: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process project documents: generate disk cache and/or update token counts.")
    parser.add_argument(
        "project_directory",
        type=str,
        help="Path to a specific project directory to process."
    )
    parser.add_argument(
        "--token-counts-only",
        action="store_true",
        help="Only update token counts in filelist.csv (no cache generation)"
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only generate cache (no filelist.csv update)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing for document parsing (less reliable for some PDFs)"
    )
    parser.add_argument(
        "--all-projects",
        action="store_true",
        help="Process all valid projects in the projects directory"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of filelist.csv and cache even if they exist"
    )

    args = parser.parse_args()

    # Re-calculate paths based on the script's new location
    script_dir = Path(__file__).parent.resolve()
    backend_dir = script_dir.parent.resolve() # backend directory is one level up
    project_root = backend_dir.parent.resolve() # project root is two levels up

    projects_base_dir = project_root / "projects"
    cache_target_dir = backend_dir / CACHE_DIR_NAME / PARSED_DOCS_SUBDIR # Cache is relative to backend dir

    if not projects_base_dir.is_dir():
        logger.critical(f"Projects base directory not found at {projects_base_dir}. Cannot proceed.")
        sys.exit(1)
        
    # Validate arguments
    if args.token_counts_only and args.cache_only:
        logger.critical("Cannot use both --token-counts-only and --cache-only at the same time.")
        sys.exit(1)

    # Process project(s)
    if args.all_projects:
        logger.info(f"Processing all projects in {projects_base_dir}...")
        processed_count = 0
        for item in projects_base_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                process_project(
                    item, 
                    cache_target_dir,
                    projects_base_dir,
                    token_counts_only=args.token_counts_only,
                    cache_only=args.cache_only,
                    use_parallel=args.parallel,
                    force_regenerate=args.regenerate
                )
                processed_count += 1
        logger.info(f"Finished processing all projects. Total processed: {processed_count}")
    else:
        # Process a single specified project
        project_path = Path(args.project_directory).resolve()
        if not project_path.is_dir():
            logger.critical(f"Provided project path is not a valid directory: {project_path}")
            sys.exit(1)
            
        if projects_base_dir not in project_path.parents and project_path != projects_base_dir:
            logger.critical(f"Provided project path is not within the main projects directory: {projects_base_dir}")
            sys.exit(1)
            
        if project_path.name.startswith('_'):
            logger.warning(f"Skipping specified project as it starts with an underscore: {project_path.name}")
        else:
            process_project(
                project_path, 
                cache_target_dir,
                projects_base_dir,
                token_counts_only=args.token_counts_only,
                cache_only=args.cache_only,
                use_parallel=args.parallel,
                force_regenerate=args.regenerate
            )

    logger.info("Document processing script finished.")

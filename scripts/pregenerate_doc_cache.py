#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to pre-generate a disk cache of parsed document content for faster loading
in the backend application.
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import os
import time
import traceback

# --- Add backend directory to sys.path ---
# This assumes the script is run from the project root or via a wrapper in scripts/
try:
    project_root = Path(__file__).parent.parent.resolve()
    backend_dir = project_root / "backend"
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

# --- Main Processing Logic ---
def process_project_for_cache(project_dir: Path, cache_base_dir: Path):
    """Parses documents and creates/updates the disk cache for a single project."""
    project_name = project_dir.name
    logger.info(f"Processing project for cache generation: {project_name}")

    filelist_path = project_dir / FILELIST_NAME
    if not filelist_path.is_file():
        logger.error(f"{FILELIST_NAME} not found in {project_dir}. Skipping cache generation.")
        return

    cache_file_path = cache_base_dir / f"{project_name}.json"
    cached_documents = {}
    parse_errors = 0
    files_processed = 0

    try:
        # Use pandas to read the file list robustly
        import pandas as pd # Import pandas here to avoid making it a hard dep if only reading simple CSVs later
        try:
            df = pd.read_csv(filelist_path, usecols=[FILENAME_COLUMN], skipinitialspace=True)
        except ValueError as ve:
             logger.error(f"Required column '{FILENAME_COLUMN}' not found in {filelist_path}. Skipping cache generation. Error: {ve}")
             return
        except pd.errors.EmptyDataError:
             logger.warning(f"{filelist_path} is empty. No documents to cache.")
             # Write an empty cache file? Or just skip? Skipping for now.
             return
        except Exception as e:
             logger.error(f"Error reading {filelist_path}: {e}. Skipping cache generation.", exc_info=True)
             return

        filenames_to_parse = df[FILENAME_COLUMN].dropna().unique().tolist()
        logger.info(f"Found {len(filenames_to_parse)} unique filenames to parse and cache.")

        for filename in filenames_to_parse:
            files_processed += 1
            file_path = (project_dir / filename).resolve()
            logger.info(f"Parsing and caching: {filename} ({file_path})")

            if project_dir not in file_path.parents:
                logger.warning(f"Skipping file outside project directory: {filename}")
                cached_documents[filename] = {
                    "parsed_content": None,
                    "source_mod_time": None,
                    "parsing_error": "Access Denied: File is outside project scope."
                }
                parse_errors += 1
                continue

            mod_time = get_modification_time(file_path)
            if mod_time is None:
                 # File not found or error getting mod time
                 cached_documents[filename] = {
                    "parsed_content": None,
                    "source_mod_time": None,
                    "parsing_error": "File not found or inaccessible during cache generation."
                 }
                 parse_errors += 1
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
                    logger.warning(f"Unsupported file type for caching: {filename} ({suffix}). Storing error.")
                    doc_content = DocumentContent(filename=filename, content=None, error=f"Unsupported file type: {suffix}")

                # Store result in cache structure
                if doc_content:
                    cached_documents[filename] = {
                        "parsed_content": doc_content.content,
                        "source_mod_time": mod_time,
                        "parsing_error": doc_content.error
                    }
                    if doc_content.error:
                        logger.error(f" -> Parsing error for {filename}: {doc_content.error}")
                        parse_errors += 1
                    else:
                         logger.info(f" -> Successfully parsed and cached {filename}.")
                else:
                     # Should not happen if all branches create DocumentContent
                     logger.error(f" -> Failed to get DocumentContent object for {filename}. Storing error.")
                     cached_documents[filename] = {
                        "parsed_content": None,
                        "source_mod_time": mod_time,
                        "parsing_error": "Unknown parsing failure."
                     }
                     parse_errors += 1

            except Exception as e:
                 logger.error(f" -> Unexpected error parsing file {filename}: {e}", exc_info=True)
                 cached_documents[filename] = {
                    "parsed_content": None,
                    "source_mod_time": mod_time,
                    "parsing_error": f"Unexpected error: {e}"
                 }
                 parse_errors += 1

        # --- Write Cache File ---
        final_cache_data = {
            "cache_version": CACHE_VERSION,
            "project_name": project_name,
            "generation_timestamp": time.time(),
            "documents": cached_documents
        }
        write_cache_file(cache_file_path, final_cache_data)
        logger.info(f"Finished caching for project {project_name}. Files processed: {files_processed}, Errors: {parse_errors}")

    except ImportError:
         logger.error("Pandas library not found. Please install it in the backend virtual environment (`pip install pandas`) to read filelist.csv.")
         # Optionally, could implement a simpler CSV reader here as a fallback if pandas is truly optional
    except Exception as e:
        logger.error(f"Critical error processing project {project_name} for cache: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Pre-generate disk cache for parsed documents listed in project filelist.csv files.")
    parser.add_argument(
        "--project-directory",
        type=str,
        help="Path to a specific project directory to process. If omitted, all projects will be processed.",
        default=None
    )
    args = parser.parse_args()

    projects_base_dir = project_root / "projects"
    cache_target_dir = backend_dir / CACHE_DIR_NAME / PARSED_DOCS_SUBDIR

    if not projects_base_dir.is_dir():
        logger.critical(f"Projects base directory not found at {projects_base_dir}. Cannot proceed.")
        sys.exit(1)

    if args.project_directory:
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
             process_project_for_cache(project_path, cache_target_dir)
    else:
        # Process all valid projects
        logger.info(f"Processing all projects in {projects_base_dir}...")
        processed_count = 0
        for item in projects_base_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                process_project_for_cache(item, cache_target_dir)
                processed_count += 1
        logger.info(f"Finished processing all projects. Total processed: {processed_count}")

    logger.info("Cache pre-generation script finished.")

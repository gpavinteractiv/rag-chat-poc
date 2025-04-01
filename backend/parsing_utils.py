import logging
from pathlib import Path
import pdfplumber
from docx import Document as DocxDocument
import markdown as md_parser
import pandas as pd
import traceback
import concurrent.futures
from pydantic import BaseModel # Re-import BaseModel for DocumentContent
from typing import Optional, Tuple, List # Import Optional and collection types

# Setup logger for this module
logger = logging.getLogger(__name__)

# Re-define DocumentContent here or import from main if preferred (requires careful import handling)
class DocumentContent(BaseModel):
    filename: str
    content: Optional[str] # Content can be None if parsing fails
    token_count: Optional[int] = None # Add field for pre-calculated token count
    error: Optional[str] = None

def process_pdf_page(page_tuple: Tuple[int, object]) -> Tuple[int, str, Optional[str]]:
    """Process a single PDF page in parallel with improved error handling.
    
    Returns:
        Tuple containing:
        - Page number
        - Extracted text (or error placeholder)
        - Error message (if any)
    """
    i, page = page_tuple
    try:
        page_text = page.extract_text()
        if page_text:
            return (i, f"\n--- Page {i+1} ---\n{page_text}", None)
        else:
            return (i, f"\n--- Page {i+1} --- [No text extracted]\n", "No text extracted")
    except Exception as e:
        # Return standardized error format
        error_msg = f"{type(e).__name__}: {str(e)}"
        return (i, f"\n--- Page {i+1} --- [Error extracting text: {error_msg}]\n", error_msg)

def extract_page_sequentially(page, i: int) -> Tuple[str, Optional[str]]:
    """Extract text from a single page with detailed error handling."""
    try:
        page_text = page.extract_text()
        if page_text:
            return (f"\n--- Page {i+1} ---\n{page_text}", None)
        else:
            return (f"\n--- Page {i+1} --- [No text extracted]\n", "No text extracted")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.warning(f"Error extracting page {i+1}: {error_msg}")
        return (f"\n--- Page {i+1} --- [Error extracting text: {error_msg}]\n", error_msg)

def parse_pdf(file_path: Path, precalculated_token_count: Optional[int] = None, use_parallel: bool = False) -> DocumentContent:
    """
    Parses text content from a PDF file with improved error handling.
    
    Args:
        file_path: Path to the PDF file
        precalculated_token_count: Optional token count to use if parsing fails
        use_parallel: Whether to use parallel processing (defaults to False, sequential processing)
    
    Returns:
        DocumentContent object with extracted text and metadata
    """
    logger.info(f"Parsing PDF: {file_path.name}")
    full_text = ""
    error_msg = None
    extraction_errors = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF {file_path.name} has {total_pages} pages")
            
            # Sequential processing (default and most reliable)
            if not use_parallel or total_pages <= 5:
                if use_parallel and total_pages <= 5:
                    logger.info(f"Processing {file_path.name} sequentially (small document, parallel not needed)")
                else:
                    logger.info(f"Processing {file_path.name} sequentially")
                
                # Process pages one by one, continuing even if individual pages fail
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text, page_error = extract_page_sequentially(page, i)
                        full_text += page_text
                        if page_error:
                            extraction_errors.append(f"Page {i+1}: {page_error}")
                    except Exception as page_e:
                        # Catch any page-level exceptions and continue with next page
                        logger.warning(f"Error on page {i+1} of {file_path.name}: {page_e}")
                        extraction_errors.append(f"Page {i+1}: {str(page_e)}")
                        # Add placeholder for failed page
                        full_text += f"\n--- Page {i+1} --- [Error: {str(page_e)}]\n"
            
            # Parallel processing (optional, may fail on some PDFs)
            else:
                try:
                    logger.info(f"Processing {file_path.name} with parallel page extraction")
                    max_workers = min(8, total_pages)
                    page_tuples = [(i, page) for i, page in enumerate(pdf.pages)]
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        page_results = list(executor.map(process_pdf_page, page_tuples))
                    
                    # Sort results by page number and collect errors
                    page_texts = []
                    for page_num, text, error in page_results:
                        page_texts.append((page_num, text))
                        if error:
                            extraction_errors.append(f"Page {page_num+1}: {error}")
                    
                    # Sort by page number to maintain document order
                    page_texts.sort(key=lambda x: x[0])
                    full_text = "".join([text for _, text in page_texts])
                    
                except Exception as e:
                    # If parallel processing fails, fall back to sequential
                    logger.warning(f"Parallel processing failed for {file_path.name}, falling back to sequential. Error: {e}")
                    extraction_errors.append(f"Parallel processing failed: {str(e)}")
                    
                    # Reset text and try sequential processing
                    full_text = ""
                    for i, page in enumerate(pdf.pages):
                        try:
                            page_text, page_error = extract_page_sequentially(page, i)
                            full_text += page_text
                            if page_error:
                                extraction_errors.append(f"Page {i+1}: {page_error}")
                        except Exception as page_e:
                            # Catch page-level exceptions
                            logger.warning(f"Error on page {i+1} of {file_path.name}: {page_e}")
                            extraction_errors.append(f"Page {i+1}: {str(page_e)}")
                            full_text += f"\n--- Page {i+1} --- [Error: {str(page_e)}]\n"
        
        # Summarize errors if any
        if extraction_errors:
            error_count = len(extraction_errors)
            if error_count <= 5:
                error_msg = "PDF extraction issues: " + "; ".join(extraction_errors)
            else:
                error_msg = "PDF extraction issues: " + "; ".join(extraction_errors[:5]) + f"; and {error_count - 5} more errors"
            logger.warning(f" -> Parsing error for {file_path.name}: {error_msg}")
            
        if not full_text:
            logger.warning(f"No text extracted from the entire PDF: {file_path.name}")
            if not error_msg:
                error_msg = "No text could be extracted from PDF"
                
    except Exception as e:
        logger.error(f"Error parsing PDF {file_path.name}: {e}\n{traceback.format_exc()}")
        error_msg = f"Failed to parse PDF: {e}"
    
    # Even if parsing fails completely, we still return a valid object with the error info
    if error_msg and precalculated_token_count is not None:
        logger.debug(f" -> Assigned precalculated token count: {precalculated_token_count}")
    
    return DocumentContent(
        filename=file_path.name,
        content=full_text.strip() if full_text else "",  # Ensure content is never None
        token_count=precalculated_token_count,  # Use precalculated count if available
        error=error_msg
    )

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

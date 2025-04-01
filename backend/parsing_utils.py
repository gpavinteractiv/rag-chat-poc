import logging
from pathlib import Path
import pdfplumber
from docx import Document as DocxDocument
import markdown as md_parser
import pandas as pd
import traceback
from pydantic import BaseModel # Re-import BaseModel for DocumentContent
from typing import Optional # Import Optional

# Setup logger for this module
logger = logging.getLogger(__name__)

# Re-define DocumentContent here or import from main if preferred (requires careful import handling)
class DocumentContent(BaseModel):
    filename: str
    content: Optional[str] # Content can be None if parsing fails
    token_count: Optional[int] = None # Add field for pre-calculated token count
    error: Optional[str] = None

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

"""XLSX/XLS Conversion Module"""
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging
import tempfile
import os
import subprocess
import uuid
import json
import pandas as pd

logger = logging.getLogger(__name__)


def _create_chunks_with_metadata(text: str, excel_path: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into chunks and add Excel metadata to each chunk."""
    if not text or not text.strip():
        return []
    
    # Quick metadata extraction from Excel
    metadata_dict = {}
    try:
        with pd.ExcelFile(excel_path) as xl:
            for sheet in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet)
                if not df.empty:
                    metadata_dict[sheet] = {
                        "headers": list(df.columns)[:10],
                        "total_rows": len(df),
                        "total_columns": len(df.columns)
                    }
    except:
        pass
    
    # Split into chunks
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if current_length + len(line) > chunk_size and current_chunk:
            chunk_text = '\n'.join(current_chunk)
            
            # Detect sheet name
            sheet_name = "example"
            for sheet in metadata_dict.keys():
                if sheet.lower() in chunk_text.lower():
                    sheet_name = sheet
                    break
            
            # Build chunk with metadata
            chunk_obj = {
                "text": chunk_text,
                "metadata": {
                    "sheet_name": sheet_name,
                    "chunk_index": len(chunks),
                    "chunk_size": len(chunk_text)
                }
            }
            
            # Add sheet metadata if available
            if sheet_name in metadata_dict:
                chunk_obj["metadata"].update(metadata_dict[sheet_name])
            
            chunks.append(chunk_obj)
            
            # Overlap
            if overlap > 0 and len(current_chunk) > 1:
                overlap_lines = []
                overlap_len = 0
                for prev_line in reversed(current_chunk):
                    if overlap_len + len(prev_line) <= overlap:
                        overlap_lines.insert(0, prev_line)
                        overlap_len += len(prev_line)
                    else:
                        break
                current_chunk = overlap_lines
                current_length = overlap_len
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(line)
        current_length += len(line)
    
    # Add last chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        sheet_name = "Unknown"
        for sheet in metadata_dict.keys():
            if sheet.lower() in chunk_text.lower():
                sheet_name = sheet
                break
        
        chunk_obj = {
            "text": chunk_text,
            "metadata": {
                "sheet_name": sheet_name,
                "chunk_index": len(chunks),
                "chunk_size": len(chunk_text)
            }
        }
        if sheet_name in metadata_dict:
            chunk_obj["metadata"].update(metadata_dict[sheet_name])
        
        chunks.append(chunk_obj)
    
    return chunks


async def convert_xlsx_service(
    file: UploadFile,
    input_json: str = None,
    xml_file: str = None
):
    """
    Convert XLSX/XLS file to plain text using Apache Tika.
    Returns JSON with extracted text and metadata.
    
    Args:
        file: Excel file to convert
        input_json: Optional JSON string with additional data
        xml_file: Optional XML file content/path
    """
    # Validate file type
    if not (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .xlsx and .xls files are accepted."
        )
    
    try:
        # Read file content
        content = await file.read()
        logger.info(f"Processing Excel file: {file.filename}, size: {len(content)} bytes")
        
        # Check if content is empty
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="File is empty. Please ensure the file is properly uploaded."
            )
        
        # Validate Excel format
        # XLSX files start with PK (ZIP format)
        # XLS files start with various headers like D0CF (OLE2 format) or older formats
        is_xlsx = file.filename.endswith('.xlsx')
        is_xls = file.filename.endswith('.xls')
        
        if is_xlsx and len(content) >= 2 and not content[:2] == b'PK':
            raise HTTPException(
                status_code=400,
                detail=f"File does not appear to be a valid XLSX file."
            )
        
        if is_xls and len(content) >= 2:
            # XLS files typically start with D0CF (OLE2) or other legacy formats
            # We'll be more lenient here as there are multiple XLS format versions
            valid_xls_headers = [b'\xD0\xCF', b'\x09\x08', b'\x09\x04', b'\x09\x00']
            if not any(content[:2] == header for header in valid_xls_headers):
                logger.warning(f"XLS file may have unexpected format, proceeding anyway")
        
        # Determine file extension for temporary file
        file_ext = '.xlsx' if is_xlsx else '.xls'
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_input:
            tmp_input.write(content)
            tmp_input_path = tmp_input.name
        
        try:
            # Extract text using Apache Tika
            logger.info("Converting with Apache Tika...")
            result = subprocess.run([
                'java', '-jar', '/app/tika-app.jar', '--text', tmp_input_path
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"Tika extraction failed: {result.stderr}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to extract text from Excel file."
                )
            
            extracted_text = result.stdout.strip()
            
            if not extracted_text:
                raise HTTPException(
                    status_code=500,
                    detail="No text could be extracted from the document."
                )
            
            logger.info(f"Successfully extracted {len(extracted_text)} characters")
            
            # Create chunks with metadata
            chunks = _create_chunks_with_metadata(extracted_text, tmp_input_path)
            
            # Build response
            response_data = {
                "id": str(uuid.uuid4()),
                "success": True,
                "name": file.filename,
                "text": extracted_text,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "length": len(extracted_text),
                "method": "apache-tika",
                "file_type": "xlsx" if is_xlsx else "xls"
            }
            
            # Extend with input_json if provided
            if input_json:
                try:
                    additional_data = json.loads(input_json)
                    if isinstance(additional_data, dict):
                        response_data.update(additional_data)
                    else:
                        response_data["input_json"] = additional_data
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in input_json parameter")
                    response_data["input_json"] = input_json
            
            # Add xml_file if provided
            if xml_file:
                response_data["xml_file"] = xml_file
            
            return JSONResponse(content=response_data)
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

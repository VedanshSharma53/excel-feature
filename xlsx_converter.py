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


def _detect_header_row(df: pd.DataFrame, sheet_name: str) -> dict:
    """
    Intelligently detect if the first row is a header and extract metadata.
    
    Returns:
        dict with keys:
            - has_header: bool
            - header_row_index: int (0 if first row is header, None if no header)
            - detected_headers: list of column names
            - first_data_row: int
            - header_confidence: float (0-1)
            - reasons: list of detection reasons
    """
    if df.empty:
        return {
            "has_header": False,
            "header_row_index": None,
            "detected_headers": [],
            "first_data_row": 0,
            "header_confidence": 0.0,
            "reasons": ["Empty dataframe"]
        }
    
    reasons = []
    confidence_score = 0.0
    
    # Get first row and second row for comparison
    first_row = df.iloc[0] if len(df) > 0 else None
    second_row = df.iloc[1] if len(df) > 1 else None
    
    # Check 1: pandas default column names (Unnamed: 0, Unnamed: 1, etc.)
    has_unnamed_columns = any(str(col).startswith('Unnamed:') for col in df.columns)
    
    # Check 2: All values in first row are strings (common for headers)
    first_row_all_strings = all(isinstance(val, str) for val in first_row) if first_row is not None else False
    
    # Check 3: First row has unique values (headers should be unique)
    first_row_unique = len(first_row.dropna().unique()) == len(first_row.dropna()) if first_row is not None else False
    
    # Check 4: Data type consistency - if second row has different types than first
    type_mismatch = False
    if first_row is not None and second_row is not None:
        for col in df.columns:
            val1_type = type(first_row[col])
            val2_type = type(second_row[col])
            if val1_type != val2_type and not (pd.isna(first_row[col]) or pd.isna(second_row[col])):
                type_mismatch = True
                break
    
    # Check 5: First row has no numeric values (strong indicator of header)
    first_row_no_numbers = all(not isinstance(val, (int, float)) or pd.isna(val) for val in first_row) if first_row is not None else False
    
    # Check 6: Column names look like meaningful headers (not generic like 0, 1, 2)
    meaningful_column_names = not all(isinstance(col, int) for col in df.columns)
    generic_column_names = all(isinstance(col, int) for col in df.columns)
    
    # Decision logic
    has_header = False
    header_row_index = None
    
    if meaningful_column_names and not has_unnamed_columns:
        # pandas already detected headers
        has_header = True
        header_row_index = None  # Already used by pandas
        confidence_score = 0.9
        reasons.append("pandas detected meaningful column names")
    elif has_unnamed_columns or generic_column_names:
        # pandas didn't detect headers, check if first row should be headers
        if first_row_all_strings:
            confidence_score += 0.3
            reasons.append("First row contains all strings")
        if first_row_unique:
            confidence_score += 0.3
            reasons.append("First row values are unique")
        if first_row_no_numbers:
            confidence_score += 0.2
            reasons.append("First row has no numeric values")
        if type_mismatch:
            confidence_score += 0.2
            reasons.append("Type mismatch between row 1 and row 2")
        
        if confidence_score >= 0.5:
            has_header = True
            header_row_index = 0
        else:
            has_header = False
            reasons.append("First row appears to be data, not headers")
    
    # Extract detected headers
    if has_header and header_row_index == 0:
        detected_headers = [str(val) for val in first_row.values]
    elif has_header:
        detected_headers = [str(col) for col in df.columns]
    else:
        # No headers, use generic column names
        detected_headers = [f"Column_{i}" for i in range(len(df.columns))]
    
    first_data_row = (header_row_index + 1) if header_row_index is not None else 0
    
    return {
        "has_header": has_header,
        "header_row_index": header_row_index,
        "detected_headers": detected_headers,
        "first_data_row": first_data_row,
        "header_confidence": round(confidence_score, 2),
        "reasons": reasons
    }


def _analyze_sheet_structure(df: pd.DataFrame, sheet_name: str) -> dict:
    """
    Deep analysis of sheet structure for LLM context.
    
    Returns rich metadata about the sheet structure, data types, patterns, etc.
    """
    if df.empty:
        return {
            "is_empty": True,
            "sheet_name": sheet_name,
            "total_rows": 0,
            "total_columns": 0
        }
    
    # Header detection
    header_info = _detect_header_row(df, sheet_name)
    
    # Reread with correct header if needed
    if header_info["header_row_index"] == 0:
        # First row is header, recreate df with proper headers
        df_analyzed = df.copy()
        df_analyzed.columns = header_info["detected_headers"]
        df_analyzed = df_analyzed.iloc[1:].reset_index(drop=True)
    else:
        df_analyzed = df.copy()
        if not header_info["has_header"]:
            df_analyzed.columns = header_info["detected_headers"]
    
    # Column analysis
    column_info = []
    for col in df_analyzed.columns:
        col_data = df_analyzed[col].dropna()
        
        if len(col_data) == 0:
            col_type = "empty"
            sample_values = []
        else:
            # Detect column type
            if col_data.apply(lambda x: isinstance(x, (int, float))).all():
                col_type = "numeric"
            elif col_data.apply(lambda x: isinstance(x, str)).all():
                col_type = "text"
            elif col_data.apply(lambda x: isinstance(x, (pd.Timestamp, pd.DatetimeTZDtype))).all():
                col_type = "datetime"
            else:
                col_type = "mixed"
            
            # Get sample values (first 3 unique non-null values)
            sample_values = col_data.unique()[:3].tolist()
        
        column_info.append({
            "name": str(col),
            "type": col_type,
            "null_count": int(df_analyzed[col].isna().sum()),
            "unique_count": int(df_analyzed[col].nunique()),
            "sample_values": [str(v)[:100] for v in sample_values]  # Limit string length
        })
    
    # Pattern detection
    patterns = []
    
    # Check for ID columns
    id_columns = [col for col in df_analyzed.columns 
                  if any(keyword in str(col).lower() for keyword in ['id', 'ref', 'number', '#', 'no'])]
    if id_columns:
        patterns.append(f"ID columns detected: {', '.join(id_columns)}")
    
    # Check for title/name columns
    title_columns = [col for col in df_analyzed.columns 
                     if any(keyword in str(col).lower() for keyword in ['title', 'name', 'description', 'requirement'])]
    if title_columns:
        patterns.append(f"Title/Name columns detected: {', '.join(title_columns)}")
    
    # Check for date columns
    date_columns = [col for col in df_analyzed.columns 
                    if any(keyword in str(col).lower() for keyword in ['date', 'time', 'created', 'modified'])]
    if date_columns:
        patterns.append(f"Date columns detected: {', '.join(date_columns)}")
    
    # Check for categorical columns (low unique count relative to total)
    categorical_columns = [col for col in df_analyzed.columns 
                          if df_analyzed[col].nunique() < len(df_analyzed) * 0.1 and df_analyzed[col].nunique() > 1]
    if categorical_columns:
        patterns.append(f"Categorical columns detected: {', '.join(categorical_columns[:5])}")
    
    # Data quality metrics
    total_cells = len(df_analyzed) * len(df_analyzed.columns)
    null_cells = df_analyzed.isna().sum().sum()
    data_completeness = round((total_cells - null_cells) / total_cells * 100, 2) if total_cells > 0 else 0
    
    return {
        "is_empty": False,
        "sheet_name": sheet_name,
        "total_rows": len(df_analyzed),
        "total_columns": len(df_analyzed.columns),
        "total_data_rows": len(df_analyzed),  # Excluding header
        
        # Header information
        "header_detection": header_info,
        "headers": header_info["detected_headers"][:20],  # First 20 headers
        
        # Column details
        "columns": column_info[:20],  # First 20 columns detailed info
        
        # Patterns and insights
        "patterns": patterns,
        "has_id_column": len(id_columns) > 0,
        "has_title_column": len(title_columns) > 0,
        "has_date_column": len(date_columns) > 0,
        
        # Data quality
        "data_completeness_pct": data_completeness,
        "null_cells": int(null_cells),
        "total_cells": int(total_cells),
        
        # Sample data (first 2 rows for context)
        "sample_rows": df_analyzed.head(2).to_dict('records') if len(df_analyzed) > 0 else []
    }


def _create_chunks_with_metadata(text: str, excel_path: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into chunks and add Excel metadata to each chunk."""
    if not text or not text.strip():
        return []
    
    # Deep metadata extraction from Excel with intelligent header detection
    metadata_dict = {}
    sheet_markers = {}  # Track where each sheet's content appears in text
    
    try:
        with pd.ExcelFile(excel_path) as xl:
            current_position = 0
            for sheet in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet)
                if not df.empty:
                    # Use intelligent structure analysis
                    metadata_dict[sheet] = _analyze_sheet_structure(df, sheet)
                    
                    # Create a signature for this sheet to find it in text
                    # Use first few non-null values as markers
                    markers = []
                    for col in df.columns[:3]:  # First 3 columns
                        vals = df[col].dropna().head(2).astype(str).tolist()
                        markers.extend(vals)
                    
                    if markers:
                        # Find where this content appears in the text
                        search_text = ' '.join(markers[:5])  # Use first 5 values
                        pos = text.lower().find(search_text.lower())
                        if pos != -1:
                            sheet_markers[sheet] = pos
                        else:
                            # Fallback: use sheet name position
                            sheet_markers[sheet] = current_position
                        current_position = pos if pos != -1 else current_position + 1000
    except Exception as e:
        logger.warning(f"Could not extract detailed metadata: {e}")
        # Fallback to basic metadata
        try:
            with pd.ExcelFile(excel_path) as xl:
                for sheet in xl.sheet_names:
                    df = pd.read_excel(xl, sheet_name=sheet)
                    if not df.empty:
                        metadata_dict[sheet] = {
                            "sheet_name": sheet,
                            "headers": list(df.columns)[:10],
                            "total_rows": len(df),
                            "total_columns": len(df.columns)
                        }
        except:
            pass
    
    # If no sheets detected, use a default
    if not metadata_dict:
        metadata_dict["Sheet1"] = {"sheet_name": "Sheet1"}
    
    # Split into chunks
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    current_char_position = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            current_char_position += 1
            continue
        
        if current_length + len(line) > chunk_size and current_chunk:
            chunk_text = '\n'.join(current_chunk)
            
            # Improved sheet name detection
            sheet_name = _detect_sheet_for_chunk(
                chunk_text, 
                current_char_position, 
                metadata_dict, 
                sheet_markers
            )
            
            # Build chunk with metadata
            chunk_obj = {
                "text": chunk_text,
                "metadata": {
                    "sheet_name": sheet_name,
                    "chunk_index": len(chunks),
                    "chunk_size": len(chunk_text),
                    "char_position": current_char_position
                }
            }
            
            # Add sheet metadata if available
            if sheet_name in metadata_dict:
                sheet_meta = metadata_dict[sheet_name].copy()
                # Ensure sheet_name is set correctly
                sheet_meta["sheet_name"] = sheet_name
                chunk_obj["metadata"].update(sheet_meta)
            
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
        current_char_position += len(line) + 1
    
    # Add last chunk
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        sheet_name = _detect_sheet_for_chunk(
            chunk_text,
            current_char_position,
            metadata_dict,
            sheet_markers
        )
        
        chunk_obj = {
            "text": chunk_text,
            "metadata": {
                "sheet_name": sheet_name,
                "chunk_index": len(chunks),
                "chunk_size": len(chunk_text),
                "char_position": current_char_position
            }
        }
        
        if sheet_name in metadata_dict:
            sheet_meta = metadata_dict[sheet_name].copy()
            sheet_meta["sheet_name"] = sheet_name
            chunk_obj["metadata"].update(sheet_meta)
        
        chunks.append(chunk_obj)
    
    return chunks


def _detect_sheet_for_chunk(chunk_text: str, char_position: int, metadata_dict: dict, sheet_markers: dict) -> str:
    """
    Intelligently detect which sheet a chunk belongs to.
    
    Uses multiple strategies:
    1. Position-based detection (where in document)
    2. Content-based detection (matching text patterns)
    3. Sheet name in text
    """
    if not metadata_dict:
        return "Unknown"
    
    # Strategy 1: Use position markers
    if sheet_markers:
        # Find the closest sheet based on position
        best_sheet = None
        min_distance = float('inf')
        
        for sheet, position in sheet_markers.items():
            distance = abs(char_position - position)
            if distance < min_distance:
                min_distance = distance
                best_sheet = sheet
        
        if best_sheet:
            return best_sheet
    
    # Strategy 2: Check if sheet name appears in chunk
    chunk_lower = chunk_text.lower()
    for sheet in metadata_dict.keys():
        if sheet.lower() in chunk_lower:
            return sheet
    
    # Strategy 3: Check if any header values appear in chunk
    for sheet, meta in metadata_dict.items():
        if isinstance(meta, dict):
            headers = meta.get('headers', [])
            if isinstance(headers, list):
                # Check if multiple headers appear in chunk
                matches = sum(1 for h in headers[:5] if str(h).lower() in chunk_lower)
                if matches >= 2:  # At least 2 headers match
                    return sheet
            
            # Check sample values
            sample_rows = meta.get('sample_rows', [])
            if sample_rows and isinstance(sample_rows, list):
                for row in sample_rows[:2]:
                    if isinstance(row, dict):
                        values = [str(v).lower() for v in row.values() if v]
                        matches = sum(1 for v in values[:3] if v in chunk_lower)
                        if matches >= 2:
                            return sheet
    
    # Strategy 4: Default to first sheet
    return list(metadata_dict.keys())[0] if metadata_dict else "Unknown"


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

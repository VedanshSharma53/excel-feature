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
    
    # Enhanced column analysis with more details
    column_info = []
    numeric_columns = []
    text_columns = []
    date_columns = []
    boolean_columns = []
    
    for col in df_analyzed.columns:
        col_data = df_analyzed[col].dropna()
        
        if len(col_data) == 0:
            col_type = "empty"
            sample_values = []
            value_distribution = {}
            stats = {}
        else:
            # Enhanced type detection with sub-types
            col_type, stats = _detect_column_type_advanced(col_data, str(col))
            
            # Track columns by type
            if col_type in ["integer", "float", "numeric"]:
                numeric_columns.append(str(col))
            elif col_type in ["text", "string"]:
                text_columns.append(str(col))
            elif col_type in ["datetime", "date", "time"]:
                date_columns.append(str(col))
            elif col_type == "boolean":
                boolean_columns.append(str(col))
            
            # Get sample values (more diverse samples)
            if len(col_data) <= 5:
                sample_values = col_data.tolist()
            else:
                # Get first, middle, and last values for better representation
                sample_values = [
                    col_data.iloc[0],
                    col_data.iloc[len(col_data)//2],
                    col_data.iloc[-1]
                ]
                # Add unique values if column is categorical-like
                if col_data.nunique() <= 10:
                    sample_values = col_data.unique()[:5].tolist()
            
            # Value distribution for categorical-like columns
            value_distribution = {}
            if col_data.nunique() <= 20:  # Categorical-like
                value_counts = col_data.value_counts()
                value_distribution = {
                    str(k): int(v) for k, v in value_counts.head(10).items()
                }
        
        # Enhanced column metadata
        column_info.append({
            "name": str(col),
            "type": col_type,
            "null_count": int(df_analyzed[col].isna().sum()),
            "null_percentage": round((df_analyzed[col].isna().sum() / len(df_analyzed)) * 100, 2),
            "unique_count": int(df_analyzed[col].nunique()),
            "uniqueness_ratio": round(df_analyzed[col].nunique() / len(col_data), 2) if len(col_data) > 0 else 0,
            "sample_values": [str(v)[:100] for v in sample_values],
            "value_distribution": value_distribution if value_distribution else None,
            "statistics": stats
        })
    
    # Enhanced pattern detection
    patterns = []
    id_columns = []
    title_columns = []
    categorical_columns = []
    key_value_pairs = {}
    
    # Check for ID columns (more comprehensive)
    id_keywords = ['id', 'ref', 'number', '#', 'no', 'code', 'key', 'identifier']
    for col in df_analyzed.columns:
        col_lower = str(col).lower()
        col_data = df_analyzed[col].dropna()
        
        # ID column detection
        if any(keyword in col_lower for keyword in id_keywords):
            id_columns.append(str(col))
            # Check if it's sequential
            if len(col_data) > 0 and col_data.dtype in ['int64', 'float64']:
                try:
                    is_sequential = (col_data.diff().dropna().abs() == 1).all()
                    if is_sequential:
                        patterns.append(f"Sequential ID column: {col}")
                except:
                    pass
    
    if id_columns:
        patterns.append(f"ID columns detected: {', '.join(id_columns)}")
        key_value_pairs['id_columns'] = id_columns
    
    # Check for title/name/description columns
    title_keywords = ['title', 'name', 'description', 'requirement', 'summary', 'subject', 'label']
    for col in df_analyzed.columns:
        if any(keyword in str(col).lower() for keyword in title_keywords):
            title_columns.append(str(col))
    
    if title_columns:
        patterns.append(f"Title/Name columns detected: {', '.join(title_columns)}")
        key_value_pairs['title_columns'] = title_columns
    
    # Check for date/time columns
    detected_date_columns = []
    for col in df_analyzed.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'modified', 'updated', 'timestamp']):
            detected_date_columns.append(str(col))
    
    if detected_date_columns:
        patterns.append(f"Date columns detected: {', '.join(detected_date_columns)}")
        key_value_pairs['date_columns'] = detected_date_columns
    
    # Check for categorical columns (enhanced detection)
    for col in df_analyzed.columns:
        unique_ratio = df_analyzed[col].nunique() / len(df_analyzed)
        if unique_ratio < 0.1 and df_analyzed[col].nunique() > 1:
            categorical_columns.append(str(col))
    
    if categorical_columns:
        patterns.append(f"Categorical columns detected: {', '.join(categorical_columns[:5])}")
        key_value_pairs['categorical_columns'] = categorical_columns
    
    # Check for status/state columns
    status_keywords = ['status', 'state', 'stage', 'phase', 'priority', 'severity', 'type']
    status_columns = [col for col in df_analyzed.columns 
                     if any(keyword in str(col).lower() for keyword in status_keywords)]
    if status_columns:
        patterns.append(f"Status/State columns detected: {', '.join(status_columns)}")
        key_value_pairs['status_columns'] = status_columns
    
    # Detect potential relationships
    relationships = _detect_relationships(df_analyzed, id_columns, title_columns)
    if relationships:
        patterns.extend(relationships)
    
    # Data quality metrics
    total_cells = len(df_analyzed) * len(df_analyzed.columns)
    null_cells = df_analyzed.isna().sum().sum()
    data_completeness = round((total_cells - null_cells) / total_cells * 100, 2) if total_cells > 0 else 0
    
    # Calculate column-wise completeness
    column_completeness = {}
    for col in df_analyzed.columns:
        completeness = round((1 - df_analyzed[col].isna().sum() / len(df_analyzed)) * 100, 2)
        column_completeness[str(col)] = completeness
    
    # Detect data anomalies
    anomalies = _detect_anomalies(df_analyzed, column_info)
    
    return {
        "is_empty": False,
        "sheet_name": sheet_name,
        "total_rows": len(df_analyzed),
        "total_columns": len(df_analyzed.columns),
        "total_data_rows": len(df_analyzed),
        
        # Header information
        "header_detection": header_info,
        "headers": header_info["detected_headers"][:20],
        
        # Column details (enhanced)
        "columns": column_info[:20],
        "column_types_summary": {
            "numeric": len(numeric_columns),
            "text": len(text_columns),
            "date": len(date_columns),
            "boolean": len(boolean_columns)
        },
        
        # Patterns and insights (enhanced)
        "patterns": patterns,
        "key_columns": key_value_pairs,
        "has_id_column": len(id_columns) > 0,
        "has_title_column": len(title_columns) > 0,
        "has_date_column": len(detected_date_columns) > 0,
        "has_status_column": len(status_columns) > 0 if 'status_columns' in key_value_pairs else False,
        
        # Data quality (enhanced)
        "data_completeness_pct": data_completeness,
        "column_completeness": column_completeness,
        "null_cells": int(null_cells),
        "total_cells": int(total_cells),
        "data_quality_score": _calculate_quality_score(data_completeness, column_info),
        "anomalies": anomalies,
        
        # Sample data (enhanced with more context)
        "sample_rows": df_analyzed.head(3).to_dict('records') if len(df_analyzed) > 0 else [],
        "row_count_by_category": _get_category_counts(df_analyzed, categorical_columns),
        
        # Metadata summary for LLM
        "llm_summary": _generate_llm_summary(
            sheet_name, 
            len(df_analyzed), 
            header_info, 
            key_value_pairs, 
            patterns,
            data_completeness
        )
    }


def _detect_column_type_advanced(col_data: pd.Series, col_name: str) -> tuple:
    """
    Advanced column type detection with statistics.
    
    Returns:
        (type_name, statistics_dict)
    """
    stats = {}
    
    # Check for boolean
    unique_vals = col_data.unique()
    if len(unique_vals) <= 2:
        unique_lower = [str(v).lower() for v in unique_vals]
        if all(v in ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n'] for v in unique_lower):
            stats['values'] = list(unique_vals)
            return "boolean", stats
    
    # Check for numeric types
    if pd.api.types.is_numeric_dtype(col_data):
        if pd.api.types.is_integer_dtype(col_data):
            stats['min'] = int(col_data.min())
            stats['max'] = int(col_data.max())
            stats['mean'] = round(float(col_data.mean()), 2)
            stats['median'] = int(col_data.median())
            return "integer", stats
        else:
            stats['min'] = round(float(col_data.min()), 2)
            stats['max'] = round(float(col_data.max()), 2)
            stats['mean'] = round(float(col_data.mean()), 2)
            stats['median'] = round(float(col_data.median()), 2)
            return "float", stats
    
    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(col_data):
        stats['earliest'] = str(col_data.min())
        stats['latest'] = str(col_data.max())
        stats['range_days'] = (col_data.max() - col_data.min()).days
        return "datetime", stats
    
    # Check if string could be datetime
    if col_data.dtype == 'object':
        try:
            parsed = pd.to_datetime(col_data, errors='coerce')
            if parsed.notna().sum() / len(col_data) > 0.8:  # 80% parseable
                stats['earliest'] = str(parsed.min())
                stats['latest'] = str(parsed.max())
                return "date", stats
        except:
            pass
    
    # Text/String type
    if col_data.dtype == 'object':
        # Calculate text statistics
        lengths = col_data.astype(str).str.len()
        stats['avg_length'] = round(float(lengths.mean()), 2)
        stats['min_length'] = int(lengths.min())
        stats['max_length'] = int(lengths.max())
        
        # Check if it's URL, email, etc.
        sample = col_data.astype(str).str.lower()
        if sample.str.contains('@').sum() / len(sample) > 0.8:
            return "email", stats
        if sample.str.contains('http').sum() / len(sample) > 0.8:
            return "url", stats
        
        return "text", stats
    
    return "mixed", stats


def _detect_relationships(df: pd.DataFrame, id_columns: list, title_columns: list) -> list:
    """Detect potential relationships between columns."""
    relationships = []
    
    # Check for foreign key patterns
    for col in df.columns:
        col_lower = str(col).lower()
        if '_id' in col_lower or col_lower.endswith('id'):
            # Potential foreign key
            relationships.append(f"Potential foreign key: {col}")
    
    # Check for hierarchical structures
    if any('parent' in str(col).lower() for col in df.columns):
        relationships.append("Hierarchical structure detected (parent-child relationship)")
    
    # Check for many-to-one relationships (duplicate IDs)
    for id_col in id_columns:
        if df[id_col].duplicated().any():
            relationships.append(f"One-to-many relationship detected in: {id_col}")
    
    return relationships


def _detect_anomalies(df: pd.DataFrame, column_info: list) -> list:
    """Detect data anomalies and quality issues."""
    anomalies = []
    
    for col_meta in column_info:
        col_name = col_meta['name']
        
        # High null percentage
        if col_meta['null_percentage'] > 50:
            anomalies.append(f"High null rate in '{col_name}': {col_meta['null_percentage']}%")
        
        # All values are unique (might indicate no actual data or IDs)
        if col_meta['uniqueness_ratio'] == 1.0 and len(df) > 10:
            anomalies.append(f"All values unique in '{col_name}' - possibly ID or no duplicate check")
        
        # Very low uniqueness in supposedly unique column
        if 'id' in col_name.lower() and col_meta['uniqueness_ratio'] < 0.5:
            anomalies.append(f"Low uniqueness in ID column '{col_name}': {col_meta['uniqueness_ratio']}")
    
    return anomalies


def _calculate_quality_score(completeness: float, column_info: list) -> float:
    """Calculate overall data quality score (0-100)."""
    score = completeness * 0.5  # 50% weight on completeness
    
    # Add points for column type consistency
    type_consistency = sum(1 for col in column_info if col['type'] != 'mixed')
    if len(column_info) > 0:
        score += (type_consistency / len(column_info)) * 30  # 30% weight
    
    # Add points for low null rates
    low_null_columns = sum(1 for col in column_info if col['null_percentage'] < 10)
    if len(column_info) > 0:
        score += (low_null_columns / len(column_info)) * 20  # 20% weight
    
    return round(score, 2)


def _get_category_counts(df: pd.DataFrame, categorical_columns: list) -> dict:
    """Get counts for categorical columns."""
    counts = {}
    for col in categorical_columns[:3]:  # First 3 categorical columns
        if col in df.columns:
            value_counts = df[col].value_counts().head(5).to_dict()
            counts[col] = {str(k): int(v) for k, v in value_counts.items()}
    return counts


def _generate_llm_summary(sheet_name: str, row_count: int, header_info: dict, 
                          key_columns: dict, patterns: list, completeness: float) -> str:
    """Generate a human-readable summary for LLM context."""
    summary_parts = [
        f"Sheet '{sheet_name}' contains {row_count} rows of data."
    ]
    
    # Header info
    if header_info['has_header']:
        summary_parts.append(f"Headers detected with {header_info['header_confidence']*100:.0f}% confidence.")
    else:
        summary_parts.append("No headers detected - data starts from first row.")
    
    # Key columns
    if key_columns.get('id_columns'):
        summary_parts.append(f"ID columns: {', '.join(key_columns['id_columns'])}.")
    if key_columns.get('title_columns'):
        summary_parts.append(f"Descriptive columns: {', '.join(key_columns['title_columns'])}.")
    
    # Data quality
    if completeness >= 90:
        summary_parts.append(f"Data quality is excellent ({completeness}% complete).")
    elif completeness >= 70:
        summary_parts.append(f"Data quality is good ({completeness}% complete).")
    else:
        summary_parts.append(f"Data quality needs attention ({completeness}% complete with missing values).")
    
    return " ".join(summary_parts)
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

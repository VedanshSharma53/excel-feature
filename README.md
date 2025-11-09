# Excel Feature API

FastAPI service for Excel file processing with **intelligent header detection** and rich metadata extraction.

## 🎯 Key Features

### ✨ Intelligent Header Detection
- **Automatically detects** if first row is a header or data
- **Multi-strategy detection** with confidence scoring (6 different checks)
- **Handles edge cases**: Unnamed columns, numeric headers, mixed data
- **No assumptions**: Works with files that have headers OR start directly with data

### 📊 Rich Metadata Extraction
- **Column type detection**: Numeric, text, datetime, mixed
- **Pattern recognition**: Automatically identifies ID, title, date, and categorical columns
- **Data quality metrics**: Completeness percentage, null counts
- **Sample data**: First rows for LLM context understanding
- **Structure analysis**: Deep insights about sheet organization

## Files

- `xlsx_converter.py` - XLSX/XLS to text API with chunking & metadata
- `excel_pinecone_processor.py` - Excel to Pinecone Q&A system
- `main.py` - CLI interface
- `requirements.txt` - Dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### XLSX Converter API

```python
from xlsx_converter import convert_xlsx_service

result = await convert_xlsx_service(file)
```

**Returns:**
```json
{
  "text": "extracted content",
  "chunks": [
    {
      "text": "chunk content",
      "metadata": {
        "sheet_name": "Sheet1",
        "header_detection": {
          "has_header": true,
          "header_confidence": 0.9,
          "detected_headers": ["ID", "Title", "Priority"],
          "first_data_row": 1,
          "reasons": ["pandas detected meaningful column names"]
        },
        "headers": ["ID", "Title", "Priority", "Status"],
        "total_rows": 100,
        "total_columns": 5,
        "columns": [
          {
            "name": "ID",
            "type": "text",
            "null_count": 0,
            "unique_count": 100,
            "sample_values": ["REQ-001", "REQ-002", "REQ-003"]
          }
        ],
        "patterns": [
          "ID columns detected: ID",
          "Title/Name columns detected: Title"
        ],
        "data_completeness_pct": 95.5,
        "sample_rows": [
          {"ID": "REQ-001", "Title": "User Login"}
        ],
        "chunk_index": 0
      }
    }
  ],
  "success": true
}
```

### Header Detection Examples

**Example 1: File with headers**
```
| ID      | Title          | Priority |
|---------|----------------|----------|
| REQ-001 | User Login     | High     |
```
Result: `has_header: true`, `header_confidence: 0.9`

**Example 2: File without headers**
```
| REQ-001 | User Login | High |
| REQ-002 | Data Export | Low |
```
Result: `has_header: false`, uses `["Column_0", "Column_1", "Column_2"]`

**Example 3: First row IS header**
```
| ID | Title | Priority |  (← This is data, looks like header)
| REQ-001 | User Login | High |
```
Result: Detects first row as header with `header_confidence: 0.8`

### Excel Pinecone Q&A

```bash
python main.py
```

## Dependencies

- FastAPI, pandas, Apache Tika
- Pinecone, sentence-transformers

## 📚 Documentation

- **[HEADER_DETECTION.md](HEADER_DETECTION.md)** - Detailed explanation of intelligent header detection
- **[test_header_detection.py](test_header_detection.py)** - Test suite with examples

## 🧪 Testing Header Detection

Run the test suite to see header detection in action:

```bash
python test_header_detection.py
```

This demonstrates:
- ✅ Detection with proper headers
- ✅ Detection without headers (raw data)
- ✅ First row as header detection
- ✅ Mixed data types handling
- ✅ Missing values handling

## 🎯 Benefits for LLM Processing

The rich metadata enables your LLM to:

1. **Understand structure** before extraction
2. **Identify relationships** between columns
3. **Extract values** with correct context
4. **Handle missing data** intelligently
5. **Recognize patterns** (IDs, categories, dates)
6. **Quality assessment** (completeness, null counts)
7. **Type-aware processing** (numeric vs text vs dates)

No more "assuming first row is always a header"! 🚀
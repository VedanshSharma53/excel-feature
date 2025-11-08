# Excel Feature API

FastAPI service for Excel file processing with metadata extraction.

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
        "headers": ["col1", "col2"],
        "total_rows": 100,
        "total_columns": 5,
        "chunk_index": 0
      }
    }
  ],
  "success": true
}
```

### Excel Pinecone Q&A

```bash
python main.py
```

## Dependencies

- FastAPI, pandas, Apache Tika
- Pinecone, sentence-transformers

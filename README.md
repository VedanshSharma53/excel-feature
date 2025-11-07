# Excel to Pinecone - Natural Language Q&A

Process Excel files with multiple sheets and answer natural language questions by automatically detecting the right sheet and returning complete content.

## Features

- **Natural Language Questions**: "what are integration requirements" → Get ALL content from that sheet
- **Smart Sheet Detection**: AI finds the best matching sheet (50-85% confidence)
- **Local Embeddings**: Uses sentence-transformers (free, fast, works offline)
- **Heavy Model**: all-mpnet-base-v2 (768 dimensions) for high accuracy

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- pandas, openpyxl (Excel processing)
- pinecone-client (vector database)
- sentence-transformers, torch (local embeddings)
- numpy

## Setup

**Set API Key:**
```bash
# Windows CMD
set PINECONE_API_KEY=your-key

# Windows PowerShell
$env:PINECONE_API_KEY="your-key"

# Linux/Mac
export PINECONE_API_KEY=your-key
```

Get Pinecone key from: https://app.pinecone.io/

## Usage

### Option 1: Interactive CLI (Easiest)

```bash
python main.py
```

Then choose:
1. Process new Excel file
2. Ask questions about existing data

### Option 2: Programmatic

```python
from excel_pinecone_processor import ExcelPineconeProcessor
import os

# Initialize
processor = ExcelPineconeProcessor(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    index_name="excel-mpnet-index",
    embedding_model="all-mpnet-base-v2",
    dimension=768
)

# Process Excel (one-time)
processor.process_excel_to_pinecone("my_file.xlsx")

# Ask questions
result = processor.answer_question_with_sheet(
    query="what are integration requirements",
    score_threshold=0.3
)

print(f"Matched: {result['matched_sheet']}")
print(f"Confidence: {result['score']:.2%}")
print(f"Chunks: {len(result['content'])}")

# Access content
for chunk in result['content']:
    print(chunk['content'])
```

## Example Questions

- "what are integration requirements"
- "show me security requirements"
- "what are the functional requirements"
- "tell me about operational requirements"

## Model Details

**Current Model: all-mpnet-base-v2**
- **Dimensions**: 768
- **Size**: ~420MB (downloads on first run)
- **Quality**: High (50-85% confidence scores)
- **Speed**: Fast (~1-2 seconds per query)
- **Cost**: FREE (runs locally)
- **Offline**: Yes (after first download)

**Alternative Models:**
```python
# Faster, smaller (384 dim)
embedding_model="all-MiniLM-L6-v2"
dimension=384

# Multilingual support (384 dim)
embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
dimension=384
```

## How It Works

1. **Process Excel**: Extracts all sheets, creates chunks, generates embeddings
2. **Store in Pinecone**: Saves with metadata (sheet_name, row_numbers, content)
3. **Question Asked**: Creates embedding for the question
4. **Match Sheet**: Compares question embedding with sheet names using cosine similarity
5. **Return Content**: If confidence ≥ threshold, returns ALL chunks from matched sheet

## Configuration

**Adjust Confidence Threshold:**
```python
result = processor.answer_question_with_sheet(
    query="your question",
    score_threshold=0.25  # Lower = more permissive (default: 0.3)
)
```

**Custom Chunking:**
```python
processor.process_excel_to_pinecone(
    excel_file_path="file.xlsx",
    chunk_size=10  # Rows per chunk (default: 5)
)
```

**Different Index:**
```python
# Create separate index for different projects
processor = ExcelPineconeProcessor(
    pinecone_api_key=key,
    index_name="project-xyz-index",  # Custom name
    embedding_model="all-mpnet-base-v2",
    dimension=768
)
```

## Real Example

**Excel file with 12 sheets:**
- 'Scope of Fields'
- '1. Functional requirements'
- '3. Integration Requirements'
- '7. Security'
- etc.

**Query:** "what are integration requirements"

**Result:**
```
✅ Matched Sheet: '3. Integration Requirements'
✅ Confidence Score: 84.11%
✅ Found 11 chunks

Chunk 1 (Rows 0-5):
3: ABC | N/A.2: Provider Responses
3: Ref. # | INTEGRATION REQUIREMENTS: ...

[All 11 chunks from that sheet returned]
```

## Performance

- **Processing**: ~2-3 seconds per sheet
- **Query**: ~1-2 seconds
- **Accuracy**: 80-85% for exact matches, 50-60% for partial
- **Cost**: $0 (local embeddings, only Pinecone storage)

## Metadata in Each Chunk

Every chunk stored in Pinecone includes rich metadata:

### Sheet/Tab Information
- `sheet_name` / `tab_name` - Name of the Excel sheet/tab
- `start_row` - Starting row number (0-indexed)
- `end_row` - Ending row number
- `chunk_size` - Number of rows in this chunk

### Column/Header Information
- `headers` - Comma-separated list of column headers
- `column_names` - Array of column names
- `total_columns` - Total number of columns
- `has_id` - Boolean: contains ID/Ref column
- `has_title` - Boolean: contains Title/Requirement column

### Content
- `content` - First 1000 chars of chunk content
- `full_content` - Complete chunk text
- `first_row_preview` - Preview of first row data

### Extracted Column Values
- `col_id` / `col_ref_num` - ID/Reference numbers
- `col_title` / `col_requirement` - Titles/Requirements
- `col_category` / `col_type` - Categories/Types

**Example metadata:**
```python
{
    "sheet_name": "3. Integration Requirements",
    "tab_name": "3. Integration Requirements",
    "start_row": 0,
    "end_row": 5,
    "headers": "Ref. #, Requirement, Description, Importance, Author",
    "column_names": ["Ref. #", "Requirement", "Description", ...],
    "total_columns": 10,
    "has_id": True,
    "has_title": True,
    "col_ref_num": "3.1.1, 3.1.2, 3.1.3",
    "col_requirement": "WhatsApp integration, API support, ...",
    "first_row_preview": "{'Ref. #': '3.1.1', 'Requirement': 'WhatsApp'}",
    "content": "3: 3.1.1 | INTEGRATION REQUIREMENTS: WhatsApp...",
    "full_content": "..."
}
```

### Using Metadata for Filtering
```python
# Search with metadata filter
results = processor.index.query(
    vector=query_embedding,
    top_k=10,
    filter={
        "sheet_name": {"$eq": "3. Integration Requirements"},
        "has_id": {"$eq": True}
    },
    include_metadata=True
)
```

## Files

- `excel_pinecone_processor.py` - Core processor class
- `main.py` - Interactive CLI
- `example_usage.py` - Code examples
- `requirements.txt` - Dependencies
- `.env` - API keys (excluded from git)

## Troubleshooting

**"PINECONE_API_KEY not set"**
```bash
set PINECONE_API_KEY=your-actual-key
```

**"Vector dimension mismatch"**
- Delete old index or use different index name
- Ensure dimension matches model (768 for all-mpnet-base-v2)

**"No results found"**
- Wait 10-20 seconds after uploading (Pinecone indexing time)
- Check sheet names with `processor.list_all_sheets()`
- Lower score_threshold to 0.2 or 0.15

**"Model download slow"**
- Normal for first run (~420MB)
- Subsequent runs use cached model

## API Reference

```python
# Process Excel
processor.process_excel_to_pinecone(excel_file_path, chunk_size=5)

# Ask question (recommended)
processor.answer_question_with_sheet(query, score_threshold=0.3)

# Search specific sheet
processor.search_in_sheet(query, sheet_name, return_all_from_sheet=True)

# Find best matching sheet
processor.find_sheet_for_query(query, top_k=3)

# List all sheets
processor.list_all_sheets()

# Delete sheet data
processor.delete_sheet(sheet_name)
```

## License

MIT

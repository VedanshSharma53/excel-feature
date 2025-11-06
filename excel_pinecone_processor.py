"""
Excel to Pinecone Processor
Automatically extracts all sheets from Excel, creates embeddings, and stores in Pinecone
with sheet-based metadata filtering for targeted search.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time
import hashlib
import os


class ExcelPineconeProcessor:
    """
    Process Excel files with multiple sheets and store in Pinecone with metadata.
    Supports sheet-based filtering and retrieval.
    """
    
    def __init__(
        self,
        pinecone_api_key: str,
        index_name: str = "excel-sheets-index",
        embedding_model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,  # all-MiniLM-L6-v2 dimension
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """
        Initialize the processor with API keys and configuration.
        
        Args:
            pinecone_api_key: Pinecone API key
            index_name: Name for the Pinecone index
            embedding_model: Sentence-transformer model name
                - "all-MiniLM-L6-v2": Fast, 384 dim (recommended)
                - "all-mpnet-base-v2": Better quality, 768 dim
                - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual, 384 dim
            dimension: Embedding dimension (384 for MiniLM, 768 for mpnet)
            cloud: Cloud provider for Pinecone serverless
            region: Region for Pinecone serverless
        """
        # Initialize sentence-transformer model (runs locally, no API needed!)
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"Model loaded successfully! Dimension: {dimension}")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.dimension = dimension
        
        # Create or connect to index
        self._setup_index(cloud, region)
        
    def _setup_index(self, cloud: str, region: str):
        """Create Pinecone index if it doesn't exist."""
        try:
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region)
                )
                # Wait for index to be ready
                time.sleep(1)
            
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            print(f"Error setting up index: {e}")
            raise
    
    def extract_sheet_names(self, excel_file_path: str) -> List[str]:
        """
        Extract all sheet names from an Excel file.
        
        Args:
            excel_file_path: Path to the Excel file
            
        Returns:
            List of sheet names
        """
        try:
            excel_file = pd.ExcelFile(excel_file_path)
            sheet_names = excel_file.sheet_names
            print(f"Found {len(sheet_names)} sheets: {sheet_names}")
            return sheet_names
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding using local sentence-transformer model.
        Fast and free - runs on your machine!
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Clean text
            text = str(text).strip()
            if not text:
                text = "empty"
            
            # Create embedding locally (no API call!)
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"Error creating embedding: {e}")
            raise
    
    def process_sheet(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        chunk_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process a single sheet and prepare data for Pinecone.
        
        Args:
            df: DataFrame containing sheet data
            sheet_name: Name of the sheet
            chunk_size: Number of rows to combine into one chunk
            
        Returns:
            List of dictionaries with embeddings and metadata
        """
        chunks = []
        
        # Drop completely empty rows
        df = df.dropna(how='all')
        
        # Process in chunks
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            
            # Convert chunk to text
            chunk_text = self._dataframe_to_text(chunk_df)
            
            if not chunk_text.strip():
                continue
            
            # Create unique ID
            chunk_id = self._generate_id(sheet_name, i)
            
            # Create embedding
            embedding = self.create_embedding(chunk_text)
            
            # Prepare metadata
            metadata = {
                "sheet_name": sheet_name,
                "start_row": i,
                "end_row": min(i + chunk_size, len(df)),
                "content": chunk_text[:1000],  # Store first 1000 chars in metadata
                "full_content": chunk_text,  # Store full content
                "chunk_size": len(chunk_df)
            }
            
            chunks.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": metadata
            })
        
        return chunks
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame rows to text format.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            Text representation
        """
        text_parts = []
        
        for idx, row in df.iterrows():
            # Create row text with column names
            row_text = []
            for col in df.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    row_text.append(f"{col}: {value}")
            
            if row_text:
                text_parts.append(" | ".join(row_text))
        
        return "\n".join(text_parts)
    
    def _generate_id(self, sheet_name: str, row_number: int) -> str:
        """Generate unique ID for a chunk."""
        unique_string = f"{sheet_name}_{row_number}_{time.time()}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def process_excel_to_pinecone(
        self,
        excel_file_path: str,
        chunk_size: int = 5,
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Process entire Excel file and upload to Pinecone.
        
        Args:
            excel_file_path: Path to Excel file
            chunk_size: Number of rows per chunk
            batch_size: Batch size for Pinecone upsert
            
        Returns:
            Dictionary with statistics
        """
        print(f"\n{'='*60}")
        print(f"Processing Excel file: {excel_file_path}")
        print(f"{'='*60}\n")
        
        # Extract sheet names
        sheet_names = self.extract_sheet_names(excel_file_path)
        
        stats = {"total_chunks": 0, "sheets_processed": 0}
        
        # Process each sheet
        for sheet_name in sheet_names:
            print(f"\nProcessing sheet: '{sheet_name}'")
            
            try:
                # Read sheet
                df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
                print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
                
                # Process sheet
                chunks = self.process_sheet(df, sheet_name, chunk_size)
                print(f"  Created {len(chunks)} chunks")
                
                # Upload to Pinecone in batches
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                    print(f"  Uploaded batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
                stats["total_chunks"] += len(chunks)
                stats["sheets_processed"] += 1
                
                # Small delay between sheets
                time.sleep(0.1)
                
            except Exception as e:
                print(f"  Error processing sheet '{sheet_name}': {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Sheets processed: {stats['sheets_processed']}/{len(sheet_names)}")
        print(f"Total chunks uploaded: {stats['total_chunks']}")
        print(f"{'='*60}\n")
        
        return stats
    
    def search_in_sheet(
        self,
        query: str,
        sheet_name: Optional[str] = None,
        return_all_from_sheet: bool = False,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for content with optional sheet filtering.
        
        Args:
            query: Search query
            sheet_name: Filter by specific sheet name (optional)
            return_all_from_sheet: If True and sheet_name provided, returns all content from that sheet
            top_k: Number of results to return (ignored if return_all_from_sheet=True)
            
        Returns:
            List of search results with metadata
        """
        try:
            # If requesting all content from a specific sheet
            if return_all_from_sheet and sheet_name:
                return self._get_all_from_sheet(sheet_name)
            
            # Create query embedding
            query_embedding = self.create_embedding(query)
            
            # Build filter
            filter_dict = None
            if sheet_name:
                filter_dict = {"sheet_name": {"$eq": sheet_name}}
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "score": match.score,
                    "sheet_name": match.metadata.get("sheet_name"),
                    "rows": f"{match.metadata.get('start_row')}-{match.metadata.get('end_row')}",
                    "content": match.metadata.get("full_content", match.metadata.get("content")),
                    "metadata": match.metadata
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching: {e}")
            raise
    
    def _get_all_from_sheet(self, sheet_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve all content from a specific sheet.
        
        Args:
            sheet_name: Name of the sheet
            
        Returns:
            List of all chunks from the sheet
        """
        print(f"\nRetrieving all content from sheet: '{sheet_name}'")
        
        # Create a dummy query vector
        dummy_embedding = [0.0] * self.dimension
        
        # Query with high top_k and sheet filter
        results = self.index.query(
            vector=dummy_embedding,
            top_k=10000,  # High number to get all results
            include_metadata=True,
            filter={"sheet_name": {"$eq": sheet_name}}
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "sheet_name": match.metadata.get("sheet_name"),
                "rows": f"{match.metadata.get('start_row')}-{match.metadata.get('end_row')}",
                "content": match.metadata.get("full_content", match.metadata.get("content")),
                "metadata": match.metadata
            })
        
        # Sort by row number
        formatted_results.sort(key=lambda x: x["metadata"].get("start_row", 0))
        
        print(f"Retrieved {len(formatted_results)} chunks from '{sheet_name}'")
        
        return formatted_results
    
    def list_all_sheets(self) -> List[str]:
        """
        Get list of all unique sheet names in the index.
        
        Returns:
            List of sheet names
        """
        # Query with dummy vector to get sample results
        dummy_embedding = [0.0] * self.dimension
        results = self.index.query(
            vector=dummy_embedding,
            top_k=10000,
            include_metadata=True
        )
        
        # Extract unique sheet names
        sheet_names = set()
        for match in results.matches:
            if "sheet_name" in match.metadata:
                sheet_names.add(match.metadata["sheet_name"])
        
        return sorted(list(sheet_names))

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def find_sheet_for_query(self, query: str, candidate_sheets: Optional[List[str]] = None, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Given a natural language query, find the most likely sheet(s) that the user intends.

        Strategy:
        - Embed the query with the local model
        - Embed each sheet name (or use cached embeddings) and compute cosine similarity
        - Return top_k candidate sheets with scores

        Args:
            query: Natural language question (e.g. "what are integration requirements")
            candidate_sheets: Optional list of sheet names to consider. If None, uses all sheets found in the index.
            top_k: Number of top candidate sheets to return

        Returns:
            List of dicts: [{"sheet_name": str, "score": float}, ...]
        """
        # Get candidate sheet names
        if candidate_sheets is None:
            candidate_sheets = self.list_all_sheets()

        if not candidate_sheets:
            return []

        # Embed query and sheet names
        q_emb = self.embedding_model.encode(str(query), convert_to_tensor=False)

        sheet_scores = []
        for sheet in candidate_sheets:
            # Simple embedding for sheet name (short text)
            s_emb = self.embedding_model.encode(str(sheet), convert_to_tensor=False)
            score = self._cosine_similarity(q_emb, s_emb)
            sheet_scores.append({"sheet_name": sheet, "score": score})

        # Sort by score desc and return top_k
        sheet_scores.sort(key=lambda x: x["score"], reverse=True)
        return sheet_scores[:top_k]

    def answer_question_with_sheet(self, query: str, score_threshold: float = 0.2) -> Dict[str, Any]:
        """
        High-level helper to answer natural-language questions that ask for content from a specific sheet.

        Behavior:
        - Detects the best matching sheet for the query
        - If the best score >= score_threshold, returns ALL content from that sheet (sheet name and chunks)
        - Otherwise, returns an empty candidates list and suggests sheet names

        Returns a dict with keys:
            {"query": ..., "matched_sheet": ..., "score": ..., "content": [...chunks...]}
        """
        candidates = self.find_sheet_for_query(query, top_k=3)
        if not candidates:
            return {"query": query, "matched_sheet": None, "score": 0.0, "content": [], "candidates": []}

        best = candidates[0]
        result = {"query": query, "matched_sheet": best["sheet_name"], "score": best["score"], "content": [], "candidates": candidates}

        # If confidence is high enough, return all content from the matched sheet
        if best["score"] >= score_threshold:
            all_chunks = self._get_all_from_sheet(best["sheet_name"])
            result["content"] = all_chunks
        else:
            # Low confidence - don't assume. Return candidates so user can pick.
            result["content"] = []

        return result
    
    def delete_sheet(self, sheet_name: str):
        """
        Delete all vectors from a specific sheet.
        
        Args:
            sheet_name: Name of the sheet to delete
        """
        print(f"Deleting all data from sheet: '{sheet_name}'")
        
        # Note: Pinecone doesn't support direct deletion by metadata filter
        # We need to fetch IDs first, then delete
        dummy_embedding = [0.0] * self.dimension
        results = self.index.query(
            vector=dummy_embedding,
            top_k=10000,
            include_metadata=True,
            filter={"sheet_name": {"$eq": sheet_name}}
        )
        
        # Extract IDs
        ids_to_delete = [match.id for match in results.matches]
        
        if ids_to_delete:
            self.index.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} vectors from sheet '{sheet_name}'")
        else:
            print(f"No vectors found for sheet '{sheet_name}'")


def main():
    """
    Example usage of the ExcelPineconeProcessor class.
    """
    # Configuration - Replace with your actual API keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
    EXCEL_FILE = "my_file.xlsx"  # Replace with your Excel file path
    
    # Initialize processor (no OpenAI needed!)
    processor = ExcelPineconeProcessor(
        pinecone_api_key=PINECONE_API_KEY,
        index_name="excel-sheets-index",
        embedding_model="all-MiniLM-L6-v2",  # Fast local model
        dimension=384
    )
    
    # Process Excel file
    print("\n=== STEP 1: Processing Excel File ===")
    stats = processor.process_excel_to_pinecone(
        excel_file_path=EXCEL_FILE,
        chunk_size=5  # Combine 5 rows per chunk
    )
    
    # List all sheets in index
    print("\n=== STEP 2: List All Sheets ===")
    sheets = processor.list_all_sheets()
    print(f"Sheets in index: {sheets}")
    
    # Search with semantic similarity (within specific sheet)
    print("\n=== STEP 3: Semantic Search within Sheet ===")
    results = processor.search_in_sheet(
        query="authentication requirements",
        sheet_name="Functional Requirements",
        return_all_from_sheet=False,
        top_k=5
    )
    
    print(f"\nTop {len(results)} semantic matches:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Sheet: {result['sheet_name']}")
        print(f"   Rows: {result['rows']}")
        print(f"   Content: {result['content'][:200]}...")
    
    # Get ALL content from a specific sheet
    print("\n=== STEP 4: Get All Content from Sheet ===")
    all_results = processor.search_in_sheet(
        query="",  # Query not used when return_all_from_sheet=True
        sheet_name="Functional Requirements",
        return_all_from_sheet=True
    )
    
    print(f"\nAll content from 'Functional Requirements' sheet:")
    print(f"Total chunks: {len(all_results)}")
    for i, result in enumerate(all_results[:3], 1):  # Show first 3
        print(f"\n{i}. Sheet: {result['sheet_name']}")
        print(f"   Rows: {result['rows']}")
        print(f"   Content: {result['content'][:200]}...")


if __name__ == "__main__":
    main()

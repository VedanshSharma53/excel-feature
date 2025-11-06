"""
Example Usage Script for Excel to Pinecone Processing
Demonstrates various use cases and search scenarios
"""

from excel_pinecone_processor import ExcelPineconeProcessor
import os
import time


def example_0_list_sheets():
    """Example 0: List all available sheets in the index"""
    print("\n" + "="*70)
    print("EXAMPLE 0: List All Available Sheets")
    print("="*70)
    
    processor = ExcelPineconeProcessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="excel-demo-mpnet",  # Different index for mpnet model
        embedding_model="all-mpnet-base-v2",  # Better quality model
        dimension=768
    )
    
    sheets = processor.list_all_sheets()
    print(f"\nFound {len(sheets)} sheets in index:")
    for i, sheet in enumerate(sheets, 1):
        print(f"  {i}. '{sheet}'")
    
    return sheets


def example_0a_answer_natural_question():
    """Example 0a: Answer natural language question with automatic sheet detection"""
    print("\n" + "="*70)
    print("EXAMPLE 0a: Natural Language Question Answering")
    print("="*70)
    
    processor = ExcelPineconeProcessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="excel-demo-mpnet",  # Different index for mpnet model
        embedding_model="all-mpnet-base-v2",  # Better quality model
        dimension=768
    )
    
    # Ask natural language questions
    questions = [
        "what are integration requirements",
        "show me security requirements",
        "what are the functional requirements"
    ]
    
    for question in questions:
        print(f"\n{'─'*70}")
        print(f"Question: '{question}'")
        print(f"{'─'*70}")
        
        # Use the new helper method
        response = processor.answer_question_with_sheet(
            query=question,
            score_threshold=0.15  # Lower threshold for better recall
        )
        
        print(f"\n✓ Matched Sheet: '{response['matched_sheet']}'")
        print(f"✓ Confidence Score: {response['score']:.4f}")
        
        # Show top 3 candidates
        print(f"\n  Top 3 Sheet Candidates:")
        for i, candidate in enumerate(response.get('candidates', [])[:3], 1):
            print(f"    {i}. '{candidate['sheet_name']}' (score: {candidate['score']:.4f})")
        
        # Show content if found
        if response['content']:
            print(f"\n✓ Found {len(response['content'])} chunks from '{response['matched_sheet']}'")
            print(f"\n  First 3 chunks preview:")
            for i, chunk in enumerate(response['content'][:3], 1):
                print(f"\n    Chunk {i} (Rows {chunk['rows']}):")
                print(f"    {chunk['content'][:200]}...")
        else:
            print(f"\n⚠ Low confidence - no content returned")
            print(f"  Please select from candidates above")
        
        time.sleep(0.5)  # Brief pause between questions


def example_1_basic_processing():
    """Example 1: Basic Excel file processing"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Excel File Processing")
    print("="*70)

    # Initialize processor (no OpenAI needed!)
    processor = ExcelPineconeProcessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="excel-demo-index"
    )
    
    # Process Excel file
    stats = processor.process_excel_to_pinecone("example.xlsx")
    print(f"\nProcessing complete: {stats}")


def example_2_search_within_sheet():
    """Example 2: Search within a specific sheet"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Search within Specific Sheet")
    print("="*70)
    
    processor = ExcelPineconeProcessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="excel-demo-index"
    )
    
    # Wait a bit for Pinecone to finish indexing (if data was just uploaded)
    import time
    print("Waiting for Pinecone to index data...")
    time.sleep(3)
    
    # Search for authentication-related requirements
    # NOTE: Use exact sheet name from your Excel file!
    results = processor.search_in_sheet(
        query="authentication and login requirements",
        sheet_name="1. Functional requirements",  # Exact name from your file
        return_all_from_sheet=False,
        top_k=5
    )
    
    print(f"\nFound {len(results)} relevant results:")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Relevance Score: {result['score']:.4f}")
        print(f"Sheet: {result['sheet_name']}")
        print(f"Rows: {result['rows']}")
        print(f"Content:\n{result['content'][:300]}...")


def example_3_get_all_from_sheet():
    """Example 3: Get ALL content from a specific sheet"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Get ALL Content from Specific Sheet")
    print("="*70)
    
    processor = ExcelPineconeProcessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="excel-demo-index"
    )
    
    # Get everything from Functional Requirements sheet
    # NOTE: Use exact sheet name from your Excel file!
    all_content = processor.search_in_sheet(
        query="",  # Query not used
        sheet_name="1. Functional requirements",  # Exact name from your file
        return_all_from_sheet=True
    )
    
    print(f"\nRetrieved ALL content from '1. Functional requirements' sheet:")
    print(f"Total chunks: {len(all_content)}")
    print(f"\nFirst 3 chunks:")
    
    for i, result in enumerate(all_content[:3], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Rows: {result['rows']}")
        print(f"Content:\n{result['content'][:200]}...")


def example_4_natural_language_query():
    """Example 4: Natural language query understanding"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Natural Language Query")
    print("="*70)
    
    processor = ExcelPineconeProcessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="excel-demo-index"
    )
    
    # Understand natural language queries
    queries = [
        "show me all requirements from the Functional Requirements tab",
        "what are the security requirements?",
        "find all user interface specifications"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Extract sheet name if mentioned (simple logic - can be enhanced)
        sheet_name = None
        return_all = False
        
        if "all" in query.lower() and "functional requirements" in query.lower():
            sheet_name = "Functional Requirements"
            return_all = True
        elif "security requirements" in query.lower():
            sheet_name = "Security Requirements"
        
        results = processor.search_in_sheet(
            query=query,
            sheet_name=sheet_name,
            return_all_from_sheet=return_all,
            top_k=3
        )
        
        print(f"Found {len(results)} results")
        if results:
            print(f"First result: {results[0]['content'][:150]}...")


def example_5_list_and_search_all_sheets():
    """Example 5: List all sheets and search across them"""
    print("\n" + "="*70)
    print("EXAMPLE 5: List All Sheets and Cross-Sheet Search")
    print("="*70)
    
    processor = ExcelPineconeProcessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="excel-demo-index"
    )
    
    # List all available sheets
    sheets = processor.list_all_sheets()
    print(f"\nAvailable sheets in index: {sheets}")
    
    # Search across all sheets (no sheet filter)
    print("\nSearching across ALL sheets for 'user permissions':")
    results = processor.search_in_sheet(
        query="user permissions and access control",
        sheet_name=None,  # Search all sheets
        top_k=5
    )
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['sheet_name']}] Rows {result['rows']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   {result['content'][:150]}...")


def example_6_export_sheet_content():
    """Example 6: Export all content from a sheet to text file"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Export Sheet Content to File")
    print("="*70)
    
    processor = ExcelPineconeProcessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="excel-demo-index"
    )
    
    # Get all content from a sheet
    sheet_name = "Functional Requirements"
    all_content = processor.search_in_sheet(
        query="",
        sheet_name=sheet_name,
        return_all_from_sheet=True
    )
    
    # Export to text file
    output_file = f"{sheet_name.replace(' ', '_')}_export.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Export from Sheet: {sheet_name}\n")
        f.write(f"{'='*70}\n\n")
        
        for i, result in enumerate(all_content, 1):
            f.write(f"\nChunk {i} (Rows {result['rows']})\n")
            f.write(f"{'-'*70}\n")
            f.write(f"{result['content']}\n")
    
    print(f"\nExported {len(all_content)} chunks to '{output_file}'")


def example_7_compare_sheets():
    """Example 7: Compare content between two sheets"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Compare Content Between Sheets")
    print("="*70)
    
    processor = ExcelPineconeProcessor(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        index_name="excel-demo-index"
    )
    
    # Search same query in different sheets
    query = "user authentication"
    sheets_to_compare = ["Functional Requirements", "Security Requirements"]
    
    for sheet in sheets_to_compare:
        print(f"\nSearching in '{sheet}':")
        results = processor.search_in_sheet(
            query=query,
            sheet_name=sheet,
            top_k=3
        )
        
        if results:
            print(f"  Found {len(results)} results")
            print(f"  Top result: {results[0]['content'][:150]}...")
        else:
            print(f"  No results found")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("Excel to Pinecone - Complete Usage Examples")
    print("="*70)
    
    # Make sure to set your API key as environment variable:
    # export PINECONE_API_KEY="your-key"
    
    # Check if API key is set
    if not os.getenv("PINECONE_API_KEY"):
        print("\n⚠️  WARNING: Please set environment variable:")
        print("   PINECONE_API_KEY")
        print("\nOn Windows PowerShell:")
        print('   $env:PINECONE_API_KEY="your-key"')
        print("\nOn Windows CMD:")
        print('   set PINECONE_API_KEY=your-key')
        return
    
    # Run examples (comment out the ones you don't want to run)
    
    # NEW: Natural language question answering with automatic sheet detection
    example_0a_answer_natural_question()
    
    # First, list all available sheets to see exact names
    # example_0_list_sheets()
    
    # Then run other examples
    # example_1_basic_processing()
    # example_2_search_within_sheet()
    # example_3_get_all_from_sheet()
    # example_4_natural_language_query()
    # example_5_list_and_search_all_sheets()
    # example_6_export_sheet_content()
    # example_7_compare_sheets()
    
    print("\n✅ Examples completed!")
    print("\nUncomment the examples you want to run in the main() function")


if __name__ == "__main__":
    main()

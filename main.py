"""
Main Script - Excel to Pinecone with Natural Language Questions
Simple interface to process Excel files and ask questions
"""

import os
from excel_pinecone_processor import ExcelPineconeProcessor


def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print("Excel to Pinecone - Natural Language Question Answering")
    print("="*70)
    
    # Check API key
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        print("\n❌ PINECONE_API_KEY not set!")
        print("\nSet it first:")
        print("  Windows CMD:        set PINECONE_API_KEY=your-key")
        print("  Windows PowerShell: $env:PINECONE_API_KEY=\"your-key\"")
        print("  Linux/Mac:          export PINECONE_API_KEY=your-key")
        return
    
    # Initialize processor
    print("\nInitializing processor with heavy model (all-mpnet-base-v2)...")
    processor = ExcelPineconeProcessor(
        pinecone_api_key=pinecone_key,
        index_name="excel-mpnet-index",
        embedding_model="all-mpnet-base-v2",
        dimension=768
    )
    print("✅ Processor ready!")
    
    # Check if you want to process a file
    print("\n" + "="*70)
    print("Options:")
    print("  1. Process new Excel file")
    print("  2. Ask questions about existing data")
    print("="*70)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Process Excel file
        excel_file = input("Enter Excel file path: ").strip()
        if not os.path.exists(excel_file):
            print(f"❌ File not found: {excel_file}")
            return
        
        print(f"\nProcessing {excel_file}...")
        stats = processor.process_excel_to_pinecone(excel_file)
        print(f"\n✅ Done! Processed {stats['sheets_processed']} sheets, {stats['total_chunks']} chunks")
    
    elif choice == "2":
        # Ask questions
        print("\n" + "="*70)
        print("Ask Questions (type 'exit' to quit)")
        print("="*70)
        print("\nExample questions:")
        print("  - what are integration requirements")
        print("  - show me security requirements")
        print("  - what are the functional requirements")
        print()
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print(f"\n🔍 Searching for: '{question}'")
            print("-" * 70)
            
            # Get answer
            result = processor.answer_question_with_sheet(
                query=question,
                score_threshold=0.25
            )
            
            # Display results
            if result['matched_sheet']:
                print(f"\n✅ Matched Sheet: '{result['matched_sheet']}'")
                print(f"   Confidence Score: {result['score']:.2%}")
                
                if result['content']:
                    print(f"   Found {len(result['content'])} chunks\n")
                    
                    # Show first 3 chunks
                    for i, chunk in enumerate(result['content'][:3], 1):
                        print(f"\n   Chunk {i} (Rows {chunk['rows']}):")
                        print(f"   {chunk['content'][:300]}...")
                    
                    if len(result['content']) > 3:
                        print(f"\n   ... and {len(result['content']) - 3} more chunks")
                else:
                    print("\n   ⚠️ Low confidence. Did you mean one of these sheets?")
                    for candidate in result['candidates'][:3]:
                        print(f"      - {candidate['sheet_name']} ({candidate['score']:.2%})")
            else:
                print("\n❌ No matching sheet found")
            
            print("-" * 70)
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()

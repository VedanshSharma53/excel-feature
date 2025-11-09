"""
Test Sheet Name Detection in Chunks
"""

import pandas as pd
import tempfile
import os
from xlsx_converter import _create_chunks_with_metadata


def test_sheet_name_detection():
    """Test that chunks correctly identify their source sheet"""
    
    print("\n" + "="*70)
    print("Testing Sheet Name Detection in Chunks")
    print("="*70 + "\n")
    
    # Create a test Excel file with multiple sheets
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        excel_path = tmp.name
    
    try:
        # Create sample data for multiple sheets
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Requirements
            df1 = pd.DataFrame({
                'ID': ['REQ-001', 'REQ-002', 'REQ-003'],
                'Title': ['User Login', 'Password Reset', 'Data Export'],
                'Priority': ['High', 'Medium', 'Low']
            })
            df1.to_excel(writer, sheet_name='Requirements', index=False)
            
            # Sheet 2: Test Cases
            df2 = pd.DataFrame({
                'TestID': ['TC-001', 'TC-002', 'TC-003'],
                'Description': ['Test Login', 'Test Password', 'Test Export'],
                'Status': ['Pass', 'Fail', 'Pass']
            })
            df2.to_excel(writer, sheet_name='TestCases', index=False)
            
            # Sheet 3: Bugs
            df3 = pd.DataFrame({
                'BugID': ['BUG-001', 'BUG-002'],
                'Summary': ['Login fails', 'Export crashes'],
                'Severity': ['Critical', 'High']
            })
            df3.to_excel(writer, sheet_name='Bugs', index=False)
        
        # Simulate extracted text (like Tika would produce)
        # In reality, Tika extracts text from each sheet sequentially
        simulated_text = """
        Requirements
        ID Title Priority
        REQ-001 User Login High
        REQ-002 Password Reset Medium
        REQ-003 Data Export Low
        
        TestCases
        TestID Description Status
        TC-001 Test Login Pass
        TC-002 Test Password Fail
        TC-003 Test Export Pass
        
        Bugs
        BugID Summary Severity
        BUG-001 Login fails Critical
        BUG-002 Export crashes High
        """
        
        # Create chunks with metadata
        chunks = _create_chunks_with_metadata(
            text=simulated_text,
            excel_path=excel_path,
            chunk_size=200,  # Small chunks to test multiple
            overlap=50
        )
        
        print(f"Created {len(chunks)} chunks\n")
        
        # Display chunk information
        for i, chunk in enumerate(chunks, 1):
            meta = chunk['metadata']
            sheet_name = meta.get('sheet_name', 'Unknown')
            chunk_preview = chunk['text'][:100].replace('\n', ' ')
            
            print(f"Chunk {i}:")
            print(f"  Sheet Name: {sheet_name}")
            print(f"  Has Header Detection: {meta.get('has_detected_header', 'N/A')}")
            
            if 'header_detection' in meta:
                print(f"  Header Confidence: {meta['header_detection'].get('header_confidence', 'N/A')}")
                print(f"  Detected Headers: {meta['header_detection'].get('detected_headers', [])[:3]}")
            
            print(f"  Preview: {chunk_preview}...")
            print()
        
        # Verify sheet names were detected
        sheet_names = [chunk['metadata'].get('sheet_name') for chunk in chunks]
        unique_sheets = set(sheet_names)
        
        print("="*70)
        print(f"Summary:")
        print(f"  Total Chunks: {len(chunks)}")
        print(f"  Unique Sheets Detected: {unique_sheets}")
        print(f"  Expected Sheets: {'Requirements', 'TestCases', 'Bugs'}")
        
        # Check if detection worked
        unknown_count = sum(1 for s in sheet_names if s == 'Unknown')
        if unknown_count > 0:
            print(f"  ⚠️ Warning: {unknown_count} chunks have 'Unknown' sheet name")
        else:
            print(f"  ✅ All chunks have identified sheet names!")
        
        print("="*70 + "\n")
        
    finally:
        # Cleanup
        if os.path.exists(excel_path):
            os.unlink(excel_path)


def test_single_sheet():
    """Test with single sheet Excel file"""
    
    print("\n" + "="*70)
    print("Testing Single Sheet Detection")
    print("="*70 + "\n")
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        excel_path = tmp.name
    
    try:
        # Create single sheet
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Paris']
        })
        df.to_excel(excel_path, sheet_name='People', index=False)
        
        simulated_text = """
        People
        Name Age City
        Alice 25 New York
        Bob 30 London
        Charlie 35 Paris
        """
        
        chunks = _create_chunks_with_metadata(
            text=simulated_text,
            excel_path=excel_path,
            chunk_size=100,
            overlap=20
        )
        
        print(f"Created {len(chunks)} chunks\n")
        
        for i, chunk in enumerate(chunks, 1):
            sheet_name = chunk['metadata'].get('sheet_name', 'Unknown')
            print(f"Chunk {i}: Sheet = '{sheet_name}'")
        
        # Verify
        all_same_sheet = all(chunk['metadata'].get('sheet_name') == 'People' for chunk in chunks)
        
        print("\n" + "="*70)
        if all_same_sheet:
            print("✅ All chunks correctly identified as 'People' sheet")
        else:
            sheet_names = [chunk['metadata'].get('sheet_name') for chunk in chunks]
            print(f"⚠️ Sheet names vary: {set(sheet_names)}")
        print("="*70 + "\n")
        
    finally:
        if os.path.exists(excel_path):
            os.unlink(excel_path)


if __name__ == "__main__":
    print("\n" + "🧪 SHEET NAME DETECTION TEST SUITE 🧪".center(70))
    
    test_sheet_name_detection()
    test_single_sheet()
    
    print("\n" + "="*70)
    print("✅ Tests completed!")
    print("="*70 + "\n")

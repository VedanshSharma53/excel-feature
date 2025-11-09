"""
Test Script for Intelligent Header Detection
Demonstrates the new header detection capabilities
"""

import pandas as pd
import json
from xlsx_converter import _detect_header_row, _analyze_sheet_structure


def test_with_headers():
    """Test case: Excel file with proper headers"""
    print("\n" + "="*70)
    print("TEST 1: File WITH Headers")
    print("="*70)
    
    # Create sample DataFrame with headers
    df = pd.DataFrame({
        'ID': ['REQ-001', 'REQ-002', 'REQ-003'],
        'Title': ['User Login', 'Password Reset', 'Data Export'],
        'Priority': ['High', 'Medium', 'Low'],
        'Status': ['Open', 'In Progress', 'Closed']
    })
    
    print("\nDataFrame:")
    print(df)
    
    # Test header detection
    result = _detect_header_row(df, "Requirements")
    print(f"\n📊 Header Detection Result:")
    print(f"   Has Header: {result['has_header']}")
    print(f"   Confidence: {result['header_confidence']}")
    print(f"   Headers: {result['detected_headers']}")
    print(f"   First Data Row: {result['first_data_row']}")
    print(f"   Reasons: {result['reasons']}")
    
    # Test structure analysis
    analysis = _analyze_sheet_structure(df, "Requirements")
    print(f"\n🔍 Structure Analysis:")
    print(f"   Total Rows: {analysis['total_rows']}")
    print(f"   Total Columns: {analysis['total_columns']}")
    print(f"   Data Completeness: {analysis['data_completeness_pct']}%")
    print(f"   Patterns: {analysis['patterns']}")
    
    print(f"\n📋 Column Details:")
    for col in analysis['columns']:
        print(f"   - {col['name']}: {col['type']} (unique: {col['unique_count']}, nulls: {col['null_count']})")
        print(f"     Samples: {col['sample_values']}")


def test_without_headers():
    """Test case: Excel file WITHOUT headers (pandas gives Unnamed columns)"""
    print("\n" + "="*70)
    print("TEST 2: File WITHOUT Headers (Raw Data)")
    print("="*70)
    
    # Simulate pandas reading a file without headers
    # Pandas would name columns as 0, 1, 2 or Unnamed: 0, etc.
    df = pd.DataFrame([
        ['REQ-001', 'User Login', 'High', 'Open'],
        ['REQ-002', 'Password Reset', 'Medium', 'In Progress'],
        ['REQ-003', 'Data Export', 'Low', 'Closed']
    ])
    
    # Rename to simulate unnamed columns
    df.columns = [f'Unnamed: {i}' for i in range(len(df.columns))]
    
    print("\nDataFrame (as pandas would read it):")
    print(df)
    
    # Test header detection
    result = _detect_header_row(df, "Data")
    print(f"\n📊 Header Detection Result:")
    print(f"   Has Header: {result['has_header']}")
    print(f"   Confidence: {result['header_confidence']}")
    print(f"   Headers: {result['detected_headers'][:4]}")
    print(f"   First Data Row: {result['first_data_row']}")
    print(f"   Reasons: {result['reasons']}")


def test_first_row_as_header():
    """Test case: First row should be detected as header"""
    print("\n" + "="*70)
    print("TEST 3: First Row IS Header (Unnamed columns)")
    print("="*70)
    
    # Create DataFrame where first row contains headers
    # but pandas didn't detect them (common scenario)
    df = pd.DataFrame([
        ['ID', 'Title', 'Priority', 'Status'],
        ['REQ-001', 'User Login', 'High', 'Open'],
        ['REQ-002', 'Password Reset', 'Medium', 'In Progress'],
        ['REQ-003', 'Data Export', 'Low', 'Closed']
    ])
    
    df.columns = [0, 1, 2, 3]  # Simulate generic column names
    
    print("\nDataFrame (pandas didn't detect headers):")
    print(df)
    
    # Test header detection
    result = _detect_header_row(df, "Sheet1")
    print(f"\n📊 Header Detection Result:")
    print(f"   Has Header: {result['has_header']}")
    print(f"   Confidence: {result['header_confidence']}")
    print(f"   Header Row Index: {result['header_row_index']}")
    print(f"   Detected Headers: {result['detected_headers']}")
    print(f"   First Data Row: {result['first_data_row']}")
    print(f"   Reasons: {result['reasons']}")
    
    # Test structure analysis (should use first row as headers)
    analysis = _analyze_sheet_structure(df, "Sheet1")
    print(f"\n🔍 Structure Analysis (after header correction):")
    print(f"   Headers: {analysis['headers']}")
    print(f"   Total Data Rows: {analysis['total_data_rows']}")
    print(f"   Sample Row: {analysis['sample_rows'][0] if analysis['sample_rows'] else 'N/A'}")


def test_mixed_types():
    """Test case: Mixed data types"""
    print("\n" + "="*70)
    print("TEST 4: Mixed Data Types")
    print("="*70)
    
    df = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Score': [95.5, 87.3, 92.1, 88.7],
        'Grade': ['A', 'B', 'A', 'B'],
        'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])
    })
    
    print("\nDataFrame:")
    print(df)
    print(f"\nData Types:\n{df.dtypes}")
    
    # Test structure analysis
    analysis = _analyze_sheet_structure(df, "Students")
    print(f"\n🔍 Structure Analysis:")
    print(f"   Data Completeness: {analysis['data_completeness_pct']}%")
    
    print(f"\n📋 Column Type Detection:")
    for col in analysis['columns']:
        print(f"   - {col['name']}: {col['type']}")
        print(f"     Unique values: {col['unique_count']}")
        print(f"     Samples: {col['sample_values']}")
    
    print(f"\n🔍 Detected Patterns:")
    for pattern in analysis['patterns']:
        print(f"   - {pattern}")


def test_with_nulls():
    """Test case: Data with missing values"""
    print("\n" + "="*70)
    print("TEST 5: Data with Missing Values")
    print("="*70)
    
    df = pd.DataFrame({
        'ID': ['A001', 'A002', None, 'A004'],
        'Name': ['John', None, 'Jane', 'Jack'],
        'Email': ['john@ex.com', 'bob@ex.com', None, None],
        'Phone': [None, '123-456', '789-012', '345-678']
    })
    
    print("\nDataFrame:")
    print(df)
    
    # Test structure analysis
    analysis = _analyze_sheet_structure(df, "Contacts")
    print(f"\n🔍 Structure Analysis:")
    print(f"   Total Cells: {analysis['total_cells']}")
    print(f"   Null Cells: {analysis['null_cells']}")
    print(f"   Data Completeness: {analysis['data_completeness_pct']}%")
    
    print(f"\n📋 Column Null Counts:")
    for col in analysis['columns']:
        null_pct = (col['null_count'] / analysis['total_rows']) * 100
        print(f"   - {col['name']}: {col['null_count']} nulls ({null_pct:.1f}%)")


if __name__ == "__main__":
    print("\n" + "🧪 INTELLIGENT HEADER DETECTION TEST SUITE 🧪".center(70))
    
    test_with_headers()
    test_without_headers()
    test_first_row_as_header()
    test_mixed_types()
    test_with_nulls()
    
    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("="*70 + "\n")

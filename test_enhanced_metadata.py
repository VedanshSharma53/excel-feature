"""
Test Enhanced Metadata Accuracy
Demonstrates the improved metadata extraction capabilities
"""

import pandas as pd
import tempfile
import os
import json
from xlsx_converter import _analyze_sheet_structure


def test_enhanced_metadata():
    """Test enhanced metadata with realistic data"""
    
    print("\n" + "="*80)
    print("ENHANCED METADATA TEST - Realistic Requirements Sheet")
    print("="*80 + "\n")
    
    # Create realistic requirements data
    df = pd.DataFrame({
        'Req_ID': ['REQ-001', 'REQ-002', 'REQ-003', 'REQ-004', 'REQ-005'],
        'Title': [
            'User Authentication System',
            'Password Reset Functionality',
            'Two-Factor Authentication',
            'Session Management',
            'User Role Management'
        ],
        'Description': [
            'Users must be able to login with email and password',
            'Users should be able to reset forgotten passwords via email',
            'Support 2FA using SMS or authenticator app',
            'Manage user sessions with 30-minute timeout',
            'Implement role-based access control (Admin, User, Guest)'
        ],
        'Priority': ['Critical', 'High', 'Medium', 'High', 'Critical'],
        'Status': ['In Progress', 'Completed', 'Planned', 'In Progress', 'Planned'],
        'Assigned_To': ['John Doe', 'Jane Smith', None, 'John Doe', 'Bob Johnson'],
        'Story_Points': [8, 5, 13, 5, 8],
        'Created_Date': pd.to_datetime([
            '2024-01-15', '2024-01-16', '2024-01-18', '2024-01-20', '2024-01-22'
        ]),
        'Target_Release': ['v1.0', 'v1.0', 'v1.1', 'v1.0', 'v1.1']
    })
    
    # Analyze structure
    metadata = _analyze_sheet_structure(df, "Requirements")
    
    # Display enhanced metadata
    print("📊 BASIC INFORMATION")
    print(f"  Sheet Name: {metadata['sheet_name']}")
    print(f"  Total Rows: {metadata['total_rows']}")
    print(f"  Total Columns: {metadata['total_columns']}")
    print(f"  Data Quality Score: {metadata['data_quality_score']}/100")
    print()
    
    print("🎯 HEADER DETECTION")
    header_det = metadata['header_detection']
    print(f"  Has Header: {header_det['has_header']}")
    print(f"  Confidence: {header_det['header_confidence'] * 100}%")
    print(f"  Headers: {header_det['detected_headers']}")
    print()
    
    print("📋 COLUMN ANALYSIS (Enhanced)")
    print(f"  Column Types Summary: {metadata['column_types_summary']}")
    print()
    for col in metadata['columns'][:5]:  # Show first 5
        print(f"  • {col['name']} ({col['type']})")
        print(f"    - Null: {col['null_count']} ({col['null_percentage']}%)")
        print(f"    - Unique: {col['unique_count']} (ratio: {col['uniqueness_ratio']})")
        print(f"    - Samples: {col['sample_values']}")
        if col['value_distribution']:
            print(f"    - Distribution: {col['value_distribution']}")
        if col['statistics']:
            print(f"    - Stats: {col['statistics']}")
        print()
    
    print("🔍 DETECTED PATTERNS")
    for pattern in metadata['patterns']:
        print(f"  • {pattern}")
    print()
    
    print("🔑 KEY COLUMNS")
    for key, values in metadata['key_columns'].items():
        print(f"  • {key}: {values}")
    print()
    
    print("📈 DATA QUALITY METRICS")
    print(f"  Overall Completeness: {metadata['data_completeness_pct']}%")
    print(f"  Quality Score: {metadata['data_quality_score']}/100")
    print(f"  Null Cells: {metadata['null_cells']} / {metadata['total_cells']}")
    print()
    print("  Column-wise Completeness:")
    for col, completeness in metadata['column_completeness'].items():
        status = "✅" if completeness == 100 else "⚠️" if completeness >= 80 else "❌"
        print(f"    {status} {col}: {completeness}%")
    print()
    
    if metadata['anomalies']:
        print("⚠️ DETECTED ANOMALIES")
        for anomaly in metadata['anomalies']:
            print(f"  • {anomaly}")
        print()
    
    print("📊 CATEGORY DISTRIBUTION")
    for col, dist in metadata['row_count_by_category'].items():
        print(f"  {col}:")
        for value, count in dist.items():
            print(f"    - {value}: {count}")
    print()
    
    print("🤖 LLM SUMMARY")
    print(f"  {metadata['llm_summary']}")
    print()
    
    print("💾 SAMPLE DATA (First 3 rows)")
    for i, row in enumerate(metadata['sample_rows'], 1):
        print(f"  Row {i}: {json.dumps(row, indent=4, default=str)}")
    print()


def test_data_with_issues():
    """Test with problematic data to show anomaly detection"""
    
    print("\n" + "="*80)
    print("ANOMALY DETECTION TEST - Data Quality Issues")
    print("="*80 + "\n")
    
    # Create data with quality issues
    df = pd.DataFrame({
        'ID': [1, 2, 2, 3, 4, 4, 5],  # Duplicate IDs
        'Name': ['Alice', None, 'Charlie', None, None, 'Frank', 'Grace'],  # Many nulls
        'Email': [
            'alice@example.com', 
            'bob@example', 
            None, 
            'invalid', 
            'dave@example.com',
            None,
            None
        ],
        'Score': [95, 87, 92, 88, 76, None, 82],
        'Active': [True, True, False, True, False, True, None]
    })
    
    metadata = _analyze_sheet_structure(df, "UserData")
    
    print("📊 DATA QUALITY")
    print(f"  Completeness: {metadata['data_completeness_pct']}%")
    print(f"  Quality Score: {metadata['data_quality_score']}/100")
    print()
    
    print("⚠️ DETECTED ANOMALIES")
    if metadata['anomalies']:
        for anomaly in metadata['anomalies']:
            print(f"  • {anomaly}")
    else:
        print("  No anomalies detected")
    print()
    
    print("📋 COLUMN DETAILS")
    for col in metadata['columns']:
        print(f"  • {col['name']} ({col['type']})")
        print(f"    Null %: {col['null_percentage']}%")
        print(f"    Uniqueness: {col['uniqueness_ratio']}")
    print()


def test_mixed_types():
    """Test advanced type detection"""
    
    print("\n" + "="*80)
    print("ADVANCED TYPE DETECTION TEST")
    print("="*80 + "\n")
    
    df = pd.DataFrame({
        'Integer_Col': [1, 2, 3, 4, 5],
        'Float_Col': [1.5, 2.7, 3.2, 4.1, 5.9],
        'Email_Col': [
            'user1@example.com',
            'user2@example.com',
            'user3@example.com',
            'user4@example.com',
            'user5@example.com'
        ],
        'URL_Col': [
            'https://example.com/page1',
            'https://example.com/page2',
            'http://test.com/page3',
            'https://example.com/page4',
            'https://example.com/page5'
        ],
        'Boolean_Col': [True, False, True, True, False],
        'Date_Col': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01'])
    })
    
    metadata = _analyze_sheet_structure(df, "TypeTest")
    
    print("📋 DETECTED COLUMN TYPES")
    for col in metadata['columns']:
        print(f"  • {col['name']}: {col['type']}")
        if col['statistics']:
            print(f"    Stats: {col['statistics']}")
    print()


def test_relationships():
    """Test relationship detection"""
    
    print("\n" + "="*80)
    print("RELATIONSHIP DETECTION TEST")
    print("="*80 + "\n")
    
    df = pd.DataFrame({
        'Task_ID': ['T-001', 'T-002', 'T-003', 'T-004'],
        'Title': ['Setup Database', 'Create API', 'Write Tests', 'Deploy'],
        'Parent_ID': [None, 'T-001', 'T-002', 'T-003'],
        'User_ID': ['U-1', 'U-2', 'U-1', 'U-3'],
        'Status': ['Done', 'In Progress', 'Done', 'Planned']
    })
    
    metadata = _analyze_sheet_structure(df, "Tasks")
    
    print("🔍 DETECTED PATTERNS")
    for pattern in metadata['patterns']:
        print(f"  • {pattern}")
    print()


if __name__ == "__main__":
    print("\n" + "🧪 ENHANCED METADATA ACCURACY TEST SUITE 🧪".center(80))
    
    test_enhanced_metadata()
    test_data_with_issues()
    test_mixed_types()
    test_relationships()
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80 + "\n")

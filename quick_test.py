"""Quick test of enhanced metadata"""
import pandas as pd
from xlsx_converter import _analyze_sheet_structure
import json

# Create sample data
df = pd.DataFrame({
    'ID': ['REQ-001', 'REQ-002', 'REQ-003'],
    'Title': ['User Login', 'Password Reset', 'Data Export'],
    'Priority': ['High', 'Medium', 'Low'],
    'Status': ['Open', 'In Progress', 'Closed'],
    'Points': [8, 5, 3]
})

print("\n" + "="*70)
print("ENHANCED METADATA TEST")
print("="*70 + "\n")

# Analyze
metadata = _analyze_sheet_structure(df, "Requirements")

# Show key improvements
print("✅ ENHANCED FEATURES:\n")

print("1. Column Types Summary:")
print(f"   {metadata['column_types_summary']}\n")

print("2. Enhanced Column Info (with statistics):")
for col in metadata['columns'][:3]:
    print(f"   • {col['name']} ({col['type']})")
    print(f"     - Null: {col['null_percentage']}%")
    print(f"     - Uniqueness Ratio: {col['uniqueness_ratio']}")
    if col.get('statistics'):
        print(f"     - Stats: {col['statistics']}")
    if col.get('value_distribution'):
        print(f"     - Distribution: {col['value_distribution']}")

print("\n3. Key Columns Identified:")
print(f"   {metadata['key_columns']}\n")

print("4. Data Quality Score:")
print(f"   {metadata['data_quality_score']}/100\n")

print("5. Column-wise Completeness:")
for col, comp in list(metadata['column_completeness'].items())[:3]:
    print(f"   {col}: {comp}%")

print("\n6. LLM Summary:")
print(f"   '{metadata['llm_summary']}'\n")

if metadata.get('anomalies'):
    print("7. Anomalies Detected:")
    for anomaly in metadata['anomalies']:
        print(f"   • {anomaly}")

print("\n" + "="*70)
print("✅ Enhanced metadata provides much richer context for LLM!")
print("="*70 + "\n")

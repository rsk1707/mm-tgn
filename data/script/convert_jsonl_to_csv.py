#!/usr/bin/env python3
"""
Convert JSONL metadata files to CSV format for generate_embeddings.py

Usage:
    python convert_jsonl_to_csv.py \
        --input sports-5core-raw-text.jsonl \
        --output sports-5core-metadata.csv
"""

import json
import argparse
import pandas as pd
from pathlib import Path


def convert_jsonl_to_csv(input_path: str, output_path: str):
    """Convert JSONL to CSV format."""
    
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line: {e}")
    
    df = pd.DataFrame(records)
    
    # Rename columns to match expected format
    # The JSONL has: asin, raw_text
    # We want: asin, description (for text column)
    if 'raw_text' in df.columns:
        df = df.rename(columns={'raw_text': 'description'})
    
    df.to_csv(output_path, index=False)
    print(f"✅ Converted {len(records)} records")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Columns: {list(df.columns)}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to CSV")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()
    
    convert_jsonl_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()


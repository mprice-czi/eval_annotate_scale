#!/usr/bin/env python3
"""
Verify CLEAR.csv record count and demonstrate correct parsing approach.

This script confirms that CLEAR.csv contains 4,726 total rows but only 4,724 valid records,
with 2 empty rows at the end of the file.
"""

import sys
from pathlib import Path

import pandas as pd


def verify_clear_count(csv_path: str = "data/CLEAR.csv") -> None:
    """Verify the CLEAR.csv record count using pandas."""

    if not Path(csv_path).exists():
        print(f"❌ CLEAR.csv not found at {csv_path}")
        sys.exit(1)

    try:
        # Read CSV with pandas (handles BOM and complex structure automatically)
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        print("📊 CLEAR.csv Analysis Results:")
        print("=" * 40)
        print(f"Total rows in CSV: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")

        # Check for null IDs (invalid records)
        id_column = df.iloc[:, 0]  # First column should be ID
        null_ids = id_column.isnull().sum()
        valid_records = len(df) - null_ids

        print(f"Records with null/empty ID: {null_ids}")
        print(f"Valid records with data: {valid_records}")

        if null_ids > 0:
            print(f"\n🔍 Rows with null IDs:")
            null_rows = df[id_column.isnull()]
            for idx in null_rows.index:
                print(f"  Row {idx}: First 3 fields = {df.iloc[idx, :3].tolist()}")

        # Show first and last few valid record IDs
        valid_df = df[id_column.notnull()]
        if len(valid_df) > 0:
            print(f"\n✅ First 5 valid IDs: {valid_df.iloc[:5, 0].tolist()}")
            print(f"✅ Last 5 valid IDs: {valid_df.iloc[-5:, 0].tolist()}")

        print(f"\n📈 Summary:")
        print(f"  - Use len(df) for total rows: {len(df)}")
        print(
            f"  - Use len(df[df.iloc[:, 0].notnull()]) for valid records: {valid_records}"
        )
        print(f"  - Recommended: Filter out null IDs before processing")

    except Exception as e:
        print(f"❌ Error reading CLEAR.csv: {e}")
        sys.exit(1)


if __name__ == "__main__":
    verify_clear_count()

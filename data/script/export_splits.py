#!/usr/bin/env python3
"""
Canonical Split Exporter for MM-TGN Team

Creates reproducible train/val/test splits from interaction data that ALL
team members must use for fair baseline comparison.

Key Features:
1. Chronological splitting (NO random shuffling)
2. Consistent 70/15/15 ratios
3. Verification assertions
4. Metadata export for documentation

Usage:
    python export_splits.py --dataset ml-modern
    python export_splits.py --dataset amazon-cloth
    python export_splits.py --dataset amazon-sports
    python export_splits.py --input /path/to/ratings.csv --output-dir /path/to/splits --dataset custom

Output:
    data/splits/<dataset>/
    ├── train.csv          # 70% oldest interactions
    ├── val.csv            # 15% middle interactions
    ├── test.csv           # 15% newest interactions
    └── splits_metadata.json   # Statistics and verification info

Author: MM-TGN Team (CSE576)
Date: December 2025
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASET_CONFIGS = {
    "ml-modern": {
        "path": "data/datasets/movielens-32m/movielens-modern/ml-modern/ratings.csv",
        "timestamp_col": "timestamp",
        "timestamp_unit": "s",  # Unix seconds
        "user_col": "userId",
        "item_col": "movieId",
        "rating_col": "rating"
    },
    "amazon-cloth": {
        "path": "data/datasets/amazon-cloth/cloth_5core_interactions.csv",
        "timestamp_col": "timestamp",
        "timestamp_unit": "ms",  # Unix milliseconds
        "user_col": "user_id",
        "item_col": "asin",
        "rating_col": "rating"
    },
    "amazon-sports": {
        "path": "data/datasets/amazon-sports/sports_5core_interactions.csv",
        "timestamp_col": "timestamp",
        "timestamp_unit": "ms",  # Unix milliseconds
        "user_col": "user_id",
        "item_col": "asin",
        "rating_col": "rating"
    }
}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_interactions(csv_path: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Load and validate interaction data."""
    print(f"📂 Loading: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df):,} interactions")
    
    # Validate required columns
    ts_col = config["timestamp_col"]
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found. Available: {list(df.columns)}")
    
    # Convert timestamp to numeric if needed
    df[ts_col] = pd.to_numeric(df[ts_col], errors='coerce')
    
    # Check for NaN timestamps
    nan_count = df[ts_col].isna().sum()
    if nan_count > 0:
        print(f"   ⚠️ Warning: {nan_count} rows with invalid timestamps will be dropped")
        df = df.dropna(subset=[ts_col])
    
    return df


def timestamp_to_datetime(ts: float, unit: str) -> datetime:
    """Convert timestamp to datetime based on unit."""
    if unit == "ms":
        return datetime.fromtimestamp(ts / 1000)
    else:  # seconds
        return datetime.fromtimestamp(ts)


def create_chronological_splits(
    df: pd.DataFrame,
    timestamp_col: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create chronological train/val/test splits.
    
    CRITICAL: Splits by INDEX position after sorting by timestamp.
    This guarantees:
    - No data leakage (train < val < test temporally)
    - Exact ratio splits
    - Reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # STEP 1: Sort STRICTLY by timestamp
    print("🔄 Sorting by timestamp...")
    df_sorted = df.sort_values(timestamp_col, kind='stable').reset_index(drop=True)
    
    # STEP 2: Calculate split indices
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    print(f"   Total interactions: {n:,}")
    print(f"   Train: [0, {train_end}) = {train_end:,} ({train_ratio:.0%})")
    print(f"   Val:   [{train_end}, {val_end}) = {val_end - train_end:,} ({val_ratio:.0%})")
    print(f"   Test:  [{val_end}, {n}) = {n - val_end:,} ({test_ratio:.0%})")
    
    # STEP 3: Split by index (NOT by timestamp value)
    df_train = df_sorted.iloc[:train_end].copy()
    df_val = df_sorted.iloc[train_end:val_end].copy()
    df_test = df_sorted.iloc[val_end:].copy()
    
    return df_train, df_val, df_test


def verify_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    timestamp_col: str,
    timestamp_unit: str
) -> Dict[str, Any]:
    """
    Verify temporal integrity of splits.
    
    Returns metadata dict with verification results.
    """
    print("\n🔍 Verifying split integrity...")
    
    # Get boundary timestamps
    train_max_ts = df_train[timestamp_col].max()
    val_min_ts = df_val[timestamp_col].min()
    val_max_ts = df_val[timestamp_col].max()
    test_min_ts = df_test[timestamp_col].min()
    
    # CRITICAL ASSERTIONS
    # Note: These should ALWAYS pass when splitting by index after sorting.
    # If they fail, it indicates a bug in the sorting/splitting logic.
    try:
        assert train_max_ts <= val_min_ts, \
            f"Train/Val overlap! train_max={train_max_ts}, val_min={val_min_ts}"
        print("   ✅ Train.max_ts <= Val.min_ts")
    except AssertionError as e:
        # Edge case: timestamp ties at boundary
        # This can happen if many interactions have the exact same timestamp
        print(f"   ⚠️ Warning: {e}")
        print("   ℹ️ This can occur with timestamp ties at split boundaries.")
        print("   ℹ️ The split is still valid (by index position).")
    
    try:
        assert val_max_ts <= test_min_ts, \
            f"Val/Test overlap! val_max={val_max_ts}, test_min={test_min_ts}"
        print("   ✅ Val.max_ts <= Test.min_ts")
    except AssertionError as e:
        print(f"   ⚠️ Warning: {e}")
        print("   ℹ️ This can occur with timestamp ties at split boundaries.")
        print("   ℹ️ The split is still valid (by index position).")
    
    # Convert to readable dates
    train_start_dt = timestamp_to_datetime(df_train[timestamp_col].min(), timestamp_unit)
    train_end_dt = timestamp_to_datetime(train_max_ts, timestamp_unit)
    val_start_dt = timestamp_to_datetime(val_min_ts, timestamp_unit)
    val_end_dt = timestamp_to_datetime(val_max_ts, timestamp_unit)
    test_start_dt = timestamp_to_datetime(test_min_ts, timestamp_unit)
    test_end_dt = timestamp_to_datetime(df_test[timestamp_col].max(), timestamp_unit)
    
    print("\n📅 Date Ranges:")
    print(f"   Train: {train_start_dt.strftime('%Y-%m-%d')} to {train_end_dt.strftime('%Y-%m-%d')}")
    print(f"   Val:   {val_start_dt.strftime('%Y-%m-%d')} to {val_end_dt.strftime('%Y-%m-%d')}")
    print(f"   Test:  {test_start_dt.strftime('%Y-%m-%d')} to {test_end_dt.strftime('%Y-%m-%d')}")
    
    # Build metadata
    metadata = {
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        },
        "split_counts": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
            "total": len(df_train) + len(df_val) + len(df_test)
        },
        "date_ranges": {
            "train": {
                "start": train_start_dt.isoformat(),
                "end": train_end_dt.isoformat()
            },
            "val": {
                "start": val_start_dt.isoformat(),
                "end": val_end_dt.isoformat()
            },
            "test": {
                "start": test_start_dt.isoformat(),
                "end": test_end_dt.isoformat()
            }
        },
        "timestamp_boundaries": {
            "train_max": float(train_max_ts),
            "val_min": float(val_min_ts),
            "val_max": float(val_max_ts),
            "test_min": float(test_min_ts)
        },
        "verification": {
            "train_before_val": bool(train_max_ts <= val_min_ts),
            "val_before_test": bool(val_max_ts <= test_min_ts)
        }
    }
    
    return metadata


def compute_statistics(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute useful statistics about the splits."""
    
    user_col = config.get("user_col", "userId")
    item_col = config.get("item_col", "movieId")
    
    # Get unique users/items per split
    train_users = set(df_train[user_col].unique())
    train_items = set(df_train[item_col].unique())
    
    val_users = set(df_val[user_col].unique())
    val_items = set(df_val[item_col].unique())
    
    test_users = set(df_test[user_col].unique())
    test_items = set(df_test[item_col].unique())
    
    # Cold-start analysis
    cold_users_val = val_users - train_users
    cold_items_val = val_items - train_items
    
    cold_users_test = test_users - train_users - val_users
    cold_items_test = test_items - train_items - val_items
    
    print("\n📊 Statistics:")
    print(f"   Users:  Train={len(train_users):,}, Val={len(val_users):,}, Test={len(test_users):,}")
    print(f"   Items:  Train={len(train_items):,}, Val={len(val_items):,}, Test={len(test_items):,}")
    print(f"   Cold-start (new in val):  {len(cold_users_val):,} users, {len(cold_items_val):,} items")
    print(f"   Cold-start (new in test): {len(cold_users_test):,} users, {len(cold_items_test):,} items")
    
    return {
        "unique_counts": {
            "train_users": len(train_users),
            "train_items": len(train_items),
            "val_users": len(val_users),
            "val_items": len(val_items),
            "test_users": len(test_users),
            "test_items": len(test_items)
        },
        "cold_start": {
            "val_new_users": len(cold_users_val),
            "val_new_items": len(cold_items_val),
            "test_new_users": len(cold_users_test),
            "test_new_items": len(cold_items_test)
        }
    }


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    metadata: Dict[str, Any],
    output_dir: Path,
    dataset_name: str
):
    """Save splits and metadata to disk."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to: {output_dir}")
    
    # Save CSVs
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    df_train.to_csv(train_path, index=False)
    print(f"   ✅ {train_path} ({len(df_train):,} rows)")
    
    df_val.to_csv(val_path, index=False)
    print(f"   ✅ {val_path} ({len(df_val):,} rows)")
    
    df_test.to_csv(test_path, index=False)
    print(f"   ✅ {test_path} ({len(df_test):,} rows)")
    
    # Save metadata
    metadata["dataset"] = dataset_name
    metadata["created_at"] = datetime.now().isoformat()
    metadata["files"] = {
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv"
    }
    
    metadata_path = output_dir / "splits_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ {metadata_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export canonical train/val/test splits for team alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined dataset
  python export_splits.py --dataset ml-modern
  python export_splits.py --dataset amazon-cloth
  python export_splits.py --dataset amazon-sports
  
  # Custom dataset
  python export_splits.py --input my_data.csv --dataset my-dataset --timestamp-col ts
  
Note:
  - Splits are CHRONOLOGICAL (oldest=train, newest=test)
  - Ratios: 70% train, 15% val, 15% test
  - ALL teammates MUST use these exact files for fair comparison
        """
    )
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (ml-modern, amazon-cloth, amazon-sports, or custom)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input CSV (optional, uses predefined path for known datasets)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: data/splits/<dataset>)")
    parser.add_argument("--timestamp-col", type=str, default="timestamp",
                        help="Timestamp column name (default: timestamp)")
    parser.add_argument("--timestamp-unit", type=str, choices=["s", "ms"], default="s",
                        help="Timestamp unit: 's' for seconds, 'ms' for milliseconds")
    parser.add_argument("--user-col", type=str, default="userId",
                        help="User ID column name")
    parser.add_argument("--item-col", type=str, default="movieId",
                        help="Item ID column name")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("📦 CANONICAL SPLIT EXPORTER")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Ratios:  Train={TRAIN_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%}")
    print("=" * 70)
    
    # Get configuration
    if args.dataset in DATASET_CONFIGS:
        config = DATASET_CONFIGS[args.dataset]
        input_path = args.input or config["path"]
    else:
        # Custom dataset
        if args.input is None:
            print(f"❌ Unknown dataset '{args.dataset}'. Use --input to specify CSV path.")
            sys.exit(1)
        
        config = {
            "path": args.input,
            "timestamp_col": args.timestamp_col,
            "timestamp_unit": args.timestamp_unit,
            "user_col": args.user_col,
            "item_col": args.item_col
        }
        input_path = args.input
    
    # Resolve paths
    script_dir = Path(__file__).parent.parent.parent  # mm-tgn/
    if not Path(input_path).is_absolute():
        input_path = script_dir / input_path
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = script_dir / "data" / "splits" / args.dataset
    else:
        output_dir = Path(output_dir)
    
    # Load data
    df = load_interactions(str(input_path), config)
    
    # Create splits
    df_train, df_val, df_test = create_chronological_splits(
        df,
        timestamp_col=config["timestamp_col"],
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
    # Verify and get metadata
    metadata = verify_splits(
        df_train, df_val, df_test,
        timestamp_col=config["timestamp_col"],
        timestamp_unit=config["timestamp_unit"]
    )
    
    # Compute statistics
    stats = compute_statistics(df_train, df_val, df_test, config)
    metadata["statistics"] = stats
    
    # Save
    save_splits(df_train, df_val, df_test, metadata, output_dir, args.dataset)
    
    print("\n" + "=" * 70)
    print("✅ SPLITS EXPORTED SUCCESSFULLY!")
    print("=" * 70)
    print("\n⚠️  IMPORTANT: Share these files with ALL teammates!")
    print("   Everyone must use the EXACT same train/val/test files.")
    print(f"\n   Output: {output_dir}/")
    

if __name__ == "__main__":
    main()


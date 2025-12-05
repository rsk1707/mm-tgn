#!/usr/bin/env python3
"""
Export Fixed Evaluation Samples for Fair Comparison

This script exports the exact test samples used for ranking evaluation,
ensuring all models (MM-TGN, LightGCN, SASRec, MMGCN) evaluate on the
same interactions.

Usage:
    python export_eval_samples.py \
        --splits-dir data/splits/ml-modern \
        --output-dir data/eval_samples \
        --sample-size 5000 \
        --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Export fixed evaluation samples")
    parser.add_argument("--splits-dir", type=str, required=True,
                        help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for evaluation samples")
    parser.add_argument("--sample-size", type=int, default=5000,
                        help="Number of test interactions to sample for ranking eval")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--full-eval", action="store_true",
                        help="Export full test set (no sampling)")
    
    args = parser.parse_args()
    
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test set
    test_path = splits_dir / "test.csv"
    if not test_path.exists():
        print(f"❌ Test file not found: {test_path}")
        print("   Run export_splits.py first!")
        return
    
    print(f"📊 Loading test set from {test_path}")
    df_test = pd.read_csv(test_path)
    
    # Detect column names
    user_col = 'userId' if 'userId' in df_test.columns else 'user_id'
    item_col = 'movieId' if 'movieId' in df_test.columns else 'item_id'
    if 'asin' in df_test.columns:
        item_col = 'asin'
    
    ts_col = 'timestamp' if 'timestamp' in df_test.columns else 'ts'
    
    print(f"   Total test interactions: {len(df_test):,}")
    print(f"   Columns: user={user_col}, item={item_col}, ts={ts_col}")
    
    # Sample or use full
    np.random.seed(args.seed)
    
    if args.full_eval or args.sample_size >= len(df_test):
        print(f"\n📋 Using FULL test set ({len(df_test):,} interactions)")
        df_sample = df_test.copy()
        sample_indices = np.arange(len(df_test))
    else:
        print(f"\n📋 Sampling {args.sample_size:,} interactions (seed={args.seed})")
        sample_indices = np.random.choice(len(df_test), args.sample_size, replace=False)
        sample_indices = np.sort(sample_indices)  # Keep temporal order
        df_sample = df_test.iloc[sample_indices].copy()
    
    # Get dataset name from path
    dataset_name = splits_dir.name
    
    # Save sampled test set
    sample_csv_path = output_dir / f"{dataset_name}_eval_sample.csv"
    df_sample.to_csv(sample_csv_path, index=False)
    print(f"   Saved: {sample_csv_path}")
    
    # Save sample indices (for direct indexing)
    indices_path = output_dir / f"{dataset_name}_eval_indices.npy"
    np.save(indices_path, sample_indices)
    print(f"   Saved: {indices_path}")
    
    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "total_test_size": len(df_test),
        "sample_size": len(df_sample),
        "seed": args.seed,
        "full_eval": args.full_eval or args.sample_size >= len(df_test),
        "columns": {
            "user": user_col,
            "item": item_col,
            "timestamp": ts_col
        },
        "stats": {
            "n_users": df_sample[user_col].nunique(),
            "n_items": df_sample[item_col].nunique(),
            "min_timestamp": int(df_sample[ts_col].min()),
            "max_timestamp": int(df_sample[ts_col].max())
        }
    }
    
    metadata_path = output_dir / f"{dataset_name}_eval_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved: {metadata_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SAMPLE SUMMARY")
    print("=" * 60)
    print(f"Dataset:      {dataset_name}")
    print(f"Sample size:  {len(df_sample):,} / {len(df_test):,} ({100*len(df_sample)/len(df_test):.1f}%)")
    print(f"Users:        {metadata['stats']['n_users']:,}")
    print(f"Items:        {metadata['stats']['n_items']:,}")
    print(f"Seed:         {args.seed}")
    print()
    print("📁 Files for your teammate:")
    print(f"   {sample_csv_path.name}  - Test interactions to evaluate on")
    print(f"   {indices_path.name}     - Row indices into original test.csv")
    print(f"   {metadata_path.name}   - Metadata and stats")
    print()
    print("=" * 60)
    print("SHARE THESE WITH YOUR TEAMMATE!")
    print("=" * 60)
    print()
    print("All baselines (LightGCN, SASRec, MMGCN) should evaluate on")
    print(f"the SAME {len(df_sample):,} test interactions for fair comparison.")


if __name__ == "__main__":
    main()


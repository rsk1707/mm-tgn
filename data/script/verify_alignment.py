import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys

def verify_alignment(ratings_path, features_dir, dataset_name):
    print(f"🕵️  VERIFYING ALIGNMENT: {dataset_name}")
    print(f"    Ratings:  {ratings_path}")
    print(f"    Features: {features_dir}")
    
    # -----------------------------------------------------------
    # 1. Load the Feature Index (The Source of Truth)
    # -----------------------------------------------------------
    ids_path = Path(features_dir) / f"{features_dir/dataset_name}_ids.npy"
    if not ids_path.exists():
        sys.exit(f"❌ Critical Error: ID file not found at {ids_path}")
        
    # Load and ensure they are strings
    feature_ids_raw = np.load(ids_path, allow_pickle=True)
    feature_ids = [str(x).strip() for x in feature_ids_raw]
    
    # Create the Lookup Map (ID -> Row Index)
    feat_map = {fid: idx for idx, fid in enumerate(feature_ids)}
    
    print(f"\n📘 FEATURE STORE ({dataset_name}_ids.npy)")
    print(f"   - Total Items with Features: {len(feature_ids):,}")
    print(f"   - Sample IDs (First 5): {feature_ids[:5]}")
    print(f"   - Sample IDs (Last 5):  {feature_ids[-5:]}")
    print("   ✅ This file defines the Row Order of your .npy matrices.")

    # -----------------------------------------------------------
    # 2. Load the Active Interactions
    # -----------------------------------------------------------
    print(f"\naaa INTERACTION GRAPH ({Path(ratings_path).name})")
    df = pd.read_csv(ratings_path)
    
    # Auto-detect ID column
    if 'movieId' in df.columns: id_col = 'movieId'
    elif 'asin' in df.columns: id_col = 'asin'
    elif 'book_id' in df.columns: id_col = 'book_id'
    else: sys.exit("❌ Could not detect Item ID column in ratings.csv")
        
    active_ids = df[id_col].astype(str).str.strip().unique()
    print(f"   - Total Unique Active Items: {len(active_ids):,}")

    # -----------------------------------------------------------
    # 3. The Intersection Logic (The Proof)
    # -----------------------------------------------------------
    print(f"\n🔗 ALIGNMENT CHECK")
    
    # A. Active items that HAVE features
    found_items = [i for i in active_ids if i in feat_map]
    
    # B. Active items MISSING features (The 23 items)
    missing_items = [i for i in active_ids if i not in feat_map]
    
    # C. Inactive items (The 17k "Cold Start" Reserve)
    reserve_items = set(feature_ids) - set(active_ids)
    
    coverage = len(found_items) / len(active_ids)
    
    print(f"   - Active Items with Features:    {len(found_items):,} ({coverage:.2%})")
    print(f"   - Active Items MISSING Features: {len(missing_items):,} (Zero-padded)")
    print(f"   - Inactive 'Cold Start' Items:   {len(reserve_items):,} (Ready for future use)")

    if coverage < 0.90:
        print("❌ WARNING: Coverage is dangerously low (<90%). Check ID alignment.")
    else:
        print("✅ SUCCESS: High feature coverage.")

    # -----------------------------------------------------------
    # 4. The "Spot Check" (For your Teammate)
    # -----------------------------------------------------------
    print(f"\n📍 SPOT CHECK (Proof of Indexing)")
    # Pick a random active item
    test_id = found_items[0]
    row_idx = feat_map[test_id]
    
    print(f"   Query Movie ID: '{test_id}'")
    print(f"   -> Found at Row Index: {row_idx}")
    print(f"   -> Meaning: The vector at `features[{row_idx}]` belongs to Movie '{test_id}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings-path", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    
    verify_alignment(args.ratings_path, args.features_dir, args.dataset)
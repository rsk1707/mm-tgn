import pandas as pd
import numpy as np
import argparse
import json
from pathlib import Path
import sys

def format_dataset(
    ratings_path, 
    features_dir, 
    dataset_name, 
    text_model, 
    image_model, 
    output_dir
):
    print(f"🚀 Formatting {dataset_name} for TGN...")
    print(f"   - Input Interactions: {ratings_path}")
    print(f"   - Feature Source: {features_dir} (Text: {text_model}, Image: {image_model})")
    
    # Setup Paths
    feat_path = Path(features_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Load and Normalize Interactions
    # ---------------------------------------------------------
    if not Path(ratings_path).exists():
        sys.exit(f"❌ Error: Ratings file not found at {ratings_path}")
        
    df = pd.read_csv(ratings_path)
    
    # Universal Column Mapping
    rename_map = {}
    if 'userId' in df.columns: rename_map['userId'] = 'u'
    elif 'user_id' in df.columns: rename_map['user_id'] = 'u'
    elif 'reviewerID' in df.columns: rename_map['reviewerID'] = 'u'
    
    if 'movieId' in df.columns: rename_map['movieId'] = 'i'
    elif 'book_id' in df.columns: rename_map['book_id'] = 'i'
    elif 'asin' in df.columns: rename_map['asin'] = 'i'
    elif 'parent_asin' in df.columns: rename_map['parent_asin'] = 'i'
    
    if 'timestamp' in df.columns: rename_map['timestamp'] = 'ts'
    if 'rating' in df.columns: rename_map['rating'] = 'label'
    
    df = df.rename(columns=rename_map)
    
    # Validation
    required = ['u', 'i', 'ts', 'label']
    for c in required:
        if c not in df.columns:
            sys.exit(f"❌ Error: Could not find column for '{c}'. Available: {list(df.columns)}")

    # Ensure TS is sorted (Critical for TGN)
    print("   - Sorting by timestamp...")
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Force IDs to string
    df['u'] = df['u'].astype(str)
    df['i'] = df['i'].astype(str)

    # ---------------------------------------------------------
    # 2. Bipartite Remapping (The TGN Indexing Logic)
    # ---------------------------------------------------------
    # CRITICAL: Index 0 is RESERVED for padding (null neighbors in TGN).
    # All real nodes must start at index 1.
    
    unique_users = df['u'].unique()
    unique_items = df['i'].unique()
    
    num_users = len(unique_users)
    num_items = len(unique_items)
    
    print(f"   - Graph Stats: {num_users} Users, {num_items} Items, {len(df)} Edges")
    
    # Create Mappings (1-BASED INDEXING)
    # Index 0 = Padding (reserved for TGN neighbor masking)
    # Users: 1 to N_u (inclusive)
    user_map = {uid: i + 1 for i, uid in enumerate(unique_users)}
    
    # Items: N_u + 1 to N_u + N_i (inclusive)
    item_map = {iid: i + num_users + 1 for i, iid in enumerate(unique_items)}
    
    # Apply Mapping
    df['u_idx'] = df['u'].map(user_map)
    df['i_idx'] = df['i'].map(item_map)
    df['idx'] = df.index + 1  # 1-based edge index
    
    # ---------------------------------------------------------
    # 3. Load & Align Features
    # ---------------------------------------------------------
    print("   - Loading Feature Matrices...")
    txt_file = feat_path / f"{dataset_name}_text_{text_model}.npy"
    img_file = feat_path / f"{dataset_name}_image_{image_model}.npy"
    ids_file = feat_path / f"{dataset_name}_ids.npy"
    
    if not txt_file.exists() or not img_file.exists() or not ids_file.exists():
        sys.exit(f"❌ Missing feature files in {feat_path}. Did you run verify_features.py?")

    txt_emb = np.load(txt_file)
    img_emb = np.load(img_file)
    feat_ids = np.load(ids_file)
    
    # Lookup: {Raw_Item_ID: Row_Index_in_NPY}
    feat_lookup = {str(fid): idx for idx, fid in enumerate(feat_ids)}
    
    combined_dim = txt_emb.shape[1] + img_emb.shape[1]
    total_nodes = num_users + num_items
    
    # Initialize Node Matrix with 1-based indexing
    # Row 0 = Padding (zero vector, reserved for TGN neighbor masking)
    # Rows 1 to N_u = Users (zero vectors, learned embeddings)
    # Rows N_u+1 to N_u+N_i = Items (SOTA features)
    node_features = np.zeros((total_nodes + 1, combined_dim), dtype=np.float32)
    print(f"   - Node Feature Matrix: {node_features.shape} (Row 0 = Padding)")
    
    found_count = 0
    missing_count = 0
    
    for raw_iid, global_idx in item_map.items():
        if raw_iid in feat_lookup:
            feat_row = feat_lookup[raw_iid]
            vector = np.concatenate([txt_emb[feat_row], img_emb[feat_row]])
            node_features[global_idx] = vector
            found_count += 1
        else:
            missing_count += 1

    print(f"   - Feature Mapping: {found_count} Found, {missing_count} Missing (Zero-padded)")

    # ---------------------------------------------------------
    # 4. Save Outputs (CSV, NPY, JSON)
    # ---------------------------------------------------------
    
    # A. Edge List
    final_csv = df[['u_idx', 'i_idx', 'ts', 'label', 'idx']]
    final_csv.columns = ['u', 'i', 'ts', 'label', 'idx']
    csv_out = out_path / f"ml_{dataset_name}.csv"
    final_csv.to_csv(csv_out, index=False)
    
    # B. Feature Matrix
    npy_out = out_path / f"ml_{dataset_name}.npy"
    np.save(npy_out, node_features)
    
    # C. Node Map (CRITICAL FOR TEAMMATE AND INFERENCE)
    # We save both forward and inverse mappings
    map_out = out_path / "node_map.json"
    
    # Create inverse mappings for inference (TGN ID -> Raw ID)
    inverse_user_map = {str(v): k for k, v in user_map.items()}
    inverse_item_map = {str(v): k for k, v in item_map.items()}
    
    mapping_data = {
        "num_users": num_users,
        "num_items": num_items,
        "total_nodes": total_nodes + 1,  # Including padding at index 0
        "user_id_range": [1, num_users],  # Inclusive: users are at indices 1 to N_u
        "item_id_range": [num_users + 1, num_users + num_items],  # Items at N_u+1 to N_u+N_i
        "padding_idx": 0,  # CRITICAL: Index 0 is reserved for padding
        "user_map": user_map,           # Raw ID -> TGN ID (1-based)
        "item_map": item_map,           # Raw ID -> TGN ID (1-based)
        "inverse_user_map": inverse_user_map,  # TGN ID -> Raw ID
        "inverse_item_map": inverse_item_map,  # TGN ID -> Raw ID
        "feature_dim": int(combined_dim),
        "text_model": text_model,
        "image_model": image_model
    }
    with open(map_out, "w") as f:
        json.dump(mapping_data, f, indent=4)
        
    print("✅ Formatting Complete.")
    print(f"   -> Edge List:  {csv_out}")
    print(f"   -> Node Feats: {npy_out}")
    print(f"   -> Node Map:   {map_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings-path", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--text-model", default="baseline")
    parser.add_argument("--image-model", default="clip")
    args = parser.parse_args()
    
    format_dataset(
        args.ratings_path, 
        args.features_dir, 
        args.dataset_name, 
        args.text_model, 
        args.image_model, 
        args.output_dir
    )
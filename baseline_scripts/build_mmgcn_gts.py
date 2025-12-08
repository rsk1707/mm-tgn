#!/usr/bin/env python3
"""
Build MMGCN input files for MovieLens-modern using 70-15-15 GTS splits.

Outputs under:

  mmgcn/Data/ml-modern-gts/
    train.npy            # [N_edges, 2] with (user, global_item)
    user_item_dict.npy   # dict[user] -> set(global_item)
    val_full.npy         # object array of [user, pos1, pos2, ...]
    test_full.npy        # same format for test
    FeatureVideo_normal.npy  # image features, aligned to item ids
    FeatureText_stl_normal.npy  # text features, aligned to item ids
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from baseline_scripts.data_loader import load_gts_dataset

def main():
    dataset_name = "amazon-sports"
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name=dataset_name)
    num_users = dataset.num_users
    num_items = dataset.num_items

    out_dir = os.path.join(PROJECT_ROOT, "mmgcn", "Data", "amazon-sports-gts")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Train edges (user, global_item = num_users + item)
    train_edges = []
    user_item_dict = {u: set() for u in range(num_users)}

    for inter in dataset.splits["train"]:
        u = inter.user
        g_item = num_users + inter.item
        train_edges.append([u, g_item])
        user_item_dict[u].add(g_item)

    train_edges = np.array(train_edges, dtype=np.int64)
    np.save(os.path.join(out_dir, "train.npy"), train_edges)
    np.save(os.path.join(out_dir, "user_item_dict.npy"), user_item_dict)

    # 2) Val / test full arrays: [user, global_pos1, global_pos2, ...]
    def build_full_split(user_split_dict):
        rows = []
        for u, items in user_split_dict.items():
            if not items:
                continue
            g_items = [num_users + i for i in items]
            rows.append(np.array([u] + g_items, dtype=np.int64))
        return np.array(rows, dtype=object)

    val_full = build_full_split(dataset.user_val_items)
    test_full = build_full_split(dataset.user_test_items)

    np.save(os.path.join(out_dir, "val_full.npy"), val_full)
    np.save(os.path.join(out_dir, "test_full.npy"), test_full)

    # 3) Multimodal features aligned to GTS item order.
    #   features/sota/ml-modern_ids.npy           (raw movieIds)
    #   features/sota/ml-modern_image_clip.npy   (N_items, Dv)
    #   features/sota/ml-modern_text_baseline.npy (N_items, Dt)
    feat_root = os.path.join(PROJECT_ROOT, "features", "sota")

    ids_path = os.path.join(feat_root, "ml-modern_ids.npy")
    img_path = os.path.join(feat_root, "ml-modern_image_siglip.npy")
    txt_path = os.path.join(feat_root, "ml-modern_text_efficient.npy")

    raw_ids = np.load(ids_path)        # [N_items_raw]
    img_feats = np.load(img_path)      # [N_items_raw, Dv]
    txt_feats = np.load(txt_path)      # [N_items_raw, Dt]

    # Normalize raw_ids to strings for consistent dictionary keys
    raw_ids_str = [str(x) for x in raw_ids]
    raw2row = {rid: i for i, rid in enumerate(raw_ids_str)}

    Dv = img_feats.shape[1]
    Dt = txt_feats.shape[1]
    img_aligned = np.zeros((num_items, Dv), dtype=np.float32)
    txt_aligned = np.zeros((num_items, Dt), dtype=np.float32)

    missing_count = 0

    for internal_id, raw_item in enumerate(dataset.id2item):
        key = str(raw_item)
        if key not in raw2row:
            # This item has no corresponding feature row; leave as zeros
            print(f"[WARN] item {raw_item} not found in ml-modern_ids.npy; using zero features")
            missing_count += 1
            continue

        row_idx = raw2row[key]
        img_aligned[internal_id] = img_feats[row_idx]
        txt_aligned[internal_id] = txt_feats[row_idx]

    print(f"Aligned features for {num_items - missing_count} items, "
        f"{missing_count} items had no feature rows (left as zeros).")

    # MMGCN expects item features only (no user rows). It will internally
    # concatenate user preferences in GCN, so we just save item features.
    np.save(os.path.join(out_dir, "FeatureVideo_normal.npy"), img_aligned)
    np.save(os.path.join(out_dir, "FeatureText_stl_normal.npy"), txt_aligned)

    print("Wrote MMGCN GTS data to:", out_dir)
    print("  users:", num_users, "items:", num_items)
    print("  train.npy shape:", train_edges.shape)
    print("  val_full rows:", len(val_full), "test_full rows:", len(test_full))

if __name__ == "__main__":
    main()
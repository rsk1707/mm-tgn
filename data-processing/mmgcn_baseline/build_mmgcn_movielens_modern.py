#!/usr/bin/env python3
"""
Build MMGCN input files from MovieLens-modern data.

Assumes repo layout:

  <ROOT>/
    movielens-modern/ml-modern/ratings.csv
    features/baseline/ml-modern_ids.npy
    features/baseline/ml-modern_image_clip.npy
    features/baseline/ml-modern_text_baseline.npy
    Dataset.py
    main.py (MMGCN)
    Data/
      movielens/      <-- this script will create/populate

It will create:

  Data/movielens/train.npy
  Data/movielens/user_item_dict.npy
  Data/movielens/val_full.npy
  Data/movielens/test_full.npy
  Data/movielens/FeatureVideo_normal.npy
  Data/movielens/FeatureText_stl_normal.npy
  Data/movielens/FeatureAudio_avg_normal.npy  (dummy zeros)

Splitting per user (time-ordered):
  - Train: all interactions except last 2
  - Val:   second-last interaction
  - Test:  last interaction

Users with < 3 interactions in the modern slice are dropped.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_USER_INTERACTIONS = 3   # min interactions to keep a user
RATING_THRESHOLD = None     # e.g., 4.0 to keep only strong positives; None = keep all

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]  # <ROOT>/data-processing/mmgcn -> parents[2] == <ROOT>

RATINGS_CSV = ROOT / "movielens-modern" / "ml-modern" / "ratings.csv"
MODERN_IDS_NPY = ROOT / "features" / "baseline" / "ml-modern_ids.npy"
IMAGE_NPY = ROOT / "features" / "baseline" / "ml-modern_image_clip.npy"
TEXT_NPY = ROOT / "features" / "baseline" / "ml-modern_text_baseline.npy"

MMGCN_DATA_DIR = ROOT / "Data" / "movielens"  # reuse 'movielens' dataset name


def main():
    if not RATINGS_CSV.exists():
        raise FileNotFoundError(f"ratings.csv not found at: {RATINGS_CSV}")
    if not MODERN_IDS_NPY.exists():
        raise FileNotFoundError(f"ml-modern_ids.npy not found at: {MODERN_IDS_NPY}")
    if not IMAGE_NPY.exists():
        raise FileNotFoundError(f"ml-modern_image_clip.npy not found at: {IMAGE_NPY}")
    if not TEXT_NPY.exists():
        raise FileNotFoundError(f"ml-modern_text_baseline.npy not found at: {TEXT_NPY}")

    MMGCN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    modern_ids = np.load(MODERN_IDS_NPY).astype(int)
    num_items = len(modern_ids)

    # local item index: 0...num_items-1
    movieid_to_item_local = {int(mid): i for i, mid in enumerate(modern_ids)}
    modern_id_set = set(movieid_to_item_local.keys())

    print(f"[INFO] Loaded {num_items} modern item IDs from {MODERN_IDS_NPY}")

    # events: map raw_user_id -> list of (timestamp, item_local)
    user_events = defaultdict(list)
    num_raw_rows = 0
    num_kept_rows = 0

    with RATINGS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"userId", "movieId", "rating", "timestamp"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"ratings.csv is missing expected columns: {missing}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            num_raw_rows += 1
            try:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"]) if row["movieId"] is not None else None
                rating = float(row["rating"]) if row["rating"] is not None else None
                timestamp = int(row["timestamp"]) if row["timestamp"] is not None else None
            except Exception:
                # skip malformed rows
                continue

            if movie_id is None or timestamp is None:
                continue

            # filter to modern subset
            if movie_id not in modern_id_set:
                continue

            if RATING_THRESHOLD is not None and (rating is None or rating < RATING_THRESHOLD):
                continue

            item_local = movieid_to_item_local[movie_id]
            user_events[user_id].append((timestamp, item_local))
            num_kept_rows += 1

    print(f"Read {num_raw_rows} rows from ratings.csv")
    print(f"Kept {num_kept_rows} interactions in modern subset")

    filtered = {}
    for raw_uid, events in user_events.items():
        if len(events) < MIN_USER_INTERACTIONS:
            continue
        events_sorted = sorted(events, key=lambda x: x[0])
        filtered[raw_uid] = events_sorted

    print(
        f"Users before filtering: {len(user_events)}, "
        f"after filtering: {len(filtered)} (min interactions = {MIN_USER_INTERACTIONS})"
    )

    # MMGCN convention:
    #   users: 0 .. num_user-1
    #   items (global): num_user .. num_user+num_item-1
    #
    # We'll first create per-user sequences of *local* item ids,
    # then map to global ids once we know num_user.
    user_seq_local = []   # list of (new_uid, [item_local sequence])
    raw_to_new_uid = {}
    num_users = 0

    for raw_uid, events in filtered.items():
        items_local = [it for _, it in events]  # drop timestamps
        if len(items_local) < 3:
            continue  # safety, though we already enforced >=3
        new_uid = num_users
        raw_to_new_uid[raw_uid] = new_uid
        user_seq_local.append((new_uid, items_local))
        num_users += 1

    print(f"Final num_users: {num_users}")
    print(f"num_items (from modern_ids): {num_items}")

    train_edges = []        # rows: [user_global, item_global]
    user_item_dict = {}     # user_global -> set(item_global)
    val_full_entries = []   # each: np.array([user_global, pos1, pos2, ...])
    test_full_entries = []

    for u_global, items_local in user_seq_local:
        T = len(items_local)
        assert T >= 3

        train_locals = items_local[:-2]
        val_locals = [items_local[-2]]
        test_locals = [items_local[-1]]

        if not train_locals:
            # Extremely short sequences (T == 2 or 1) should have been filtered already
            continue

        # Build train edges & user_item_dict
        pos_globals = set()
        for il in train_locals:
            ig = num_users + il  # global item id
            train_edges.append([u_global, ig])
            pos_globals.add(ig)

        user_item_dict[u_global] = pos_globals

        # Build val_full entry
        val_globals = [num_users + il for il in val_locals]
        val_full_entries.append(np.array([u_global] + val_globals, dtype=np.int64))

        # Build test_full entry
        test_globals = [num_users + il for il in test_locals]
        test_full_entries.append(np.array([u_global] + test_globals, dtype=np.int64))

    train_edges = np.array(train_edges, dtype=np.int64)
    np.save(MMGCN_DATA_DIR / "train.npy", train_edges)
    np.save(MMGCN_DATA_DIR / "user_item_dict.npy", user_item_dict)
    np.save(MMGCN_DATA_DIR / "val_full.npy", np.array(val_full_entries, dtype=object))
    np.save(MMGCN_DATA_DIR / "test_full.npy", np.array(test_full_entries, dtype=object))

    print(f"Saved train.npy with shape {train_edges.shape}")
    print(f"Saved user_item_dict.npy with {len(user_item_dict)} users")
    print(f"Saved val_full.npy with {len(val_full_entries)} users")
    print(f"Saved test_full.npy with {len(test_full_entries)} users")

    v_feat = np.load(IMAGE_NPY)  # (num_items, d_v)
    t_feat = np.load(TEXT_NPY)   # (num_items, d_t)
    if v_feat.shape[0] != num_items or t_feat.shape[0] != num_items:
        raise ValueError(
            f"Feature rows do not match ml-modern_ids length.\n"
            f"  ids: {num_items}, image rows: {v_feat.shape[0]}, text rows: {t_feat.shape[0]}"
        )

    # Visual features
    np.save(MMGCN_DATA_DIR / "FeatureVideo_normal.npy", v_feat)
    # Text features
    np.save(MMGCN_DATA_DIR / "FeatureText_stl_normal.npy", t_feat)
    # Dummy audio (zeros) - safe even if you pass --has_a False
    a_feat = np.zeros_like(v_feat, dtype=np.float32)
    np.save(MMGCN_DATA_DIR / "FeatureAudio_avg_normal.npy", a_feat)

    print(f"Saved FeatureVideo_normal.npy with shape {v_feat.shape}")
    print(f"Saved FeatureText_stl_normal.npy with shape {t_feat.shape}")
    print(f"Saved FeatureAudio_avg_normal.npy with shape {a_feat.shape}")

    possible_pairs = num_users * num_items
    num_interactions = train_edges.shape[0] + len(val_full_entries) + len(test_full_entries)
    sparsity = 1.0 - num_interactions / float(possible_pairs) if possible_pairs > 0 else 1.0

    print(f"num_users: {num_users}")
    print(f"num_items: {num_items}")
    print(f"total interactions (train+val+test): {num_interactions}")
    print(f"sparsity: {sparsity:.6f}")
    print("IMPORTANT: Open Dataset.py and in the 'movielens' branch,")
    print(f"  set num_user = {num_users}")
    print(f"  set num_item = {num_items}")
    print("Then run MMGCN with --data_path movielens.")


if __name__ == "__main__":
    main()

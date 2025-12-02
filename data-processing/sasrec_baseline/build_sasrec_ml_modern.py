#!/usr/bin/env python3
"""
Build SASRec input files for MovieLens-modern.

Assumes repo layout:

  <ROOT>/
    movielens-modern/ml-modern/ratings.csv
    features/baseline/ml-modern_ids.npy
    features/baseline/ml-modern_image_clip.npy
    features/baseline/ml-modern_text_baseline.npy
    sasrec/
      main.py, util.py, sampler.py, modules.py, model.py, ...
    data-processing/sasrec/build_sasrec_ml_modern.py  (this file)

Outputs (depending on --variant):

  <ROOT>/sasrec/data/ml-modern.txt
  <ROOT>/sasrec/data/ml-modern_item_mm_init.npy  (for multimodal variant)

Usage examples:

  # ID-only SASRec (just the interaction file)
  python data-processing/sasrec/build_sasrec_ml_modern.py --variant id

  # Multimodal init only (requires that ml-modern.txt already exists)
  python data-processing/sasrec/build_sasrec_ml_modern.py --variant mm

  # Do both in one shot
  python data-processing/sasrec/build_sasrec_ml_modern.py --variant both
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# Minimum interactions per user to keep them in the dataset.
MIN_USER_INTERACTIONS = 3

# Optional: if you want to treat only ratings >= threshold as interactions,
# set this to something like 3.5 or 4.0. Set to None to use all ratings.
RATING_THRESHOLD = None 


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2]  # <ROOT>/data-processing/sasrec -> parents[2] == <ROOT>

RATINGS_CSV = ROOT / "movielens-modern" / "ml-modern" / "ratings.csv"
MODERN_IDS_NPY = ROOT / "features" / "baseline" / "ml-modern_ids.npy"
IMAGE_NPY = ROOT / "features" / "baseline" / "ml-modern_image_clip.npy"
TEXT_NPY = ROOT / "features" / "baseline" / "ml-modern_text_baseline.npy"

SASREC_DATA_DIR = ROOT / "sasrec" / "data"
OUTPUT_TXT = SASREC_DATA_DIR / "ml-modern.txt"
OUTPUT_MM_ITEM_EMB = SASREC_DATA_DIR / "ml-modern_item_mm_init.npy"

def build_id_only_files():
    if not RATINGS_CSV.exists():
        raise FileNotFoundError(f"ratings.csv not found at: {RATINGS_CSV}")

    if not MODERN_IDS_NPY.exists():
        raise FileNotFoundError(f"ml-modern_ids.npy not found at: {MODERN_IDS_NPY}")

    SASREC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # movieId -> item_idx mapping
    modern_ids = np.load(MODERN_IDS_NPY).astype(int)
    num_items = len(modern_ids)

    movieid_to_itemidx = {int(mid): i + 1 for i, mid in enumerate(modern_ids)}
    modern_id_set = set(movieid_to_itemidx.keys())

    print(f"Loaded {num_items} modern item IDs from {MODERN_IDS_NPY}")

    user_events = defaultdict(list)
    num_raw_rows = 0
    num_kept_rows = 0

    # aggregating per user events
    with RATINGS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        expected_cols = {"userId", "movieId", "rating", "timestamp"}
        missing = expected_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"ratings.csv is missing expected columns: {missing}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            num_raw_rows += 1
            try:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])
                rating = float(row["rating"])
                timestamp = int(row["timestamp"])
            except Exception as e:
                print(f"[WARN] Skipping malformed row {num_raw_rows}: {row} ({e})")
                continue

            # Filter movies to modern subset
            if movie_id not in modern_id_set:
                continue

            # An optional rating threshold, probs wont need it
            if RATING_THRESHOLD is not None and rating < RATING_THRESHOLD:
                continue

            item_idx = movieid_to_itemidx[movie_id]
            user_events[user_id].append((timestamp, item_idx))
            num_kept_rows += 1

    print(f"Read {num_raw_rows} rows from ratings.csv")
    print(f"Kept {num_kept_rows} interactions in modern subset")

    # MAke sure the users are within limit and then sort the interactions by time
    filtered_user_events = {}
    for raw_uid, events in user_events.items():
        if len(events) < MIN_USER_INTERACTIONS:
            continue
        events_sorted = sorted(events, key=lambda x: x[0])
        filtered_user_events[raw_uid] = events_sorted

    print(
        f"Users before filtering: {len(user_events)}, "
        f"after filtering: {len(filtered_user_events)} "
        f"(min interactions per user = {MIN_USER_INTERACTIONS})"
    )

    # Reindex users and write SASRec interaction file
    num_users = 0
    num_interactions_out = 0

    with OUTPUT_TXT.open("w") as out_f:
        for raw_uid, events in filtered_user_events.items():
            num_users += 1
            new_uid = num_users
            for ts, item_idx in events:
                out_f.write(f"{new_uid} {item_idx}\n")
                num_interactions_out += 1

    print(f"Wrote SASRec data file to: {OUTPUT_TXT}")
    print(f"num_users: {num_users}")
    print(f"num_items (from modern_ids): {num_items}")
    print(f"num_interactions: {num_interactions_out}")

    if num_users > 0 and num_items > 0:
        sparsity = 1.0 - num_interactions_out / float(num_users * num_items)
        print(f"sparsity: {sparsity:.6f}")

    return modern_ids


def build_multimodal_item_init(existing_modern_ids=None):
    """
    Create ml-modern_item_mm_init.npy with shape (num_items + 1, D),
    where row 0 is zeros and rows 1..num_items are [text_emb; image_emb].

    We assume:
      - ml-modern_ids.npy defines the item order
      - image and text .npy files are in the same order and shape (N, 768)
    """

    if not MODERN_IDS_NPY.exists():
        raise FileNotFoundError(f"ml-modern_ids.npy not found at: {MODERN_IDS_NPY}")
    if not IMAGE_NPY.exists():
        raise FileNotFoundError(f"ml-modern_image_clip.npy not found at: {IMAGE_NPY}")
    if not TEXT_NPY.exists():
        raise FileNotFoundError(f"ml-modern_text_baseline.npy not found at: {TEXT_NPY}")

    SASREC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if existing_modern_ids is None:
        modern_ids = np.load(MODERN_IDS_NPY).astype(int)
    else:
        modern_ids = existing_modern_ids
    num_items = len(modern_ids)

    img_emb = np.load(IMAGE_NPY)
    txt_emb = np.load(TEXT_NPY)

    if img_emb.shape[0] != num_items or txt_emb.shape[0] != num_items:
        raise ValueError(
            f"Embedding row counts do not match ml-modern_ids length.\n"
            f"  ids: {num_items}, image rows: {img_emb.shape[0]}, "
            f"text rows: {txt_emb.shape[0]}"
        )

    if img_emb.shape[1] != 768 or txt_emb.shape[1] != 768:
        print(
            f"[MM][WARN] Expected 768-dim embeddings but got "
            f"image_dim={img_emb.shape[1]}, text_dim={txt_emb.shape[1]}"
        )

    fused = np.concatenate([txt_emb, img_emb], axis=1)  # shape (N, 1536)
    dim = fused.shape[1]

    item_mm_init = np.zeros((num_items + 1, dim), dtype=fused.dtype)
    item_mm_init[1:, :] = fused

    np.save(OUTPUT_MM_ITEM_EMB, item_mm_init)
    print(
        f"[MM] Saved multimodal item init to: {OUTPUT_MM_ITEM_EMB} "
        f"with shape {item_mm_init.shape}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SASRec MovieLens-modern inputs (ID-only and/or multimodal)."
    )
    parser.add_argument(
        "--variant",
        choices=["id", "mm", "both"],
        default="id",
        help=(
            "'id'   : build ml-modern.txt only\n"
            "'mm'   : build multimodal item init only\n"
            "'both' : build both ml-modern.txt and multimodal init"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    modern_ids_cache = None

    if args.variant in ("id", "both"):
        modern_ids_cache = build_id_only_files()

    if args.variant in ("mm", "both"):
        build_multimodal_item_init(existing_modern_ids=modern_ids_cache)


if __name__ == "__main__":
    main()
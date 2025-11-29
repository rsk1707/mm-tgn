#!/usr/bin/env python3
"""
Build LightGCN dataset (train.txt, test.txt) from MovieLens ratings.csv
and a modern_ids subset.

Assumptions:
- We already created ml-modern_ids.npy (movieIds, e.g., year > 2018).
- We want to use ONLY interactions whose movieId is in ml-modern_ids.npy.
- We want a temporal split:
    * For each user:
        - sort interactions by timestamp ascending
        - if len(items) >= 2:
              train = all but last
              test  = [last]
          else:
              train = all
              (no test entry for that user)

Output:
- {output_dir}/train.txt
- {output_dir}/test.txt

Each line matches what LightGCN's Data class expects:
    user_internal_id item1 item2 ...

where user_internal_id and item ids are 0-based contiguous integers.
"""

import argparse
import os
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ratings_path",
        type=str,
        required=True,
        help="Path to MovieLens ratings.csv",
    )
    parser.add_argument(
        "--modern_ids_path",
        type=str,
        required=True,
        help="Path to ml-modern_ids.npy (MovieLens movieIds for modern subset).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write train.txt and test.txt (e.g., ./Data/ml_modern).",
    )
    parser.add_argument(
        "--min_items_per_user",
        type=int,
        default=1,
        help="Drop users with fewer total interactions than this AFTER filtering to modern_ids.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.txt")
    test_path = os.path.join(args.output_dir, "test.txt")

    print(f"Loading modern_ids from {args.modern_ids_path} ...")
    modern_ids = np.load(args.modern_ids_path).astype(int)
    n_items = len(modern_ids)
    print(f"  Found {n_items} modern items.")

    movieid_to_idx = {int(mid): i for i, mid in enumerate(modern_ids)}

    print(f"Loading ratings from {args.ratings_path} ...")
    ratings = pd.read_csv(args.ratings_path)

    # Basic checks
    required_cols = {"userId", "movieId", "timestamp"}
    if not required_cols.issubset(ratings.columns):
        raise ValueError(
            f"ratings.csv must contain columns: {required_cols}, "
            f"got: {list(ratings.columns)}"
        )

    # Filter to interactions where movieId is in modern_ids
    before = len(ratings)
    ratings = ratings[ratings["movieId"].isin(movieid_to_idx.keys())].copy()
    after = len(ratings)
    print(f"Filtered ratings: {before} -> {after} rows with modern movieIds.")

    if after == 0:
        raise ValueError("No interactions left after filtering to modern_ids!")

    # Map MovieLens userId → internal [0..n_users)
    unique_users = np.sort(ratings["userId"].unique())
    userId_to_idx = {int(u): i for i, u in enumerate(unique_users)}
    n_users = len(unique_users)
    print(f"Found {n_users} unique users after filtering.")

    # Apply mappings
    ratings["user_idx"] = ratings["userId"].map(userId_to_idx)
    ratings["item_idx"] = ratings["movieId"].map(movieid_to_idx)

    # Drop any rows that didn't map (shouldn't happen if keys align)
    ratings = ratings.dropna(subset=["user_idx", "item_idx"])
    ratings["user_idx"] = ratings["user_idx"].astype(int)
    ratings["item_idx"] = ratings["item_idx"].astype(int)

    # Group by internal user id
    grouped = ratings.groupby("user_idx")

    train_lines = []
    test_lines = []

    num_users_train = 0
    num_users_test = 0

    print("Building per-user train/test splits (LOO)...")
    for u_idx, df_u in grouped:
        # Sort by timestamp ascending
        df_u_sorted = df_u.sort_values("timestamp")
        items = df_u_sorted["item_idx"].tolist()

        # Drop users with too few interactions
        if len(items) < args.min_items_per_user:
            continue

        if len(items) >= 2:
            train_items = items[:-1]
            test_items = [items[-1]]
        else:
            train_items = items
            test_items = []

        if len(train_items) > 0:
            line = " ".join([str(u_idx)] + [str(i) for i in train_items])
            train_lines.append(line)
            num_users_train += 1

        if len(test_items) > 0:
            line = " ".join([str(u_idx)] + [str(i) for i in test_items])
            test_lines.append(line)
            num_users_test += 1

    print(f"Users with train interactions: {num_users_train}")
    print(f"Users with test interactions:  {num_users_test}")

    if num_users_train == 0:
        raise ValueError("No users with train interactions after filtering/splitting!")

    # Write train.txt and test.txt
    print(f"Writing train file to {train_path}")
    with open(train_path, "w") as f_train:
        for line in train_lines:
            f_train.write(line + "\n")

    print(f"Writing test file to {test_path}")
    with open(test_path, "w") as f_test:
        for line in test_lines:
            f_test.write(line + "\n")

    print("Done.")
    print(f"Train lines: {len(train_lines)}, Test lines: {len(test_lines)}")
    print(f"n_users = {n_users}, n_items = {n_items}")


if __name__ == "__main__":
    main()


# Output
# python build_lightgcn_dataset.py \
#   --ratings_path ../../ml-modern/ratings.csv \
#   --modern_ids_path ../../features/ml-modern_ids.npy \
#   --output_dir ../../lightgcn/Data/ml_modern
# Loading modern_ids from ../../features/ml-modern_ids.npy ...
#   Found 21651 modern items.
# Loading ratings from ../../ml-modern/ratings.csv ...
# Filtered ratings: 9576 -> 9536 rows with modern movieIds.
# Found 759 unique users after filtering.
# Building per-user train/test splits (LOO)...
# Users with train interactions: 759
# Users with test interactions:  477
# Writing train file to ../../lightgcn/Data/ml_modern/train.txt
# Writing test file to ../../lightgcn/Data/ml_modern/test.txt
# Done.
# Train lines: 759, Test lines: 477
# n_users = 759, n_items = 21651
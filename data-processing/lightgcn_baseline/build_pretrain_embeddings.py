#!/usr/bin/env python3
"""
Build pretrained user/item embeddings for LightGCN from multimodal MovieLens features.

Pipeline assumption:
- You already ran `build_lightgcn_dataset.py` to produce:
    Data/{dataset}/train.txt
    Data/{dataset}/test.txt
- You have:
    ml-modern_ids.npy              (movieIds for the filtered modern subset)
    ml-modern_text_baseline.npy    (text embeddings, shape: [n_items, d_text])
    ml-modern_image_clip.npy       (image embeddings, shape: [n_items, d_image])

This script:
- Reads train.txt to get n_users.
- Uses TEXT+IMAGE concatenation as item embeddings.
- Initializes user embeddings randomly with matching dimension.
- Writes:
    {proj_path}/pretrain/{dataset}/embedding.npz

which LightGCN.py expects when args.pretrain = -1.
"""

import argparse
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    # Paths for multimodal item features
    parser.add_argument(
        "--modern_ids_path",
        type=str,
        required=True,
        help="Path to ml-modern_ids.npy (MovieLens movieIds for modern subset).",
    )
    parser.add_argument(
        "--text_emb_path",
        type=str,
        required=True,
        help="Path to ml-modern_text_baseline.npy (shape: [n_items, d_text]).",
    )
    parser.add_argument(
        "--image_emb_path",
        type=str,
        required=True,
        help="Path to ml-modern_image_clip.npy (shape: [n_items, d_image]).",
    )

    # Where the LightGCN Data lives (to read train.txt and get n_users)
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory that contains {dataset}/train.txt and test.txt (e.g., ./Data).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (subfolder under data_dir, e.g., ml_modern).",
    )

    # Where LightGCN expects the pretrain file
    parser.add_argument(
        "--proj_path",
        type=str,
        required=True,
        help="Project root used in LightGCN args.proj_path (e.g., ./).",
    )

    # Random user embedding options
    parser.add_argument(
        "--user_emb_std",
        type=float,
        default=0.01,
        help="Std dev for random user embeddings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def get_num_users_from_train(train_path: str) -> int:
    """
    Infer n_users from train.txt:
        each line: user_id item1 item2 ...
    user ids are assumed to be 0-based contiguous integers.
    """
    max_uid = -1
    with open(train_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            uid = int(parts[0])
            if uid > max_uid:
                max_uid = uid
    if max_uid < 0:
        raise ValueError(f"No valid lines found in train file: {train_path}")
    return max_uid + 1


def main():
    args = parse_args()

    # 1. Load multimodal item embeddings
    print(f"Loading modern_ids from {args.modern_ids_path} ...")
    modern_ids = np.load(args.modern_ids_path)
    modern_ids = modern_ids.astype(int)
    n_items = modern_ids.shape[0]
    print(f"  modern_ids: {n_items} items")

    print(f"Loading text embeddings from {args.text_emb_path} ...")
    text_embs = np.load(args.text_emb_path)
    print(f"  text_embs.shape = {text_embs.shape}")

    print(f"Loading image embeddings from {args.image_emb_path} ...")
    img_embs = np.load(args.image_emb_path)
    print(f"  img_embs.shape = {img_embs.shape}")

    # Sanity checks
    if text_embs.shape[0] != n_items or img_embs.shape[0] != n_items:
        raise ValueError(
            f"Item count mismatch: modern_ids={n_items}, "
            f"text_embs={text_embs.shape[0]}, img_embs={img_embs.shape[0]}"
        )

    # Concatenate [TEXT; IMAGE]
    item_embed = np.concatenate([text_embs, img_embs], axis=1).astype(np.float32)
    emb_dim = item_embed.shape[1]
    print(f"Built item_embed with shape: {item_embed.shape} (emb_dim = {emb_dim})")

    # 2. Infer n_users from train.txt
    train_path = os.path.join(args.data_dir, args.dataset, "train.txt")
    print(f"Reading train file from {train_path} to infer n_users ...")
    n_users = get_num_users_from_train(train_path)
    print(f"  Inferred n_users = {n_users}")

    # 3. Build random user embeddings
    print("Building random user embeddings ...")
    rng = np.random.default_rng(seed=args.seed)
    user_embed = rng.normal(
        loc=0.0,
        scale=args.user_emb_std,
        size=(n_users, emb_dim),
    ).astype(np.float32)
    print(f"user_embed.shape = {user_embed.shape}")

    # 4. Save embedding.npz where LightGCN expects it
    out_dir = os.path.join(args.proj_path, "pretrain", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "embedding.npz")

    print(f"Saving embeddings to: {out_path}")
    np.savez(out_path, user_embed=user_embed, item_embed=item_embed)
    print("Done. In LightGCN, set:")
    print(f"  --proj_path {args.proj_path}")
    print(f"  --dataset {args.dataset}")
    print(f"  --embed_size {emb_dim}")
    print("and use --pretrain -1 to load these embeddings.")


if __name__ == "__main__":
    main()
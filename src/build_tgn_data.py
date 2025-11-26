#!/usr/bin/env python3
"""
Personal Notes:
Assumes each line looks like:
u,i,ts,label,f0,f1,f2,...,fK

where:
u = source node ID
i = destination node ID
ts = timestamp
label = target/label (e.g. 1 for the TGN paper) --> supervision for model. Positive interactions obviously appear in dataset but sometimes we want negative interactions so we can sample that. This is
        helpful for link prediction and stuff, but generated during training

f0..fK = edge features (values for each feature or context of this interaction)

Each interaction in original TGN has strong semantic context for the interactions. They don't represent the node though. Our real signal will be in movie embeddings and such. Nodes will carry the rich signal from FiLM
Edges just represent a sparse interaction stream

Preprocess.py:
u, i, ts, label, idx where idx is just a autoincrement edge index
Reindexes nodes where IDs are 1-based for users and items separately. In --bipartite mode it keeps users in [1..n_users] shift item IDs to [n_users + 1, n_users + n_items]. Make sure user and item node IDs don't overlap

Saves three files:
movielens-data/ml_<name>.csv → processed edges (u, i, ts, label, idx)
movielens-data/ml_<name>.npy → edge feature matrix (one row per edge, plus a dummy zero at index 0) --> (num_edges + 1, edge_feature_dim)
movielens-data/ml_<name>_node.npy → node feature matrix, but since the example datasets don’t have node features, they just create: `rand_feat = np.zeros((max_idx + 1, 172))`

We don't care about rich edge features right now, but we will have rich node features from FiLM
Information -> on movie nodes (poster + overview + genres -> FiLM embedding)
Edges -> rating

Inputs:
- ratings.csv: userId,movieId,rating,timestamp
- embeddings.csv: movieId,e0,e1,...,eD

Output:
- ./movielens-data/ml_movielens_mm.csv: u,i,ts,label,idx
- ./movielens-data/ml_movielens_mm.npy: shape of (num_edges + 1, 1) and each edge feature = [rating]
- ./movielens-data/ml_movielens_mm_node.npy: shape of (max_node_id + 1, embed_dim)

    row 0: dummy zero row
    rows 1..num_users: user features (zeros)
    rows num_users+1..num_users+num_items: movie FiLM embeddings 
"""

#!/usr/bin/env python3
"""
Build TGN-ready data for MovieLens + FiLM embeddings.

Inputs:
  - ratings.csv with columns:
      userId,movieId,rating,timestamp

  - embeddings.csv (FiLM output) with columns:
      movieId,e0,e1,...,eD

Outputs (for dataset-name = "movielens_mm"):
  - ./movielens-data/ml_movielens_mm.csv
      columns: u,i,ts,label,idx
  - ./movielens-data/ml_movielens_mm.npy
      shape: (num_edges + 1, 1)         # edge feature = [rating]
  - ./movielens-data/ml_movielens_mm_node.npy
      shape: (max_node_id + 1, embed_dim)
      row 0: dummy zero row
      rows 1..num_users: user features (zeros)
      rows num_users+1..num_users+num_items: movie FiLM embeddings
"""

import argparse
import os

import numpy as np
import pandas as pd


def build_id_maps(ratings, user_col, item_col):
    """
    Build a contiguous 0-based ID maps for users and items using a ratings pd.dataframe

    user_id_map: raw_user_id -> internal_user_idx (0..num_users-1)
    item_id_map: raw_movie_id -> internal_item_idx (0..num_items-1)

    Returns: (user_id_map, item_id_map)
    """
    unique_users = sorted(ratings[user_col].unique())
    unique_items = sorted(ratings[item_col].unique())

    user_id_map = {}
    item_id_map = {}
    for idx, uid in enumerate(unique_users):
        user_id_map[int(uid)] = idx
    for idx, mid in enumerate(unique_items):
        item_id_map[int(mid)] = idx

    return user_id_map, item_id_map


def build_edge_dataframe(ratings, user_id_map, item_id_map, user_col, item_col, rating_col, time_col):
    """
    Build the edge/event dataframe with columns: u,i,ts,label,idx

    its going to be bipartite node indexing and rating as edge feature (returned separately).
    """

    df = ratings[[user_col, item_col, rating_col, time_col]].copy()

    # We built our maps for user and item, so now map raw IDs to those `0-based` internal indices
    df["u_int"] = df[user_col].astype(int).map(user_id_map)
    df["i_int"] = df[item_col].astype(int).map(item_id_map)

    if df["u_int"].isna().any() or df["i_int"].isna().any():
        raise ValueError("Found user/movie IDs in ratings that are not in ID maps.")

    num_users = len(user_id_map)
    # Bipartite reindexing, 1-based:
    # users: 1..num_users
    # items: num_users+1 .. num_users+num_items
    df["u"] = df["u_int"] + 1
    df["i"] = df["i_int"] + num_users + 1

    # Timestamp (assume already in unix seconds)
    df["ts"] = df[time_col].astype(float)

    # All observed edges will default to the positive class, TRAINING we should add the negative labels TODO: reminder
    df["label"] = 1.0

    # Sorting by time for to create the event stream
    df = df.sort_values("ts").reset_index(drop=True)

    # 1 base index it
    df["idx"] = np.arange(len(df), dtype=np.int64) + 1

    edge_ratings = df[rating_col].astype(float).to_numpy()

    events_df = df[["u", "i", "ts", "label", "idx"]].copy()
    return events_df, edge_ratings


def build_node_features(user_id_map, item_id_map, embeddings_df, embedding_id_col):
    """
    Build node feature Numpy matrix with shape (max_node_id + 1, embed_dim).

    row 0: dummy zero row
    rows 1..num_users: user nodes (we will keep this at zeros TODO: do we want to store any info for these?)
    rows num_users+1..num_users+num_items: movie nodes (FiLM embeddings)
    """
    num_users = len(user_id_map)
    num_items = len(item_id_map)

    embeddings_df = embeddings_df.copy()
    embeddings_df[embedding_id_col] = embeddings_df[embedding_id_col].astype(int)

    # All columns except the ID are embedding dims
    emb_cols = [c for c in embeddings_df.columns if c != embedding_id_col]
    if not emb_cols:
        raise ValueError("No embedding columns found in embeddings CSV.")

    embed_dim = len(emb_cols)

    # Total nodes = dummy row + users + items
    max_node_id = num_users + num_items  # because we start real nodes at 1
    node_features = np.zeros((max_node_id + 1, embed_dim), dtype=np.float32)

    # Build fast lookup: raw_movie_id -> embedding vector
    movie_emb_dict = {
        int(row[embedding_id_col]): row[emb_cols].to_numpy(dtype=np.float32)
        for _, row in embeddings_df.iterrows()
    }

    missing_movies = []

    for raw_movie_id, item_internal_idx in item_id_map.items():
        if raw_movie_id not in movie_emb_dict:
            missing_movies.append(raw_movie_id)
            continue

        emb_vec = movie_emb_dict[raw_movie_id]
        if emb_vec.shape[0] != embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch for movie {raw_movie_id}: "
                f"expected {embed_dim}, got {emb_vec.shape[0]}"
            )

        # Global node index for this movie in bipartite scheme:
        # users: 1..num_users
        # movies: num_users+1..num_users+num_items
        global_movie_node_id = num_users + 1 + item_internal_idx
        node_features[global_movie_node_id, :] = emb_vec

    if missing_movies:
        print(
            f"WARNING: Missing embeddings for {len(missing_movies)} movies; "
            "these movie node features remain zero."
        )

    # User rows (1..num_users) remain zeros by design for now.
    return node_features


def build_edge_features(edge_ratings):
    """
    Build edge feature matrix using rating as the only feature. 
    TODO: might need to change to build edge features differently

    Shape: (num_edges + 1, 1)
    row 0: dummy zero row
    rows 1..num_edges: [rating]
    """
    edge_ratings = edge_ratings.reshape(-1, 1).astype(np.float32) # (num_edges,) -> (num_edges, 1)
    dummy_row = np.zeros((1, 1), dtype=np.float32)
    edge_features = np.vstack([dummy_row, edge_ratings])
    return edge_features


def main():
    parser = argparse.ArgumentParser(
        description="Build TGN-ready MovieLens data with FiLM movie embeddings."
    )
    parser.add_argument("--ratings", type=str, required=True,
                        help="Path to ratings.csv (userId,movieId,rating,timestamp)")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to movie embedding CSV (movieId,e0,e1,...,eD)")
    parser.add_argument("--dataset-name", type=str, default="movielens_mm",
                        help="Name used in output files, e.g. ml_<dataset-name>.csv")
    parser.add_argument("--out-dir", type=str, default="../movielens-data",
                        help="Output directory (default: ../movielens-data)")

    # Column names (adding this bc we might want to apply this to another dataset besides MovieLens that have diff naming)
    parser.add_argument("--user-col", type=str, default="userId")
    parser.add_argument("--item-col", type=str, default="movieId")
    parser.add_argument("--rating-col", type=str, default="rating")
    parser.add_argument("--time-col", type=str, default="timestamp")
    parser.add_argument("--embedding-id-col", type=str, default="movieId")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ratings_path = args.ratings
    embeddings_path = args.embeddings

    out_csv = os.path.join(args.out_dir, f"ml_{args.dataset_name}.csv")
    out_edge_feat = os.path.join(args.out_dir, f"ml_{args.dataset_name}.npy")
    out_node_feat = os.path.join(args.out_dir, f"ml_{args.dataset_name}_node.npy")

    print(f"Loading ratings from {ratings_path}")
    ratings = pd.read_csv(ratings_path)

    print("Building user/movie ID mappings")
    user_id_map, item_id_map = build_id_maps(ratings, user_col=args.user_col, item_col=args.item_col)

    print("Building edge dataframe and edge ratings")
    events_df, edge_ratings = build_edge_dataframe(
        ratings,
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        time_col=args.time_col,
    )

    print(f"Saving edges to {out_csv}")
    events_df.to_csv(out_csv, index=False)

    print("Building edge feature matrix from ratings")
    edge_features = build_edge_features(edge_ratings)

    print(f"Edge feature matrix shape: {edge_features.shape}")
    np.save(out_edge_feat, edge_features)
    print(f"Saved edge features to {out_edge_feat}")

    print(f"Loading embeddings from {embeddings_path}")
    embeddings_df = pd.read_csv(embeddings_path)

    print("Building node feature matrix (movies = FiLM embeddings)")
    node_features = build_node_features(
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        embeddings_df=embeddings_df,
        embedding_id_col=args.embedding_id_col,
    )
    print(f"Node feature matrix shape: {node_features.shape}")
    np.save(out_node_feat, node_features)
    print(f"Saved node features to {out_node_feat}")

    print("=== DONE: TGN-ready files written: ===")
    print(f"  - {out_csv}")
    print(f"  - {out_edge_feat}")
    print(f"  - {out_node_feat}")

    """
    You can test out the output files by just running this in your terminal python3 -c "import numpy as np; print(np.load('ml_movielens_mm_test.npy'))"
    """


if __name__ == "__main__":
    main()



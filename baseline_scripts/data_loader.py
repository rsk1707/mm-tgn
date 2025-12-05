# baseline_scripts/data_loader.py

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Interaction:
    user: int
    item: int
    timestamp: int


@dataclass
class GTSDataset:
    num_users: int
    num_items: int
    user2id: dict          # raw user ID (string) -> internal int
    id2user: list          # internal -> raw user ID
    item2id: dict          # raw item ID (string) -> internal int
    id2item: list          # internal -> raw item ID
    splits: dict           # "train"/"val"/"test" -> list[Interaction]
    user_train_items: dict # internal user -> list of internal item ids
    user_val_items: dict
    user_test_items: dict
    user_all_pos_items: dict


def _read_split_file(split_dir, split_name):
    for ext in (".csv", ".xlsx", ".xls"):
        path = os.path.join(split_dir, split_name + ext)
        if os.path.exists(path):
            if ext == ".csv":
                return pd.read_csv(path)
            else:
                return pd.read_excel(path)
    raise FileNotFoundError(
        f"Could not find {split_name}.csv/.xlsx/.xls in {split_dir}"
    )


# Optional: central schema registry per dataset name
DATASET_SCHEMAS = {
    # MovieLens modern
    "ml-modern": dict(
        user_col="userId",
        item_col="movieId",
        rating_col="rating",
        ts_col="timestamp",
    ),
    # Amazon cloth
    "amazon-cloth": dict(
        user_col="user_id",
        item_col="asin",
        rating_col="rating",
        ts_col="timestamp",
    ),
    # Amazon sports
    "amazon-sports": dict(
        user_col="user_id",
        item_col="asin",
        rating_col="rating",
        ts_col="timestamp",
    ),
}


def load_gts_dataset(
    root_dir,
    dataset_name,
    user_col = None,
    item_col = None,
    rating_col = None,
    ts_col = None,
):
    """
    Generic loader for 70-15-15 temporal splits in datasets-temporal-splits/<dataset_name>/.

    It supports different column names for user/item depending on dataset:
      - MovieLens: userId, movieId, rating, timestamp
      - Amazon:    user_id, asin (or parent_asin), rating, timestamp
    """

    # 1) Resolve schema for this dataset
    schema = DATASET_SCHEMAS.get(dataset_name, {})
    user_col = user_col or schema.get("user_col", "userId")
    item_col = item_col or schema.get("item_col", "movieId")
    rating_col = rating_col or schema.get("rating_col", "rating")
    ts_col = ts_col or schema.get("ts_col", "timestamp")

    split_dir = os.path.join(root_dir, "datasets-temporal-splits", dataset_name)

    train_df = _read_split_file(split_dir, "train")
    val_df = _read_split_file(split_dir, "val")
    test_df = _read_split_file(split_dir, "test")

    expected_cols = {user_col, item_col, rating_col, ts_col}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} split for {dataset_name} missing {missing}. "
                f"Found columns: {df.columns.tolist()}"
            )

    # 2) Build global mappings for raw user / item IDs
    all_users = pd.concat(
        [train_df[user_col], val_df[user_col], test_df[user_col]]
    ).unique()
    all_items = pd.concat(
        [train_df[item_col], val_df[item_col], test_df[item_col]]
    ).unique()

    # Use native types (ints stay ints, strings stay strings)
    all_users_sorted = sorted(all_users)
    all_items_sorted = sorted(all_items)

    user2id = {uid: idx for idx, uid in enumerate(all_users_sorted)}
    id2user = list(all_users_sorted)

    item2id = {iid: idx for idx, iid in enumerate(all_items_sorted)}
    id2item = list(all_items_sorted)


    num_users = len(user2id)
    num_items = len(item2id)

    def df_to_interactions(df):
        inters = []
        for _, row in df.iterrows():
            u_raw = row[user_col]
            i_raw = row[item_col]
            ts = int(row[ts_col])

            # Skip missing IDs
            if pd.isna(u_raw) or pd.isna(i_raw):
                print("SKIP MISSING ID")
                continue

            # Look up directly using native types
            try:
                u = user2id[u_raw]
                i = item2id[i_raw]
            except KeyError:
                print("SOME WEIRD MISMATCH")
                continue

            inters.append(Interaction(user=u, item=i, timestamp=ts))

        inters.sort(key=lambda x: (x.user, x.timestamp))
        return inters


    train_inters = df_to_interactions(train_df)
    val_inters = df_to_interactions(val_df)
    test_inters = df_to_interactions(test_df)

    splits = {"train": train_inters, "val": val_inters, "test": test_inters}

    # 3) Per-user lists
    user_train_items = {u: [] for u in range(num_users)}
    user_val_items = {u: [] for u in range(num_users)}
    user_test_items = {u: [] for u in range(num_users)}
    user_all_pos_items = {u: [] for u in range(num_users)}

    for inter in train_inters:
        user_train_items[inter.user].append(inter.item)
        user_all_pos_items[inter.user].append(inter.item)

    for inter in val_inters:
        user_val_items[inter.user].append(inter.item)
        user_all_pos_items[inter.user].append(inter.item)

    for inter in test_inters:
        user_test_items[inter.user].append(inter.item)
        user_all_pos_items[inter.user].append(inter.item)

    def _dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    for u in range(num_users):
        user_train_items[u] = _dedup(user_train_items[u])
        user_val_items[u] = _dedup(user_val_items[u])
        user_test_items[u] = _dedup(user_test_items[u])
        user_all_pos_items[u] = _dedup(user_all_pos_items[u])

    return GTSDataset(
        num_users=num_users,
        num_items=num_items,
        user2id=user2id,
        id2user=id2user,
        item2id=item2id,
        id2item=id2item,
        splits=splits,
        user_train_items=user_train_items,
        user_val_items=user_val_items,
        user_test_items=user_test_items,
        user_all_pos_items=user_all_pos_items,
    )


def build_lightgcn_inputs(dataset: GTSDataset):
    """
    Build LightGCN-style inputs from the canonical GTS dataset.

    Returns:
        num_users, num_items, train_edge (np.array [N,2]),
        user_train_dict (user -> set(items))
    """
    train_pairs = [(inter.user, inter.item) for inter in dataset.splits["train"]]
    train_edge = np.array(train_pairs, dtype=np.int64)

    user_train_dict = {u: set(items) for u, items in dataset.user_train_items.items()}

    return dataset.num_users, dataset.num_items, train_edge, user_train_dict


def build_sasrec_inputs(dataset: GTSDataset):
    """
    Build SASRec-style inputs: per-user train sequences and per-user val/test items.

    Returns:
        num_users, num_items,
        user_train_seq (dict: u -> [items in order]),
        user_val_items (dict),
        user_test_items (dict)
    """
    user_train_seq = dataset.user_train_items  # already ordered
    user_val_items = dataset.user_val_items
    user_test_items = dataset.user_test_items

    return (
        dataset.num_users,
        dataset.num_items,
        user_train_seq,
        user_val_items,
        user_test_items,
    )


def build_mmgcn_inputs(dataset: GTSDataset):
    """
    Build MMGCN-style arrays:

        train_edges: [N_train, 2] (user, global_item)
        user_item_dict: {user: set(train_items)}
        val_full: [num_users_with_val, ...]
        test_full: [num_users_with_test, ...]

    Here we treat 'val' and 'test' split items as the ground truth
    for evaluation, but MMGCN itself will be trained on train_edges only.
    """
    num_users = dataset.num_users
    num_items = dataset.num_items

    train_pairs = [(inter.user, inter.item) for inter in dataset.splits["train"]]
    train_edges = np.array(train_pairs, dtype=np.int64)

    user_item_dict = {u: set(items) for u, items in dataset.user_train_items.items()}

    # val_full/test_full follow the MMGCN convention:
    # each row: [user, pos1, pos2, ...]
    val_full_entries = []
    for u, items in dataset.user_val_items.items():
        if items:
            val_full_entries.append(np.array([u] + items, dtype=np.int64))

    test_full_entries = []
    for u, items in dataset.user_test_items.items():
        if items:
            test_full_entries.append(np.array([u] + items, dtype=np.int64))

    val_full = np.array(val_full_entries, dtype=object)
    test_full = np.array(test_full_entries, dtype=object)

    return num_users, num_items, train_edges, user_item_dict, val_full, test_full


#     dataset = load_gts_dataset(
#     root_dir=PROJECT_ROOT,
#     dataset_name="ml-modern",   # schema entry uses userId/movieId
# )

# dataset = load_gts_dataset(
#     root_dir=PROJECT_ROOT,
#     dataset_name="amazon-cloth",  # uses user_id / asin
# )
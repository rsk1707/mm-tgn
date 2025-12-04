import os
import sys
from typing import Callable

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from baseline_scripts.data_loader import load_gts_dataset
from baseline_scripts.eval_sampled import (
    build_user_pos_pairs_from_test,
    evaluate_sampled,
)


def make_lightgcn_score_fn(
    user_emb: np.ndarray,
    item_emb: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    score_fn(users, items) -> [B, C] scores via dot product.
    users: np.ndarray[int], shape [B]
    items: np.ndarray[int], shape [C]
    """
    def score_fn(user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        u_vecs = user_emb[user_ids]   # [B, D]
        i_vecs = item_emb[item_ids]   # [C, D]
        scores = u_vecs @ i_vecs.T    # [B, C]
        return scores.astype(np.float32)

    return score_fn


def main():
    dataset_name = "ml-modern"
    gts_name = dataset_name + "-gts"

    # 1) Canonical GTS dataset (same as MM-TGN)
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name=dataset_name)

    # 2) Load trained LightGCN embeddings
    emb_dir = os.path.join(PROJECT_ROOT, "lightgcn", "Data", gts_name)
    user_emb = np.load(os.path.join(emb_dir, "user_emb.npy"))
    item_emb = np.load(os.path.join(emb_dir, "item_emb.npy"))

    # 3) Build score_fn adapter
    score_fn = make_lightgcn_score_fn(user_emb, item_emb)

    # 4) Build test pairs from the GTS *test* split
    user_pos_pairs = build_user_pos_pairs_from_test(dataset.user_test_items)

    # 5) Unified sampled evaluation (1 pos + 100 negs, K=10)
    hit, ndcg, mrr = evaluate_sampled(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=dataset.num_items,
        user_all_pos_items=dataset.user_all_pos_items,
        num_neg=100,
        k=10,
        seed=42,
    )

    print(f"[LightGCN | {dataset_name}] Hit@10={hit:.4f}, NDCG@10={ndcg:.4f}, MRR@10={mrr:.4f}")


if __name__ == "__main__":
    main()

# python baseline_scripts/run_lightgcn_eval.py

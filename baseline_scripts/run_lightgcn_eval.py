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
from baseline_scripts.eval_subset import (
    load_eval_subset_pairs,
    compute_link_metrics_from_score_fn,
    evaluate_ranking_multi_k,
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

def eval_main():
    dataset_name = "ml-modern"
    gts_name = dataset_name + "-gts"

    # 1) Load GTS + eval subset pairs (about 5k interactions)
    #    File: eval_samples/ml-modern_eval_sample.csv
    dataset, user_pos_pairs = load_eval_subset_pairs(
        root_dir=PROJECT_ROOT,
        dataset_name=dataset_name,
        eval_relpath=os.path.join("eval_samples", "ml-modern_eval_sample.csv"),
    )
    num_users = dataset.num_users
    num_items = dataset.num_items

    # 2) Load trained LightGCN embeddings (already saved after training)
    emb_dir = os.path.join(PROJECT_ROOT, "lightgcn", "Data", gts_name)
    user_emb = np.load(os.path.join(emb_dir, "user_emb.npy"))
    item_emb = np.load(os.path.join(emb_dir, "item_emb.npy"))

    assert user_emb.shape[0] == num_users
    assert item_emb.shape[0] == num_items

    # 3) Build score_fn adapter
    score_fn = make_lightgcn_score_fn(user_emb, item_emb)

    # 4) Link prediction metrics (AP, AUC, link-MRR)
    link_metrics = compute_link_metrics_from_score_fn(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=num_items,
        user_all_pos_items=dataset.user_all_pos_items,
        seed=42,
    )

    # 5) Ranking metrics on eval subset (Hit/Recall@10/20, NDCG@10/20, full MRR)
    ranking_metrics = evaluate_ranking_multi_k(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=num_items,
        user_all_pos_items=dataset.user_all_pos_items,
        ks=(10, 20),
        num_neg=100,
        seed=42,
    )

    # 6) Print nicely
    print(f"\n[LightGCN | {dataset_name} | eval_samples/ml-modern_eval_sample.csv]")
    print("Link prediction metrics:")
    print(f"  AP   : {link_metrics['AP']:.4f}")
    print(f"  AUC  : {link_metrics['AUC']:.4f}")
    print(f"  MRR  : {link_metrics['MRR']:.4f}")

    print("\nRanking metrics (sampled, 1 pos + 100 negs):")
    print(f"  Recall@10 / Hit@10: {ranking_metrics['recall@10']:.4f}")
    print(f"  Recall@20 / Hit@20: {ranking_metrics['recall@20']:.4f}")
    print(f"  NDCG@10          : {ranking_metrics['ndcg@10']:.4f}")
    print(f"  NDCG@20          : {ranking_metrics['ndcg@20']:.4f}")
    print(f"  MRR              : {ranking_metrics['mrr']:.4f}")


if __name__ == "__main__":
    # main()
    eval_main()

# python baseline_scripts/run_lightgcn_eval.py

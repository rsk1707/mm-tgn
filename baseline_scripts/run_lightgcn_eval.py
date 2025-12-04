# baseline_scripts/run_lightgcn_eval.py

import os
import sys
from typing import Tuple

import numpy as np
import torch

# Adjust these to point into your project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from baseline_scripts.data_loader import load_gts_dataset, build_lightgcn_inputs
from baseline_scripts.eval_sampled import (
    build_user_pos_pairs_from_test,
    evaluate_sampled,
)

# Example: import your LightGCN model and loader
# from lightgcn.LightGCN import LightGCN  # adjust to your actual path


def make_lightgcn_score_fn(model) -> callable:
    """
    Adapter to map (user_ids, item_ids) -> scores for LightGCN.

    This assumes your model exposes final user/item embeddings as something like:
      model.final_user_embeddings  [num_users, dim]
      model.final_item_embeddings  [num_items, dim]

    Adjust names according to your implementation.
    """
    # TODO: replace with your actual attributes / method
    user_emb = model.final_user_embeddings  # torch.Tensor [U, D]
    item_emb = model.final_item_embeddings  # torch.Tensor [I, D]

    user_emb = user_emb.detach().cpu()
    item_emb = item_emb.detach().cpu()

    def score_fn(user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        u = torch.from_numpy(user_ids).long()
        i = torch.from_numpy(item_ids).long()
        u_vecs = user_emb[u]               # [B, D]
        i_vecs = item_emb[i]               # [C, D]
        scores = torch.matmul(u_vecs, i_vecs.t())  # [B, C]
        return scores.numpy().astype(np.float32)

    return score_fn


def main():
    # 1) Load canonical GTS dataset
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name="ml-modern")

    # 2) Build LightGCN-style inputs if you want to retrain
    num_users, num_items, train_edge, user_train_dict = build_lightgcn_inputs(dataset)

    # 3) Load your trained LightGCN model here (no warm start)
    # Example placeholder:
    # model = LightGCN(num_users=num_users, num_items=num_items, ...)
    # model.load_state_dict(torch.load("path/to/checkpoint.pt"))
    # model.eval()

    model = ...  # TODO: construct & load your LightGCN checkpoint
    model.eval()

    # 4) Build score_fn adapter
    score_fn = make_lightgcn_score_fn(model)

    # 5) Build test pairs from the test split
    user_pos_pairs = build_user_pos_pairs_from_test(dataset.user_test_items)

    # 6) Run unified sampled evaluation
    hit, ndcg, mrr = evaluate_sampled(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=dataset.num_items,
        user_all_pos_items=dataset.user_all_pos_items,
        num_neg=100,
        k=10,
        seed=42,
    )

    print(f"[LightGCN] Hit@10={hit:.4f}, NDCG@10={ndcg:.4f}, MRR@10={mrr:.4f}")


if __name__ == "__main__":
    main()

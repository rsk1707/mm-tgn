# baseline_scripts/run_mmgcn_eval.py

import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from baseline_scripts.data_loader import load_gts_dataset, build_mmgcn_inputs
from baseline_scripts.eval_sampled import (
    build_user_pos_pairs_from_test,
    evaluate_sampled,
)

# from mmgcn.Model_MMGCN import Net  # adjust import to your path


def make_mmgcn_score_fn(model, num_users: int) -> callable:
    """
    Adapter for MMGCN.

    Assumes model.forward() computes `model.result` with shape [num_user+num_item, dim].
    First num_user rows are user embeddings, remaining are item embeddings.
    """

    device = next(model.parameters()).device

    # Run a forward pass once to populate model.result
    model.eval()
    with torch.no_grad():
        _ = model.forward()
        all_emb = model.result.detach().cpu()  # [U+I, D]

    user_emb = all_emb[:num_users]
    item_emb = all_emb[num_users:]

    def score_fn(user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        u = torch.from_numpy(user_ids).long()
        i = torch.from_numpy(item_ids).long()
        u_vecs = user_emb[u]              # [B, D]
        i_vecs = item_emb[i]              # [C, D]
        scores = torch.matmul(u_vecs, i_vecs.t())  # [B, C]
        return scores.numpy().astype(np.float32)

    return score_fn


def main():
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name="ml-modern")
    num_users, num_items, train_edges, user_item_dict, val_full, test_full = build_mmgcn_inputs(dataset)

    # Load MMGCN model and its features exactly like your existing training code;
    # here we only care about its embeddings at evaluation time.
    # Example:
    # v_feat = ...
    # a_feat = ...
    # t_feat = ...
    # model = Net(v_feat, a_feat, t_feat, None, train_edges, ..., num_user=num_users, num_item=num_items, ...)
    # model.load_state_dict(torch.load("path/to/mmgcn_checkpoint.pt"))
    model = ...  # TODO
    model.eval()

    score_fn = make_mmgcn_score_fn(model, num_users=num_users)

    user_pos_pairs = build_user_pos_pairs_from_test(dataset.user_test_items)

    hit, ndcg, mrr = evaluate_sampled(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=dataset.num_items,
        user_all_pos_items=dataset.user_all_pos_items,
        num_neg=100,
        k=10,
        seed=42,
    )

    print(f"[MMGCN] Hit@10={hit:.4f}, NDCG@10={ndcg:.4f}, MRR@10={mrr:.4f}")


if __name__ == "__main__":
    main()

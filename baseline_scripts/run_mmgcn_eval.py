# baseline_scripts/run_mmgcn_eval.py

import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from baseline_scripts.data_loader import load_gts_dataset
from baseline_scripts.eval_sampled import (
    build_user_pos_pairs_from_test,
    evaluate_sampled,
)

# import MMGCN code
sys.path.append(os.path.join(PROJECT_ROOT, "mmgcn"))
from Dataset import data_load
from Model_MMGCN import Net


def make_mmgcn_score_fn(model: torch.nn.Module, num_users: int):
    """
    Adapter for MMGCN.

    Assumes model.forward() computes `model.result` with shape [num_user+num_item, dim].
    First num_user rows are user embeddings, remaining are item embeddings.
    """
    model.eval()
    with torch.no_grad():
        _ = model.forward()
        all_emb = model.result.detach().cpu()  # [U+I, D]

    user_emb = all_emb[:num_users]       # [U, D]
    item_emb = all_emb[num_users:]       # [I, D]

    def score_fn(user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """
        user_ids: shape [B]
        item_ids: shape [C] (internal item ids 0..I-1, consistent with GTS)
        """
        u = torch.from_numpy(user_ids).long()
        i = torch.from_numpy(item_ids).long()
        u_vecs = user_emb[u]           # [B, D]
        i_vecs = item_emb[i]           # [C, D]
        scores = torch.matmul(u_vecs, i_vecs.t())  # [B, C]
        return scores.numpy().astype(np.float32)

    return score_fn


def main():
    dataset_name = "ml-modern"
    gts_name = dataset_name + "-gts"

    # 1) Canonical GTS dataset (same as MM-TGN / LightGCN / SASRec)
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name=dataset_name)
    num_users = dataset.num_users
    num_items = dataset.num_items

    # 2) Load MMGCN training data & features produced by build_mmgcn_gts.py
    data_path = gts_name  # "ml-modern-gts"
    mmgcn_data_dir = os.path.join(PROJECT_ROOT, "mmgcn", "Data", data_path)

    num_user_m, num_item_m, train_edge, user_item_dict, v_feat, a_feat, t_feat = data_load(
        data_path,
        has_v=True,
        has_a=False,
        has_t=True,
    )

    # sanity check shapes
    assert num_user_m == num_users, f"GTS users ({num_users}) != MMGCN users ({num_user_m})"
    assert num_item_m == num_items, f"GTS items ({num_items}) != MMGCN items ({num_item_m})"

    # 3) Recreate MMGCN model exactly as in training
    dim_E = 64          # must match --dim_E used in training
    batch_size = 1024   # used only for internal config
    weight_decay = 1e-5 # match training
    aggr_mode = "mean"
    num_layer = 2
    has_id = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(
        v_feat=v_feat,              # already CUDA tensors or None
        a_feat=a_feat,
        t_feat=t_feat,
        words_tensor=None,
        edge_index=train_edge,
        batch_size=batch_size,
        num_user=num_users,
        num_item=num_items,
        aggr_mode=aggr_mode,
        concate="False",
        num_layer=num_layer,
        has_id=has_id,
        user_item_dict=user_item_dict,
        reg_weight=weight_decay,
        dim_x=dim_E,
    ).to(device)

    # 4) Load trained weights
    ckpt_path = os.path.join(mmgcn_data_dir, "mmgcn_ml-modern-gts.pt")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # 5) Build score_fn from embeddings
    score_fn = make_mmgcn_score_fn(model, num_users=num_users)

    # 6) Build test user–positive pairs from GTS test split
    user_pos_pairs = build_user_pos_pairs_from_test(dataset.user_test_items)

    # 7) Unified sampled evaluation: 1 pos + 100 negs, K=10
    hit, ndcg, mrr = evaluate_sampled(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=dataset.num_items,
        user_all_pos_items=dataset.user_all_pos_items,
        num_neg=100,
        k=10,
        seed=42,
    )

    print(f"[MMGCN | {dataset_name}] Hit@10={hit:.4f}, NDCG@10={ndcg:.4f}, MRR@10={mrr:.4f}")


if __name__ == "__main__":
    main()
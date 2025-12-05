# baseline_scripts/run_mmgcn_eval.py

import os
import sys
import time
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
    Adapter for MMGCN: extracts user and item embeddings after a forward pass.
    """
    print("\n[make_score_fn] Running model.forward() to build embeddings...")
    t0 = time.time()

    model.eval()
    with torch.no_grad():
        _ = model.forward()
        all_emb = model.result.detach().cpu()

    t1 = time.time()
    print(f"[make_score_fn] forward() complete in {t1 - t0:.2f} sec.")
    print(f"[make_score_fn] all_emb shape = {all_emb.shape}")

    user_emb = all_emb[:num_users]
    item_emb = all_emb[num_users:]

    print(f"[make_score_fn] user_emb: {user_emb.shape}, item_emb: {item_emb.shape}")

    def score_fn(user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        u = torch.from_numpy(user_ids).long()
        i = torch.from_numpy(item_ids).long()

        u_vecs = user_emb[u]
        i_vecs = item_emb[i]
        scores = torch.matmul(u_vecs, i_vecs.t())

        return scores.numpy().astype(np.float32)

    print("[make_score_fn] Score function ready.")
    return score_fn


def main():
    print("\nSTEP 1: Load GTS Dataset")
    dataset_name = "ml-modern"
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name=dataset_name)
    num_users, num_items = dataset.num_users, dataset.num_items
    print(f"[GTS] num_users={num_users}, num_items={num_items}")

    print("\nSTEP 2: Load MMGCN Data (train.npy and features)")
    gts_name = dataset_name + "-gts"
    # data_path = gts_name
    # mmgcn_data_dir = os.path.join(PROJECT_ROOT, "mmgcn", "Data", data_path)

    # (num_user_m, num_item_m,
    #  train_edge, user_item_dict,
    #  v_feat, a_feat, t_feat) = data_load(
    #     data_path,
    #     has_v=True,
    #     has_a=False,
    #     has_t=True,
    # )
    # Absolute path to MMGCN GTS Data directory
    mmgcn_data_dir = os.path.join(PROJECT_ROOT, "mmgcn", "Data", gts_name)

    print(f"[Paths] MMGCN data directory = {mmgcn_data_dir}")

    num_user_m, num_item_m, train_edge, user_item_dict, v_feat, a_feat, t_feat = data_load(
            mmgcn_data_dir,
            has_v=True,
            has_a=False,
            has_t=True,
    )

    print(f"[MMGCN data] num_user_m={num_user_m}, num_item_m={num_item_m}")
    print(f"[MMGCN data] train_edge shape = {train_edge.shape}")
    if v_feat is not None: print(f"[MMGCN data] v_feat shape = {v_feat.shape}")
    if t_feat is not None: print(f"[MMGCN data] t_feat shape = {t_feat.shape}")

    assert num_user_m == num_users, "User count mismatch!"
    assert num_item_m == num_items, "Item count mismatch!"

    print("\nSTEP 3: Recreate MMGCN Model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    model = Net(
        v_feat=v_feat,
        a_feat=a_feat,
        t_feat=t_feat,
        words_tensor=None,
        edge_index=train_edge,
        batch_size=1024,
        num_user=num_users,
        num_item=num_items,
        aggr_mode="mean",
        concate="False",
        num_layer=2,
        has_id=True,
        user_item_dict=user_item_dict,
        reg_weight=1e-5,
        dim_x=64,
    ).to(device)

    print("[MMGCN model] Model instantiated.")

    print("\n=== STEP 4: Load Trained Checkpoint ===")
    ckpt_path = os.path.join(mmgcn_data_dir, "mmgcn_ml-modern-gts.pt")
    print(f"[Checkpoint] Loading from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print("[Checkpoint] Loaded successfully.")

    print("\n=== STEP 5: Build Score Function (extract embeddings) ===")
    score_fn = make_mmgcn_score_fn(model, num_users=num_users)

    print("\n=== STEP 6: Build Test User–Positive Pairs ===")
    user_pos_pairs = build_user_pos_pairs_from_test(dataset.user_test_items)
    print(f"[Eval] user_pos_pairs count = {len(user_pos_pairs)}")

    print("\n=== STEP 7: Unified Sampled Evaluation (1 pos + 100 negs) ===")
    t0 = time.time()
    hit, ndcg, mrr = evaluate_sampled(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=num_items,
        user_all_pos_items=dataset.user_all_pos_items,
        num_neg=100,
        k=10,
        seed=42,
    )
    t1 = time.time()

    print(f"[Eval] Completed in {t1 - t0:.2f} sec.")
    print(f"[MMGCN | {dataset_name}] Hit@10={hit:.4f}, NDCG@10={ndcg:.4f}, MRR@10={mrr:.4f}")


if __name__ == "__main__":
    main()
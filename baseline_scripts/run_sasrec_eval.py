# baseline_scripts/run_sasrec_eval.py

import os
import sys
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from baseline_scripts.data_loader import load_gts_dataset, build_sasrec_inputs
from baseline_scripts.eval_sampled import (
    build_user_pos_pairs_from_test,
    evaluate_sampled,
)

# from sasrec.model import SASRec  # adjust to your actual paths


def make_sasrec_score_fn(
    model,
    user_train_seq: Dict[int, List[int]],
    max_seq_len: int,
) -> callable:
    """
    Adapter for SASRec.

    Expects `model` to have something like:
      model.predict(user_sequence, item_ids) -> scores

    You will need to adapt this to your implementation (e.g., padding, device).
    """

    device = next(model.parameters()).device

    def _build_sequence(u: int) -> torch.Tensor:
        seq = user_train_seq.get(u, [])
        if len(seq) >= max_seq_len:
            seq = seq[-max_seq_len:]
        # left pad with 0 if you use 0 as [PAD] / [MASK] in SASRec
        pad_len = max_seq_len - len(seq)
        padded = [0] * pad_len + seq
        return torch.tensor(padded, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]

    def score_fn(user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        # Here we assume user_ids has shape [B], but our eval_sampled calls with B=1.
        assert user_ids.ndim == 1
        assert item_ids.ndim == 1

        u = int(user_ids[0])
        cand = torch.from_numpy(item_ids).long().to(device)  # [C]

        seq = _build_sequence(u)  # [1, L]

        # TODO: adapt this call to your SASRec implementation
        # Example:
        # scores = model.predict(seq, cand.unsqueeze(0))  # [1, C]
        scores = model.predict(seq, cand.unsqueeze(0))  # replace with actual

        scores = scores.detach().cpu().numpy()
        return scores.astype(np.float32)

    return score_fn


def main():
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name="ml-modern")
    num_users, num_items, user_train_seq, user_val_items, user_test_items = build_sasrec_inputs(dataset)

    # Load your trained SASRec here (no warm start to other models; just its own training)
    # model = SASRec(num_users, num_items, ...)  # TODO
    # model.load_state_dict(torch.load("path/to/sasrec_checkpoint.pt"))
    model = ...  # TODO
    model.eval()

    max_seq_len = 50  # or whatever you used in training
    score_fn = make_sasrec_score_fn(model, user_train_seq, max_seq_len)

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

    print(f"[SASRec] Hit@10={hit:.4f}, NDCG@10={ndcg:.4f}, MRR@10={mrr:.4f}")


if __name__ == "__main__":
    main()

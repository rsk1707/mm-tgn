# baseline_scripts/eval_subset.py

import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from baseline_scripts.data_loader import load_gts_dataset, DATASET_SCHEMAS, GTSDataset
from baseline_scripts.eval_sampled import evaluate_sampled  # reuse neg-sampling logic


def load_eval_subset_pairs(root_dir,dataset_name,eval_relpath):
    dataset = load_gts_dataset(root_dir=root_dir, dataset_name=dataset_name)

    # Resolve schema for raw column names
    schema = DATASET_SCHEMAS[dataset_name]
    user_col = schema["user_col"]
    item_col = schema["item_col"]

    eval_path = os.path.join(root_dir, eval_relpath)
    df = pd.read_csv(eval_path)

    missing = {user_col, item_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Eval subset {eval_path} missing columns {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    user_pos_pairs = []
    skipped = 0

    for _, row in df.iterrows():
        u_raw = row[user_col]
        i_raw = row[item_col]
        # Map raw IDs -> internal IDs if present
        if u_raw in dataset.user2id and i_raw in dataset.item2id:
            u = dataset.user2id[u_raw]
            i = dataset.item2id[i_raw]
            user_pos_pairs.append((u, i))
        else:
            skipped += 1

    if skipped > 0:
        print(f"[eval_subset] Skipped {skipped} rows with unmapped IDs.")

    print(f"[eval_subset] Loaded {len(user_pos_pairs)} eval pairs from {eval_relpath}.")
    return dataset, user_pos_pairs

def compute_link_pred_metrics(pos_probs, neg_probs):
    """
    Same semantics as evaluate_mmtgn.compute_link_pred_metrics:
    compute AP, AUC, and a simple pair-wise MRR.
    """
    labels = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    scores = np.concatenate([pos_probs, neg_probs])

    # Filter NaN / Inf
    valid_mask = np.isfinite(scores)
    if not valid_mask.all():
        labels = labels[valid_mask]
        scores = scores[valid_mask]

    if len(np.unique(labels)) < 2:
        return {"AP": 0.0, "AUC": 0.5, "MRR": 0.0}

    ap = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)

    # Pair-wise MRR: compare pos[i] vs neg[i]
    mrr_sum = 0.0
    n_pairs = min(len(pos_probs), len(neg_probs))
    for i in range(n_pairs):
        rank = 1 if pos_probs[i] > neg_probs[i] else 2
        mrr_sum += 1.0 / rank
    mrr = mrr_sum / n_pairs if n_pairs > 0 else 0.0

    return {"AP": ap, "AUC": auc, "MRR": mrr}

def compute_link_metrics_from_score_fn(
    score_fn,
    user_pos_pairs,
    num_items,
    user_all_pos_items,
    seed,
    num_neg,
):
    """
    Approximate link-prediction metrics for *static* recommenders.

    For each (u, pos):
      - sample `num_neg` negatives not in user's positives
      - compute scores for [pos] + negatives  (length = 1 + num_neg)
      - expand pos_scores / neg_scores and feed into AP/AUC/MRR.
    """
    rng = random.Random(seed)

    pos_scores = []
    neg_scores = []

    for (u, pos) in user_pos_pairs:
        pos_set = set(user_all_pos_items.get(u, []))
        pos_set.add(pos)

        # Sample num_neg distinct negatives
        negs = set()
        attempts = 0
        max_attempts = num_neg * 10
        while len(negs) < num_neg and attempts < max_attempts:
            cand = rng.randrange(0, num_items)
            if cand not in pos_set and cand not in negs:
                negs.add(cand)
            attempts += 1

        if not negs:
            continue

        neg_list = list(negs)

        # candidates: index 0 is always the positive item
        candidates = [pos] + neg_list
        users = np.array([u], dtype=np.int64)
        items = np.array(candidates, dtype=np.int64)

        scores = score_fn(users, items)[0]  # shape [1 + num_neg]

        pos_score = float(scores[0])
        neg_scores_list = scores[1:]

        # For AP/AUC/MRR we treat each (pos, neg_j) as a pair
        for neg_s in neg_scores_list:
            pos_scores.append(pos_score)
            neg_scores.append(float(neg_s))

    if not pos_scores or not neg_scores:
        return {"AP": 0.0, "AUC": 0.5, "MRR": 0.0}

    pos_arr = np.asarray(pos_scores, dtype=np.float32)
    neg_arr = np.asarray(neg_scores, dtype=np.float32)
    return compute_link_pred_metrics(pos_arr, neg_arr)

def evaluate_ranking_multi_k(
    score_fn,
    user_pos_pairs,
    num_items,
    user_all_pos_items,
    ks=(10, 20),
    num_neg= 100,
    seed = 42,
):
    """
    Full ranking evaluation with negative sampling, very similar in spirit to
    evaluate_mmtgn.evaluate_ranking but using our simple score_fn interface.

    Returns:
      {
        "hit@10": ...,
        "hit@20": ...,
        "recall@10": ...,
        "recall@20": ...,
        "ndcg@10": ...,
        "ndcg@20": ...,
        "mrr": full_mrr
      }
    """
    rng = random.Random(seed)
    ks = sorted(set(ks))
    max_k = max(ks)

    hits = {k: 0.0 for k in ks}
    ndcgs = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    count = 0

    for (u, pos) in user_pos_pairs:
        pos_set = set(user_all_pos_items.get(u, []))
        pos_set.add(pos)

        # Negative sampling (reuse idea from evaluate_sampled)
        negs = set()
        attempts = 0
        max_attempts = num_neg * 10
        while len(negs) < num_neg and attempts < max_attempts:
            cand = rng.randrange(0, num_items)
            if cand not in pos_set and cand not in negs:
                negs.add(cand)
            attempts += 1

        if not negs:
            continue

        neg_list = list(negs)
        candidates = [pos] + neg_list

        user_arr = np.array([u], dtype=np.int64)
        item_arr = np.array(candidates, dtype=np.int64)
        scores = score_fn(user_arr, item_arr)[0]  # [num_candidates]

        ranking = np.argsort(-scores)  # indices into candidates

        pos_rank = None
        for r_idx, cand_idx in enumerate(ranking):
            if cand_idx == 0:
                pos_rank = r_idx
                break
        if pos_rank is None:
            continue

        # Full MRR (no cutoff)
        mrr_sum += 1.0 / (pos_rank + 1.0)

        # Per-K metrics
        for k in ks:
            if pos_rank < k:
                hits[k] += 1.0
                dcg = 1.0 / np.log2(pos_rank + 2.0)
                idcg = 1.0 / np.log2(1.0 + 1.0)
                ndcgs[k] += dcg / idcg

        count += 1

    if count == 0:
        return {
            "hit@10": 0.0, "hit@20": 0.0,
            "recall@10": 0.0, "recall@20": 0.0,
            "ndcg@10": 0.0, "ndcg@20": 0.0,
            "mrr": 0.0,
        }

    results = {}
    for k in ks:
        hit_k = hits[k] / count
        ndcg_k = ndcgs[k] / count
        results[f"hit@{k}"] = hit_k
        results[f"recall@{k}"] = hit_k  # with 1 positive per query, hit == recall
        results[f"ndcg@{k}"] = ndcg_k

    results["mrr"] = mrr_sum / count
    return results
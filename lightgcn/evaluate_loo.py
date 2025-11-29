"""
@author: Zhongchuan Sun
"""
import itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys
import heapq


def argmax_top_k(a, top_k=50):
    """
    Return indices of the top_k largest elements in array a.
    """
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)


def _normalize_ground_truth(ground_truth):
    """
    Ensure ground_truth is treated as a 1D list of item ids.
    Handles:
      - int
      - list/tuple
      - np.ndarray (any shape)
    """
    if isinstance(ground_truth, (list, tuple)):
        return list(ground_truth)
    if isinstance(ground_truth, np.ndarray):
        return ground_truth.flatten().tolist()
    # scalar
    return [int(ground_truth)]


def hit(rank, ground_truth):
    """
    For each cutoff k (1..len(rank)), record 1.0 if ANY ground_truth item
    appears within top-k, else 0.0.
    """
    gt_items = _normalize_ground_truth(ground_truth)

    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item in gt_items:
            last_idx = idx
            break

    result = np.zeros(len(rank), dtype=np.float32)
    if last_idx != sys.maxsize:
        result[last_idx:] = 1.0
    return result


def ndcg(rank, ground_truth):
    """
    For each cutoff k (1..len(rank)), record DCG for the first relevant item.
    Here relevance is 1 for any ground_truth item, 0 otherwise.
    """
    gt_items = _normalize_ground_truth(ground_truth)

    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item in gt_items:
            last_idx = idx
            break

    result = np.zeros(len(rank), dtype=np.float32)
    if last_idx != sys.maxsize:
        gain = 1.0 / np.log2(last_idx + 2)  # rank indices are 0-based
        result[last_idx:] = gain
    return result


def mrr(rank, ground_truth):
    """
    For each cutoff k, record reciprocal rank of the first relevant item.
    """
    gt_items = _normalize_ground_truth(ground_truth)

    last_idx = sys.maxsize
    for idx, item in enumerate(rank):
        if item in gt_items:
            last_idx = idx
            break

    result = np.zeros(len(rank), dtype=np.float32)
    if last_idx != sys.maxsize:
        rr = 1.0 / (last_idx + 1)
        result[last_idx:] = rr
    return result


def eval_score_matrix_loo(score_matrix, test_items, top_k=50, thread_num=None):
    """
    score_matrix: shape (B, N) – scores for each user in the batch
    test_items:   list-like of length B – each element can be:
                  - a single item id (int)
                  - a list / np.array of item ids
    top_k:        maximum cutoff to evaluate metrics at

    Returns:
      ndarray of shape (B, 3 * top_k) in the order:
        [ hit@1..K, ndcg@1..K, mrr@1..K ]
    """
    def _eval_one_user(idx):
        scores = score_matrix[idx]        # scores for user idx
        test_item = test_items[idx]       # test items for user idx

        ranking = argmax_top_k(scores, top_k)  # Top-K item indices
        result = []

        # Each of these returns a vector of length top_k
        result.extend(hit(ranking, test_item))
        result.extend(ndcg(ranking, test_item))
        result.extend(mrr(ranking, test_item))

        result = np.array(result, dtype=np.float32).flatten()
        return result

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        batch_result = executor.map(_eval_one_user, range(len(test_items)))

    # (B, metric_num * top_k)
    result = list(batch_result)
    return np.array(result)
# baseline_scripts/eval_sampled.py

import random

import numpy as np


def build_user_pos_pairs_from_test(user_test_items):
    """
    Expand per-user test items into a flat list of (user, positive_item) pairs.
    If a user has multiple test items, they appear multiple times.
    """
    pairs: List[Tuple[int, int]] = []
    for u, items in user_test_items.items():
        for i in items:
            pairs.append((u, i))
    return pairs


def evaluate_sampled(
    score_fn,
    user_pos_pairs,
    num_items,
    user_all_pos_items,
    num_neg = 100,
    k = 10,
    seed = 42,
    max_examples = -1,
):
    """
    Returns:
        (hit_at_k, ndcg_at_k, mrr_at_k)
    """
    rng = random.Random(seed)

    all_items = list(range(num_items))

    hits = 0.0
    ndcgs = 0.0
    mrrs = 0.0
    count = 0

    for (u, pos) in user_pos_pairs:
        # optional cap on examples for debugging
        if max_examples > 0 and count >= max_examples:
            break

        # build negative pool: all items minus the user's positives
        pos_set = set(user_all_pos_items.get(u, []))
        # ensure this positive is treated as positive even if not in user_all_pos_items
        pos_set.add(pos)

        # sample negatives by rejection sampling to avoid heavy setdiff for large item sets
        negs = set()
        attempts = 0
        max_attempts = num_neg * 10  # safety cap
        while len(negs) < num_neg and attempts < max_attempts:
            candidate = rng.randrange(0, num_items)
            if candidate not in pos_set and candidate not in negs:
                negs.add(candidate)
            attempts += 1

        if not negs:
            # degenerate case: skip if we somehow can't find negatives
            continue

        neg_list = list(negs)

        # candidates: index 0 is always the positive item
        candidates = [pos] + neg_list
        user_arr = np.array([u], dtype=np.int64)
        item_arr = np.array(candidates, dtype=np.int64)

        scores = score_fn(user_arr, item_arr)  # [1, num_candidates]
        if scores.ndim != 2 or scores.shape[0] != 1 or scores.shape[1] != len(candidates):
            raise ValueError(
                f"score_fn must return shape [1, {len(candidates)}], "
                f"got {scores.shape}"
            )

        scores = scores[0]  # shape [num_candidates]

        # rank candidates by score descending
        ranking = np.argsort(-scores)  # indices into `candidates`

        # find rank of the positive (which is at index 0 in candidates)
        pos_rank = None
        for rank_idx, cand_idx in enumerate(ranking):
            if cand_idx == 0:
                pos_rank = rank_idx
                break

        if pos_rank is None:
            # shouldn't happen; but just in case, skip
            continue

        # Hit@k
        if pos_rank < k:
            hits += 1.0

        # NDCG@k (with a single positive, DCG=1/log2(rank+2), IDCG is 1/log2(1+1))
        if pos_rank < k:
            dcg = 1.0 / np.log2(pos_rank + 2.0)
            idcg = 1.0 / np.log2(1.0 + 1.0)
            ndcgs += dcg / idcg
        else:
            ndcgs += 0.0

        # MRR@k (reciprocal rank if within k, else 0)
        # if pos_rank < k:
        #     mrrs += 1.0 / (pos_rank + 1.0)
        # else:
        #     mrrs += 0.0
        mrrs += 1.0 / (pos_rank + 1.0)

        count += 1

    if count == 0:
        return 0.0, 0.0, 0.0

    return hits / count, ndcgs / count, mrrs / count
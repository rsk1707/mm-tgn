# baseline_scripts/run_sasrec_eval.py  (TF SASRec + unified eval)

import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "sasrec"))

from baseline_scripts.data_loader import load_gts_dataset  # GTS canonical loader
from baseline_scripts.eval_sampled import (
    build_user_pos_pairs_from_test,
    evaluate_sampled,
)
from sasrec.model import Model  # your TF SASRec model :contentReference[oaicite:11]{index=11}

def get_sasrec_itemnum_from_file():
    """
    Read sasrec/data/ml-modern-gts.txt and return the itemnum that SASRec
    was actually trained with (max 1-based item ID in the file).
    """
    data_path = os.path.join(PROJECT_ROOT, "sasrec", "data", "ml-modern-gts.txt")
    max_item_1 = 0
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                _, item_str = line.split()
            except ValueError:
                continue
            item_1 = int(item_str)
            if item_1 > max_item_1:
                max_item_1 = item_1

    if max_item_1 == 0:
        raise RuntimeError(f"No interactions found in {data_path}")
    return max_item_1  # this is exactly the `itemnum` SASRec used at train time


def make_sasrec_score_fn(sess, model, user_train_items, maxlen):
    """
    Wrap TF SASRec into a score_fn(users, items) -> [B, C] scores
    compatible with evaluate_sampled(). 
    """

    def _build_seq(u_internal: int) -> np.ndarray:
        """
        Build a 1D int32 array of length maxlen:
        left-padded with 0s, then 1-based item IDs from GTS train history.
        """
        hist_items_0 = user_train_items.get(u_internal, [])
        # convert to 1-based for SASRec
        seq = [i + 1 for i in hist_items_0]
        if len(seq) >= maxlen:
            seq = seq[-maxlen:]
        pad_len = maxlen - len(seq)
        padded = [0] * pad_len + seq
        return np.array(padded, dtype=np.int32).reshape(1, maxlen)  # [1, L]

    def score_fn(user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        # Our eval_sampled always calls with B=1, but we enforce shapes anyway.
        assert user_ids.ndim == 1 and item_ids.ndim == 1
        assert user_ids.shape[0] == 1

        u_internal = int(user_ids[0])       # 0-based GTS ID
        u_sas = np.array([u_internal + 1], dtype=np.int32)  # SASRec is 1-based

        seq = _build_seq(u_internal)        # [1, maxlen]

        # candidates: convert to 1-based item IDs for SASRec
        cand_1 = np.array([i + 1 for i in item_ids], dtype=np.int32)

        # Model.predict expects:
        #   u: [batch], seq: [batch, L], item_idx: [num_candidates]
        # and returns logits [batch, num_candidates]. :contentReference[oaicite:13]{index=13}
        logits = model.predict(sess, u_sas, seq, cand_1)

        # ensure shape [1, C]
        logits = np.asarray(logits)
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(cand_1):
            raise ValueError(
                f"SASRec logits shape mismatch, expected [1,{len(cand_1)}], got {logits.shape}"
            )

        return logits.astype(np.float32)

    return score_fn


def main():
    # # 1) Canonical GTS dataset
    # dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name="ml-modern")
    # num_users = dataset.num_users
    # num_items = dataset.num_items
    # user_train_items = dataset.user_train_items  # 0-based item IDs :contentReference[oaicite:14]{index=14}

    # # 2) Build user-pos pairs from GTS *test* split
    # user_pos_pairs = build_user_pos_pairs_from_test(dataset.user_test_items)  # :contentReference[oaicite:15]{index=15}

    # 1) Canonical GTS dataset (all splits)
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name="ml-modern")
    num_users = dataset.num_users
    num_items_global = dataset.num_items
    user_train_items = dataset.user_train_items  # 0-based item IDs

    # 2) Build user-pos pairs from GTS *test* split
    user_pos_pairs = build_user_pos_pairs_from_test(dataset.user_test_items)

    # --- NEW: get the SASRec item universe from the training file ---
    sas_itemnum = get_sasrec_itemnum_from_file()     # 1-based
    sas_num_items_internal = sas_itemnum             # internal IDs 0..sas_itemnum-1

    # Filter out test positives that SASRec has no embedding for
    user_pos_pairs = [(u, i) for (u, i) in user_pos_pairs if i < sas_num_items_internal]

    # 3) Restore trained SASRec model from a checkpoint
    #    Make sure you saved it in sasrec/main.py using tf.train.Saver().
    from argparse import Namespace
    args = Namespace(
        maxlen=50,
        hidden_units=64,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.5,
        l2_emb=0.0,
        lr=0.001,
    )

    # SASRec expects usernum, itemnum in *1-based* range
    usernum = num_users
    itemnum = sas_itemnum   # <= THIS: match training, not dataset.num_items

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model(usernum, itemnum, args)

    saver = tf.train.Saver()
    ckpt_path = os.path.join(PROJECT_ROOT, "sasrec", "ml-modern-gts_runs", "sasrec.ckpt")
    saver.restore(sess, ckpt_path)
    print("Restored SASRec checkpoint from:", ckpt_path)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    # with tf.variable_scope("SASRec"):
    #     model = Model(usernum, itemnum, args)

    # saver = tf.train.Saver()
    # ckpt_path = os.path.join(PROJECT_ROOT, "sasrec", "ml-modern-gts_runs", "sasrec.ckpt")
    # saver.restore(sess, ckpt_path)
    # print("Restored SASRec checkpoint from:", ckpt_path)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    # # IMPORTANT: no extra variable_scope wrapper here,
    # # Model already uses "SASRec" inside.
    # model = Model(usernum, itemnum, args)

    # saver = tf.train.Saver()
    # ckpt_path = os.path.join(PROJECT_ROOT, "sasrec", "ml-modern-gts_runs", "sasrec.ckpt")
    # saver.restore(sess, ckpt_path)
    # print("Restored SASRec checkpoint from:", ckpt_path)

    # 4) Make score_fn and run unified sampled eval
    # score_fn = make_sasrec_score_fn(sess, model, user_train_items, maxlen=args.maxlen)

    # hit, ndcg, mrr = evaluate_sampled(
    #     score_fn=score_fn,
    #     user_pos_pairs=user_pos_pairs,
    #     num_items=num_items,
    #     user_all_pos_items=dataset.user_all_pos_items,
    #     num_neg=100,
    #     k=10,
    #     seed=42,
    # )
    score_fn = make_sasrec_score_fn(sess, model, user_train_items, maxlen=args.maxlen)

    hit, ndcg, mrr = evaluate_sampled(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=sas_num_items_internal,             # NOT dataset.num_items
        user_all_pos_items=dataset.user_all_pos_items,
        num_neg=100,
        k=10,
        seed=42,
    )

    print(f"[SASRec | ml-modern] Hit@10={hit:.4f}, NDCG@10={ndcg:.4f}, MRR@10={mrr:.4f}")


if __name__ == "__main__":
    main()

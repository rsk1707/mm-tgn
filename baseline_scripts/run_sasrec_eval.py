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
    # 1) Canonical GTS dataset
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name="ml-modern")
    num_users = dataset.num_users
    num_items = dataset.num_items
    user_train_items = dataset.user_train_items  # 0-based item IDs :contentReference[oaicite:14]{index=14}

    # 2) Build user-pos pairs from GTS *test* split
    user_pos_pairs = build_user_pos_pairs_from_test(dataset.user_test_items)  # :contentReference[oaicite:15]{index=15}

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

    # SASRec expects usernum, itemnum in *1-based* range.
    usernum = num_users
    itemnum = num_items

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    with tf.variable_scope("SASRec"):
        model = Model(usernum, itemnum, args)

    saver = tf.train.Saver()
    ckpt_path = os.path.join(PROJECT_ROOT, "sasrec", "ml-modern-gts_runs", "sasrec.ckpt")
    saver.restore(sess, ckpt_path)
    print("Restored SASRec checkpoint from:", ckpt_path)

    # 4) Make score_fn and run unified sampled eval
    score_fn = make_sasrec_score_fn(sess, model, user_train_items, maxlen=args.maxlen)

    hit, ndcg, mrr = evaluate_sampled(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=num_items,
        user_all_pos_items=dataset.user_all_pos_items,
        num_neg=100,
        k=10,
        seed=42,
    )

    print(f"[SASRec | ml-modern] Hit@10={hit:.4f}, NDCG@10={ndcg:.4f}, MRR@10={mrr:.4f}")


if __name__ == "__main__":
    main()

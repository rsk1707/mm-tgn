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
from baseline_scripts.eval_subset import (
    load_eval_subset_pairs,
    compute_link_metrics_from_score_fn,
    evaluate_ranking_multi_k,
)
from sasrec.model import Model 

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
    # user_train_items = dataset.user_train_items  # 0-based item IDs 
    # # 2) Build user-pos pairs from GTS *test* split
    # user_pos_pairs = build_user_pos_pairs_from_test(dataset.user_test_items)

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

def eval_main():
    dataset_name = "ml-modern"

    # 1) Canonical GTS dataset (for mappings + user histories)
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name=dataset_name)
    num_users = dataset.num_users
    num_items_global = dataset.num_items
    user_train_items = dataset.user_train_items  # 0-based item IDs

    print(f"[GTS] num_users={num_users}, num_items_global={num_items_global}")

    # 2) Load eval subset from eval_samples/ml-modern_eval_sample.csv
    dataset_eval, user_pos_pairs = load_eval_subset_pairs(
        root_dir=PROJECT_ROOT,
        dataset_name=dataset_name,
        eval_relpath=os.path.join("eval_samples", "ml-modern_eval_sample.csv"),
    )

    # sanity check: we expect this to be the same dataset object
    assert dataset_eval.num_users == num_users
    assert dataset_eval.num_items == num_items_global

    print(f"[Eval subset] #pairs before SASRec filtering: {len(user_pos_pairs)}")

    # 3) SASRec item universe from training file
    sas_itemnum = get_sasrec_itemnum_from_file()  # 1-based
    sas_num_items_internal = sas_itemnum          # internal 0..itemnum-1

    # Filter eval pairs to only items SASRec has embeddings for
    user_pos_pairs = [(u, i) for (u, i) in user_pos_pairs if i < sas_num_items_internal]
    print(f"[Eval subset] #pairs after SASRec item filter: {len(user_pos_pairs)}")

    if not user_pos_pairs:
        raise RuntimeError("No eval pairs left after SASRec item filtering.")

    # Trim user_all_pos_items to SASRec item universe for negative sampling
    trimmed_user_all_pos = {
        u: [i for i in items if i < sas_num_items_internal]
        for u, items in dataset.user_all_pos_items.items()
    }

    # 4) Restore trained SASRec model
    from argparse import Namespace
    args = Namespace(
        maxlen=50,          # must match training setting
        hidden_units=64,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.5,
        l2_emb=0.0,
        lr=0.001,
    )
    
    usernum = num_users          # users are 1..usernum inside SASRec
    itemnum = sas_itemnum        # items are 1..itemnum

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model(usernum, itemnum, args)

    saver = tf.train.Saver()
    ckpt_path = os.path.join(PROJECT_ROOT, "sasrec", "ml-modern-gts_runs", "sasrec.ckpt")
    saver.restore(sess, ckpt_path)
    print("[SASRec] Restored checkpoint from:", ckpt_path)

    # 5) Build score_fn
    score_fn = make_sasrec_score_fn(sess, model, user_train_items, maxlen=args.maxlen)

    # 6) Link prediction metrics on eval subset (AP, AUC, link-MRR)
    link_metrics = compute_link_metrics_from_score_fn(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=sas_num_items_internal,
        user_all_pos_items=trimmed_user_all_pos,
        seed=42,
    )

    # 7) Ranking metrics on eval subset (Hit/Recall@10/20, NDCG@10/20, full MRR)
    ranking_metrics = evaluate_ranking_multi_k(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=sas_num_items_internal,
        user_all_pos_items=trimmed_user_all_pos,
        ks=(10, 20),
        num_neg=100,
        seed=42,
    )

    # 8) Print nicely
    print(
        "\n[SASRec | ml-modern | eval_samples/ml-modern_eval_sample.csv]"
    )

    print("Link prediction metrics:")
    print(f"  AP   : {link_metrics['AP']:.4f}")
    print(f"  AUC  : {link_metrics['AUC']:.4f}")
    print(f"  MRR  : {link_metrics['MRR']:.4f}")

    print("\nRanking metrics (sampled, 1 pos + 100 negs):")
    print(f"  Recall@10 / Hit@10: {ranking_metrics['recall@10']:.4f}")
    print(f"  Recall@20 / Hit@20: {ranking_metrics['recall@20']:.4f}")
    print(f"  NDCG@10          : {ranking_metrics['ndcg@10']:.4f}")
    print(f"  NDCG@20          : {ranking_metrics['ndcg@20']:.4f}")
    print(f"  MRR              : {ranking_metrics['mrr']:.4f}")


if __name__ == "__main__":
    # main()
    eval_main()

#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np
import torch
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Core helpers you already have
from baseline_scripts.data_loader import load_gts_dataset, DATASET_SCHEMAS
from baseline_scripts.eval_sampled import (
    build_user_pos_pairs_from_test,
    evaluate_sampled,
)
from baseline_scripts.eval_subset import (
    load_eval_subset_pairs,
    compute_link_metrics_from_score_fn,
    evaluate_ranking_multi_k,
)

DATASETS = {"ml-modern", "amazon-cloth", "amazon-sports"}
MODELS = {"lightgcn", "mmgcn", "sasrec"}


def get_mmgcn_feature_root(dataset_name):
    """
    Returns (feature_dir_name, prefix) for the SOTA embeddings.

    - ml-modern:
        features/ml-modern/sota/ml-modern_ids.npy
    - amazon-cloth:
        features/amazon-cloth-features/sota/amazon-cloth_ids.npy
    - amazon-sports:
        features/amazon-sports-features/sota/amazon-sports_ids.npy
    """
    if dataset_name == "ml-modern":
        return "ml-modern", "ml-modern"
    elif dataset_name == "amazon-cloth":
        return "amazon-cloth-features", "amazon-cloth"
    elif dataset_name == "amazon-sports":
        return "amazon-sports-features", "amazon-sports"
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}")


def run_sampled_eval(dataset_name, score_fn):
    print("\nSampled evaluation on GTS test split")
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name=dataset_name)

    user_pos_pairs = build_user_pos_pairs_from_test(dataset)
    metrics = evaluate_sampled(
        dataset,
        user_pos_pairs,
        score_fn,
        num_negative=100,
        ks=(10, 20),
    )
    print("Link prediction metrics:")
    print(f"  AP   : {metrics['link']['AP']:.4f}")
    print(f"  AUC  : {metrics['link']['AUC']:.4f}")
    print(f"  MRR  : {metrics['link']['MRR']:.4f}")

    print("\nRanking metrics (sampled, 1 pos + 100 negs):")
    print(f"  Recall@10 / Hit@10: {metrics['ranking']['recall@10']:.4f}")
    print(f"  Recall@20 / Hit@20: {metrics['ranking']['recall@20']:.4f}")
    print(f"  NDCG@10          : {metrics['ranking']['ndcg@10']:.4f}")
    print(f"  NDCG@20          : {metrics['ranking']['ndcg@20']:.4f}")
    print(f"  MRR              : {metrics['ranking']['mrr']:.4f}")

def run_eval_subset(dataset_name, score_fn, max_user_id=None, max_item_id=None):
    print("\nEvaluation on eval_samples_final subset")

    rel_path = os.path.join(
        "eval_samples_final",
        dataset_name,
        f"{dataset_name}_eval_sample.csv",
    )

    dataset, user_pos_pairs = load_eval_subset_pairs(
        root_dir=PROJECT_ROOT,
        dataset_name=dataset_name,
        eval_relpath=rel_path,
    )

    #(cold-start IDs not in train).
    if max_user_id is not None or max_item_id is not None:
        before = len(user_pos_pairs)
        filtered = []
        for u, i in user_pos_pairs:
            if max_user_id is not None and u >= max_user_id:
                continue
            if max_item_id is not None and i >= max_item_id:
                continue
            filtered.append((u, i))
        dropped = before - len(filtered)
        if dropped > 0:
            print(
                f"[eval_subset] Dropped {dropped} / {before} eval pairs "
                f"due to out-of-range IDs (max_user_id={max_user_id}, "
                f"max_item_id={max_item_id})."
            )
        user_pos_pairs = filtered

    # Use the embedding item-count for negative sampling if provided,
    # so we never sample items that lack embeddings.
    if max_item_id is not None:
        num_items_eval = min(dataset.num_items, max_item_id)
    else:
        num_items_eval = dataset.num_items

    link_metrics = compute_link_metrics_from_score_fn(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=num_items_eval,
        user_all_pos_items=dataset.user_all_pos_items,
        seed=42,
        num_neg=100,
    )

    print("Link prediction metrics:")
    print(f"  AP   : {link_metrics['AP']:.4f}")
    print(f"  AUC  : {link_metrics['AUC']:.4f}")
    print(f"  MRR  : {link_metrics['MRR']:.4f}")

    rank_metrics = evaluate_ranking_multi_k(
        score_fn=score_fn,
        user_pos_pairs=user_pos_pairs,
        num_items=num_items_eval,
        user_all_pos_items=dataset.user_all_pos_items,
        ks=(10, 20),
        num_neg=100,
        seed=42,
    )

    print("\nRanking metrics (full ranking with negative sampling):")
    print(f"  Recall@10 / Hit@10: {rank_metrics['recall@10']:.4f}")
    print(f"  Recall@20 / Hit@20: {rank_metrics['recall@20']:.4f}")
    print(f"  NDCG@10          : {rank_metrics['ndcg@10']:.4f}")
    print(f"  NDCG@20          : {rank_metrics['ndcg@20']:.4f}")
    print(f"  MRR              : {rank_metrics['mrr']:.4f}")

def eval_lightgcn(dataset_name):
    print(f"\nEvaluating LightGCN on {dataset_name}")
    gts_name = dataset_name + "-gts"

    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name=dataset_name)
    num_users_gts, num_items_gts = dataset.num_users, dataset.num_items

    emb_dir = os.path.join(PROJECT_ROOT, "lightgcn", "Data", gts_name)
    user_emb_path = os.path.join(emb_dir, "user_emb.npy")
    item_emb_path = os.path.join(emb_dir, "item_emb.npy")

    if not (os.path.exists(user_emb_path) and os.path.exists(item_emb_path)):
        raise FileNotFoundError(f"Missing LightGCN embeddings in {emb_dir}")

    user_emb = np.load(user_emb_path)  # [num_users_emb, dim]
    item_emb = np.load(item_emb_path)  # [num_items_emb, dim]

    num_users_emb, dim_u = user_emb.shape
    num_items_emb, dim_i = item_emb.shape

    if dim_u != dim_i:
        raise ValueError(
            f"Embedding dim mismatch: user_emb dim={dim_u}, item_emb dim={dim_i}"
        )

    # Soft-check shapes and warn if they don't match GTS (e.g., amazon-sports)
    if num_users_emb != num_users_gts or num_items_emb != num_items_gts:
        print(
            "[WARN] LightGCN embedding counts differ from GTS dataset counts:\n"
            f"       GTS users={num_users_gts}, emb users={num_users_emb}\n"
            f"       GTS items={num_items_gts}, emb items={num_items_emb}\n"
            "       Will restrict eval to IDs < emb sizes and drop out-of-range pairs."
        )
    else:
        print("[INFO] LightGCN embedding shapes match GTS counts exactly.")

    def score_fn(user_ids, item_ids):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)

        # One user, many items: return [1, M]
        if user_ids.shape[0] == 1:
            u_vec = user_emb[user_ids[0]]          # [D]
            i_mat = item_emb[item_ids]            # [M, D]
            scores = np.sum(i_mat * u_vec[None, :], axis=1)  # [M]
            return scores[None, :]                # [1, M]

        # Fallback: element-wise scores for each (u, i) pair
        u_mat = user_emb[user_ids]                # [N, D]
        i_mat = item_emb[item_ids]                # [N, D]
        scores = np.sum(u_mat * i_mat, axis=1)    # [N]
        return scores

    run_eval_subset(
        dataset_name,
        score_fn,
        max_user_id=num_users_emb,
        max_item_id=num_items_emb,
    )


def build_mmgcn_modal_features(dataset_name, dataset, num_items_model):
    feature_root, prefix = get_mmgcn_feature_root(dataset_name)
    sota_dir = os.path.join(PROJECT_ROOT, "features", feature_root, "sota")

    ids_path = os.path.join(sota_dir, f"{prefix}_ids.npy")
    img_path = os.path.join(sota_dir, f"{prefix}_image_siglip.npy")
    txt_path = os.path.join(sota_dir, f"{prefix}_text_efficient.npy")

    if not (os.path.exists(ids_path) and os.path.exists(img_path) and os.path.exists(txt_path)):
        raise FileNotFoundError(
            f"Missing SOTA feature files for {dataset_name} under {sota_dir}"
        )

    # ids: raw item ids (movieId or asin), typically strings for Amazon and
    # strings-of-ints for MovieLens.
    ids = np.load(ids_path, allow_pickle=True)
    img_feat = np.load(img_path)  # [N_items_feat, D_v]
    txt_feat = np.load(txt_path)  # [N_items_feat, D_t]
    assert img_feat.shape[0] == txt_feat.shape[0] == ids.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id_to_feat_idx = {str(raw_id): i for i, raw_id in enumerate(ids)}

    v_feat_full = torch.zeros(
        num_items_model,
        img_feat.shape[1],
        dtype=torch.float32,
        device=device,
    )
    t_feat_full = torch.zeros(
        num_items_model,
        txt_feat.shape[1],
        dtype=torch.float32,
        device=device,
    )

    covered = 0
    missing = 0

    # We assume MMGCN uses the same global item_id space for its items 0..num_items_model-1.
    for raw_item, gts_item_id in dataset.item2id.items():
        if gts_item_id >= num_items_model:
            # Item exists in GTS but not in MMGCN's trained item universe
            continue

        key = str(raw_item)
        if key in id_to_feat_idx:
            j = id_to_feat_idx[key]
            v_feat_full[gts_item_id] = torch.from_numpy(img_feat[j]).to(
                device=device, dtype=torch.float32
            )
            t_feat_full[gts_item_id] = torch.from_numpy(txt_feat[j]).to(
                device=device, dtype=torch.float32
            )
            covered += 1
        else:
            missing += 1

    print(
        f"[MMGCN] SOTA feature coverage over MMGCN items: "
        f"covered={covered}, missing={missing}, total_model_items={num_items_model}"
    )

    if missing > 0:
        print("[MMGCN] Warning: some items missing SOTA features; using zeros.")

    return v_feat_full, t_feat_full

def eval_mmgcn(dataset_name):
    """
    MMGCN eval:

    - Loads GTS dataset for metadata (num_users/num_items, item2id).
    - Loads train edges + user_item_dict from mmgcn/Data/<dataset>-gts.
    - Loads SOTA visual/text features and aligns them to MMGCN's item indices.
    - Instantiates MMGCN Net exactly like training.
    - Loads trained checkpoint mmgcn_<dataset>-gts.pt.
    - Evaluates on eval_samples_final subset.
    """
    print(f"\nEvaluating MMGCN on {dataset_name}")
    gts_name = dataset_name + "-gts"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mmgcn_root = os.path.join(PROJECT_ROOT, "mmgcn")
    if mmgcn_root not in sys.path:
        sys.path.append(mmgcn_root)

    import Dataset as mmgcn_Dataset   # mmgcn/Dataset.py
    import Model_MMGCN as mmgcn_Model  # mmgcn/Model_MMGCN.py

    mmgcn_data_load = mmgcn_Dataset.data_load
    MMGCNNet = mmgcn_Model.Net

    print("\nSTEP 1: Load GTS dataset (for eval metadata)")
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name=dataset_name)
    num_users_gts, num_items_gts = dataset.num_users, dataset.num_items
    print(f"  [GTS] num_users={num_users_gts}, num_items={num_items_gts}")

    print("\nSTEP 2: Load MMGCN graph structure (train.npy, user_item_dict)")
    mmgcn_data_dir = os.path.join(PROJECT_ROOT, "mmgcn", "Data", gts_name)
    num_user_m, num_item_m, train_edge, user_item_dict, _, _, _ = mmgcn_data_load(
        mmgcn_data_dir, has_v=False, has_a=False, has_t=False
    )
    print(f"  [MMGCN] num_users={num_user_m}, num_items={num_item_m}")

    if num_user_m != num_users_gts or num_item_m != num_items_gts:
        print(
            "[WARN] Mismatch between GTS and MMGCN counts:\n"
            f"       GTS users={num_users_gts}, MMGCN users={num_user_m}\n"
            f"       GTS items={num_items_gts}, MMGCN items={num_item_m}\n"
            "       Proceeding with MMGCN counts and dropping out-of-range eval IDs."
        )

    num_users_model = num_user_m
    num_items_model = num_item_m

    print("\nSTEP 3: Load SOTA modality features and align to MMGCN item indices")
    v_feat, t_feat = build_mmgcn_modal_features(dataset_name, dataset, num_items_model)
    a_feat = None  # has_a False in training

    print("\nSTEP 4: Instantiate MMGCN model and load checkpoint")
    batch_size = 1024        
    dim_x = 64         
    num_layer = 2
    weight_decay = 3e-5

    words_tensor = None        # same as training

    model = MMGCNNet(
        v_feat,
        a_feat,
        t_feat,   
        words_tensor,   
        train_edge,        
        batch_size,       
        num_users_model,   
        num_items_model, 
        "mean",       
        "False",     
        num_layer,    
        True,    
        user_item_dict, 
        weight_decay,    
        dim_x,  
    ).to(device)

    ckpt_name = f"mmgcn_{gts_name}.pt"
    ckpt_path = os.path.join(mmgcn_data_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"MMGCN checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[MMGCN] Loaded checkpoint from {ckpt_path}")

    print("\nSTEP 5: Forward pass to get user/item embeddings")
    with torch.no_grad():
        _ = model.forward()  # populates model.result
        all_emb = model.result.detach().cpu().numpy()  # [num_users_model+num_items_model, D]

    user_emb = all_emb[:num_users_model]
    item_emb = all_emb[num_users_model:]

    assert user_emb.shape[0] == num_users_model
    assert item_emb.shape[0] == num_items_model

    def score_fn(user_ids, item_ids):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)

        if user_ids.shape[0] == 1:
            u_vec = user_emb[user_ids[0]]          # [D]
            i_mat = item_emb[item_ids]            # [M, D]
            scores = np.sum(i_mat * u_vec[None, :], axis=1)  # [M]
            return scores[None, :]                # [1, M]

        u_mat = user_emb[user_ids]                # [N, D]
        i_mat = item_emb[item_ids]                # [N, D]
        scores = np.sum(u_mat * i_mat, axis=1)    # [N]
        return scores

    run_eval_subset(
        dataset_name,
        score_fn,
        max_user_id=num_users_model,
        max_item_id=num_items_model,
    )


def get_sasrec_itemnum_from_file(txt_path):
    usernum = 0
    itemnum = 0
    with open(txt_path, "r") as f:
        for line in f:
            toks = line.strip().split(" ")
            if len(toks) < 2:
                continue
            u = int(toks[0])
            items = [int(x) for x in toks[1:]]
            usernum = max(usernum, u)
            itemnum = max(itemnum, max(items))
    usernum += 1
    itemnum += 1
    return usernum, itemnum

def infer_sasrec_itemnum_from_ckpt(ckpt_prefix):
    reader = tf.train.NewCheckpointReader(ckpt_prefix)
    var_to_shape = reader.get_variable_to_shape_map()

    itemnum = None
    for name, shape in var_to_shape.items():
        if "SASRec/input_embeddings/lookup_table" in name and "Adam" not in name:
            itemnum = shape[0]

    if itemnum is None:
        raise RuntimeError(
            f"Could not infer itemnum from checkpoint {ckpt_prefix}; "
            f"lookup_table vars: "
            f"{[(n, s) for n, s in var_to_shape.items() if 'lookup_table' in n]}"
        )

    return itemnum

def get_sasrec_itemnum_from_file_for_dataset(dataset_name):
    gts_name = dataset_name + "-gts"
    data_path = os.path.join(PROJECT_ROOT, "sasrec", "data", f"{gts_name}.txt")

    max_item_1 = 0
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) <= 1:
                continue
            for item_str in parts[1:]:
                try:
                    item_1 = int(item_str)
                except ValueError:
                    continue
                if item_1 > max_item_1:
                    max_item_1 = item_1

    if max_item_1 == 0:
        raise RuntimeError(f"No interactions found in {data_path}")

    return max_item_1


def eval_sasrec(dataset_name):
    print(f"\nEvaluating SASRec on {dataset_name}")
    gts_name = dataset_name + "-gts"

    sasrec_root = os.path.join(PROJECT_ROOT, "sasrec")
    if sasrec_root not in sys.path:
        sys.path.append(sasrec_root)

    from sasrec.model import Model  # original TF SASRec
    dataset = load_gts_dataset(root_dir=PROJECT_ROOT, dataset_name=dataset_name)
    num_users = dataset.num_users
    print(f"[GTS] num_users={num_users}, num_items_global={dataset.num_items}")

    def get_sasrec_itemnum_from_file_for_dataset(dataset_name_local):
        gts_name_local = dataset_name_local + "-gts"
        data_path = os.path.join(PROJECT_ROOT, "sasrec", "data", f"{gts_name_local}.txt")

        max_item_1 = 0
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) <= 1:
                    continue
                for item_str in parts[1:]:
                    try:
                        item_1 = int(item_str)
                    except ValueError:
                        continue
                    if item_1 > max_item_1:
                        max_item_1 = item_1

        if max_item_1 == 0:
            raise RuntimeError(f"No interactions found in {data_path}")
        return max_item_1  # 1-based max id

    sas_itemnum_txt = get_sasrec_itemnum_from_file_for_dataset(dataset_name)
    print(f"[SASRec] sas_itemnum(from txt)={sas_itemnum_txt}")

    ckpt_dir = os.path.join(PROJECT_ROOT, "sasrec", f"{gts_name}_runs")
    ckpt_prefix = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_prefix is None:
        print(f"[SASRec] No TF checkpoint found in {ckpt_dir}. Skipping SASRec eval.")
        return
    print("[SASRec] Using checkpoint:", ckpt_prefix)

    reader = tf.train.NewCheckpointReader(ckpt_prefix)
    emb_shape = reader.get_tensor("SASRec/input_embeddings/lookup_table").shape
    ckpt_vocab_size = emb_shape[0]  # this is itemnum_train + 1
    itemnum = ckpt_vocab_size - 1   # item ids are 1..itemnum in SASRec

    if itemnum != sas_itemnum_txt:
        print(
            f"[WARN] sas_itemnum(from txt)={sas_itemnum_txt} "
            f"!= ckpt_vocab_size-1={itemnum}; "
            f"using itemnum={itemnum} from checkpoint."
        )
    else:
        print(f"[SASRec] txt and ckpt agree on itemnum={itemnum}")

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

    usernum = num_users  # users are 1..usernum internally
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model(usernum, itemnum, args)

    saver = tf.train.Saver()
    saver.restore(sess, ckpt_prefix)
    print("[SASRec] Restored checkpoint from:", ckpt_prefix)

    from baseline_scripts.run_sasrec_eval import make_sasrec_score_fn
    score_fn = make_sasrec_score_fn(
        sess,
        model,
        dataset.user_train_items,
        maxlen=args.maxlen,
    )

    run_eval_subset(
        dataset_name,
        score_fn,
        max_user_id=usernum,   # users: 0..usernum-1
        max_item_id=itemnum,   # items: 0..itemnum-1 (0-based GTS IDs)
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=sorted(MODELS),
        help="Which model to evaluate: lightgcn | mmgcn | sasrec",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASETS),
        help="Which dataset to evaluate on.",
    )
    args = parser.parse_args()

    if args.model == "lightgcn":
        eval_lightgcn(args.dataset)
    elif args.model == "mmgcn":
        eval_mmgcn(args.dataset)
    elif args.model == "sasrec":
        eval_sasrec(args.dataset)
    else:
        raise ValueError(f"Unknown model {args.model}")


if __name__ == "__main__":
    main()
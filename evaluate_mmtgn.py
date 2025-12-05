#!/usr/bin/env python3
"""
Standalone Evaluation Script for MM-TGN

This script loads a trained checkpoint and runs comprehensive evaluation:
- Link prediction metrics (AP, AUC, MRR)
- Ranking metrics (Recall@K, NDCG@K)
- Transductive vs Inductive splits

Usage:
    python evaluate_mmtgn.py \
        --checkpoint checkpoints/ml_vanilla_20241204/best_model.pt \
        --data-dir data/processed \
        --dataset ml-modern \
        --n-neg-eval 100 \
        --eval-sample-size 5000
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from dataset import load_dataset
from mmtgn import create_mmtgn
from utils.utils import RandEdgeSampler
from utils.metrics import compute_all_ranking_metrics


def setup_logging(log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def compute_link_pred_metrics(pos_probs: np.ndarray, neg_probs: np.ndarray) -> dict:
    """Compute link prediction metrics (AP, AUC, MRR)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    
    labels = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    scores = np.concatenate([pos_probs, neg_probs])
    
    # Filter out NaN/Inf
    valid_mask = np.isfinite(scores)
    if not valid_mask.all():
        labels = labels[valid_mask]
        scores = scores[valid_mask]
    
    if len(np.unique(labels)) < 2:
        return {"AP": 0.0, "AUC": 0.5, "MRR": 0.0}
    
    ap = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    # MRR: pair-wise comparison (pos vs neg)
    mrr_sum = 0.0
    n_pairs = min(len(pos_probs), len(neg_probs))
    for i in range(n_pairs):
        rank = 1 if pos_probs[i] > neg_probs[i] else 2
        mrr_sum += 1.0 / rank
    mrr = mrr_sum / n_pairs if n_pairs > 0 else 0.0
    
    return {"AP": ap, "AUC": auc, "MRR": mrr}


@torch.no_grad()
def evaluate_link_prediction(model, data, negative_sampler, batch_size, n_neighbors, device):
    """Fast link prediction evaluation (AP, AUC, MRR)."""
    model.eval()
    
    sources = data.sources
    destinations = data.destinations
    timestamps = data.timestamps
    edge_idxs = data.edge_idxs
    
    all_pos_probs = []
    all_neg_probs = []
    
    n_batches = (len(sources) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Link pred eval", leave=False):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(sources))
        
        batch_sources = sources[start:end]
        batch_destinations = destinations[start:end]
        batch_timestamps = timestamps[start:end]
        batch_edge_idxs = edge_idxs[start:end]
        
        # Sample negatives
        neg_destinations = negative_sampler.sample(len(batch_sources))
        
        # Get probabilities
        pos_prob, neg_prob = model.compute_edge_probabilities(
            source_nodes=batch_sources,
            destination_nodes=batch_destinations,
            negative_nodes=neg_destinations,
            edge_times=batch_timestamps,
            edge_idxs=batch_edge_idxs,
            n_neighbors=n_neighbors
        )
        
        all_pos_probs.append(pos_prob.squeeze().cpu().numpy())
        all_neg_probs.append(neg_prob.squeeze().cpu().numpy())
    
    pos_probs = np.concatenate(all_pos_probs)
    neg_probs = np.concatenate(all_neg_probs)
    
    return compute_link_pred_metrics(pos_probs, neg_probs)


@torch.no_grad()
def evaluate_ranking(
    model,
    data,
    all_items: np.ndarray,
    n_neighbors: int,
    device: str,
    n_negatives: int = 100,
    ranking_batch_size: int = 100,
    eval_sample_size: int = None
):
    """
    Full ranking evaluation with negative sampling.
    
    This is slow but provides accurate Recall@K, NDCG@K metrics.
    """
    model.eval()
    
    sources = data.sources
    destinations = data.destinations
    timestamps = data.timestamps
    edge_idxs = data.edge_idxs
    
    # Sample subset for faster evaluation (FIXED SEED for reproducibility)
    if eval_sample_size and eval_sample_size < len(sources):
        rng = np.random.RandomState(42)  # Fixed seed for fair comparison
        indices = rng.choice(len(sources), eval_sample_size, replace=False)
        indices = np.sort(indices)  # Keep temporal order
        sources = sources[indices]
        destinations = destinations[indices]
        timestamps = timestamps[indices]
        edge_idxs = edge_idxs[indices]
    
    n_samples = len(sources)
    
    # Pre-sample negatives
    all_neg_items = np.random.choice(all_items, size=(n_samples, n_negatives), replace=True)
    
    all_pos_scores = []
    all_neg_scores = []
    
    # Backup initial memory state
    initial_memory_backup = model.backup_memory()
    
    n_batches = (n_samples + ranking_batch_size - 1) // ranking_batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Ranking eval"):
        start = batch_idx * ranking_batch_size
        end = min(start + ranking_batch_size, n_samples)
        batch_size = end - start
        
        batch_sources = sources[start:end]
        batch_destinations = destinations[start:end]
        batch_timestamps = timestamps[start:end]
        batch_edge_idxs = edge_idxs[start:end]
        batch_neg_items = all_neg_items[start:end]
        
        # Backup memory before this batch
        batch_memory_backup = model.backup_memory()
        
        # Score positive edges (updates memory)
        pos_prob, _ = model.compute_edge_probabilities(
            source_nodes=batch_sources,
            destination_nodes=batch_destinations,
            negative_nodes=batch_destinations,  # dummy
            edge_times=batch_timestamps,
            edge_idxs=batch_edge_idxs,
            n_neighbors=n_neighbors
        )
        pos_scores = pos_prob.squeeze().cpu().numpy()
        all_pos_scores.append(pos_scores)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Score negatives in chunks (restore memory for each chunk)
        neg_scores = np.zeros((batch_size, n_negatives))
        neg_chunk_size = 5  # Process 5 negatives at a time
        
        for chunk_start in range(0, n_negatives, neg_chunk_size):
            chunk_end = min(chunk_start + neg_chunk_size, n_negatives)
            chunk_negs = batch_neg_items[:, chunk_start:chunk_end]
            
            # Restore memory to pre-batch state for fair comparison
            model.restore_memory(batch_memory_backup)
            
            # Expand for batched scoring
            expanded_sources = np.repeat(batch_sources, chunk_end - chunk_start)
            expanded_timestamps = np.repeat(batch_timestamps, chunk_end - chunk_start)
            expanded_edge_idxs = np.repeat(batch_edge_idxs, chunk_end - chunk_start)
            expanded_negs = chunk_negs.flatten()
            
            # Score negatives
            neg_prob, _ = model.compute_edge_probabilities(
                source_nodes=expanded_sources,
                destination_nodes=expanded_negs,
                negative_nodes=expanded_negs,  # dummy
                edge_times=expanded_timestamps,
                edge_idxs=expanded_edge_idxs,
                n_neighbors=n_neighbors,
                skip_memory_update=True
            )
            
            neg_scores_chunk = neg_prob.squeeze().cpu().numpy()
            neg_scores_chunk = neg_scores_chunk.reshape(batch_size, -1)
            neg_scores[:, chunk_start:chunk_end] = neg_scores_chunk
        
        all_neg_scores.append(neg_scores)
        
        # Restore memory for next batch
        model.restore_memory(batch_memory_backup)
        
        # Re-process positive edges to update memory correctly
        model.compute_edge_probabilities(
            source_nodes=batch_sources,
            destination_nodes=batch_destinations,
            negative_nodes=batch_destinations,
            edge_times=batch_timestamps,
            edge_idxs=batch_edge_idxs,
            n_neighbors=n_neighbors
        )
    
    # Restore initial memory state
    model.restore_memory(initial_memory_backup)
    
    pos_scores = np.concatenate(all_pos_scores)
    neg_scores_matrix = np.vstack(all_neg_scores)
    
    return compute_all_ranking_metrics(pos_scores, neg_scores_matrix)


def main():
    parser = argparse.ArgumentParser(description="MM-TGN Evaluation")
    
    # Required
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (best_model.pt)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to processed data directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., ml-modern)")
    
    # Evaluation settings
    parser.add_argument("--n-neg-eval", type=int, default=100,
                        help="Number of negative samples per positive for ranking")
    parser.add_argument("--eval-sample-size", type=int, default=None,
                        help="Sample size for ranking evaluation (None = full)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Batch size for evaluation")
    parser.add_argument("--ranking-batch-size", type=int, default=100,
                        help="Batch size for ranking evaluation (smaller = less memory)")
    
    # Model settings (loaded from checkpoint, but can override)
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="Number of neighbors for temporal attention")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: same as checkpoint)")
    
    args = parser.parse_args()
    
    # Setup
    checkpoint_path = Path(args.checkpoint)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(output_dir / "evaluation.log"))
    
    logger.info("=" * 70)
    logger.info("MM-TGN Evaluation")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {args.device}")
    
    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load checkpoint
    logger.info("\n📦 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})
    
    logger.info(f"  Trained for {checkpoint.get('epoch', '?')} epochs")
    logger.info(f"  Best val AP: {checkpoint.get('val_ap', '?'):.4f}")
    logger.info(f"  Node feature type: {saved_args.get('node_feature_type', 'sota')}")
    logger.info(f"  MM fusion: {saved_args.get('mm_fusion', 'mlp')}")
    
    # Load dataset
    logger.info("\n📊 Loading dataset...")
    dataset = load_dataset(args.data_dir, args.dataset)
    logger.info(f"{dataset}")
    
    # Get test splits
    test_data = dataset.test_data
    transductive_test_data = dataset.get_transductive_test_data()
    inductive_test_data = dataset.get_inductive_test_data()
    all_items = dataset.get_all_items()
    
    logger.info(f"\n📊 Test Splits:")
    logger.info(f"  Total:        {len(test_data):,}")
    logger.info(f"  Transductive: {len(transductive_test_data):,}")
    logger.info(f"  Inductive:    {len(inductive_test_data):,}")
    
    # Negative sampler
    full_data = dataset.get_full_data()
    full_neg_sampler = RandEdgeSampler(
        full_data.sources, full_data.destinations, seed=42
    )
    
    # Create model with same config
    logger.info("\n🏗️ Creating model...")
    
    # Get saved args or use defaults
    node_feature_type = saved_args.get('node_feature_type', 'sota')
    mm_fusion = saved_args.get('mm_fusion', 'mlp')
    embedding_dim = saved_args.get('embedding_dim', 172)
    n_layers = saved_args.get('n_layers', 2)
    n_heads = saved_args.get('n_heads', 2)
    memory_dim = saved_args.get('memory_dim', 172)
    message_dim = saved_args.get('message_dim', 100)
    dropout = saved_args.get('dropout', 0.1)
    use_memory = saved_args.get('use_memory', True)
    use_hybrid = saved_args.get('use_hybrid', True)
    embedding_module = saved_args.get('embedding_module', 'graph_attention')
    
    use_random_item_features = (node_feature_type == "random")
    
    model = create_mmtgn(
        dataset=dataset,
        device=device,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_neighbors=args.n_neighbors,
        memory_dim=memory_dim,
        message_dim=message_dim,
        dropout=dropout,
        use_memory=use_memory,
        use_hybrid_features=use_hybrid,
        embedding_module_type=embedding_module,
        structural_dim=None,
        use_random_item_features=use_random_item_features,
        mm_fusion_mode=mm_fusion
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("✅ Model weights loaded")
    
    # Results dictionary
    results = {
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "node_feature_type": node_feature_type,
        "mm_fusion": mm_fusion,
        "n_neg_eval": args.n_neg_eval,
        "eval_sample_size": args.eval_sample_size,
        "timestamp": datetime.now().isoformat()
    }
    
    # =================================================================
    # LINK PREDICTION EVALUATION
    # =================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("LINK PREDICTION EVALUATION")
    logger.info("=" * 70)
    
    # Process train+val to warm up memory
    logger.info("\n🔥 Warming up memory with train+val data...")
    model.reset_memory()
    
    train_data = dataset.train_data
    val_data = dataset.val_data
    
    # Process training data in batches
    n_train = len(train_data)
    batch_size = args.batch_size
    
    for start in tqdm(range(0, n_train, batch_size), desc="Processing train"):
        end = min(start + batch_size, n_train)
        with torch.no_grad():
            model.compute_edge_probabilities(
                source_nodes=train_data.sources[start:end],
                destination_nodes=train_data.destinations[start:end],
                negative_nodes=train_data.destinations[start:end],
                edge_times=train_data.timestamps[start:end],
                edge_idxs=train_data.edge_idxs[start:end],
                n_neighbors=args.n_neighbors
            )
    
    # Process validation data
    n_val = len(val_data)
    for start in tqdm(range(0, n_val, batch_size), desc="Processing val"):
        end = min(start + batch_size, n_val)
        with torch.no_grad():
            model.compute_edge_probabilities(
                source_nodes=val_data.sources[start:end],
                destination_nodes=val_data.destinations[start:end],
                negative_nodes=val_data.destinations[start:end],
                edge_times=val_data.timestamps[start:end],
                edge_idxs=val_data.edge_idxs[start:end],
                n_neighbors=args.n_neighbors
            )
    
    logger.info("✅ Memory warmed up")
    
    # Backup memory state
    memory_backup = model.backup_memory()
    
    # --- Overall Test ---
    logger.info("\n📊 Overall Test Set:")
    test_metrics = evaluate_link_prediction(
        model, test_data, full_neg_sampler,
        args.batch_size, args.n_neighbors, device
    )
    logger.info(f"  AP: {test_metrics['AP']:.4f} | AUC: {test_metrics['AUC']:.4f} | MRR: {test_metrics['MRR']:.4f}")
    results["test_link_pred"] = test_metrics
    
    model.restore_memory(memory_backup)
    
    # --- Transductive ---
    if len(transductive_test_data) > 0:
        logger.info("\n📊 Transductive Test (Warm Users):")
        trans_metrics = evaluate_link_prediction(
            model, transductive_test_data, full_neg_sampler,
            args.batch_size, args.n_neighbors, device
        )
        logger.info(f"  AP: {trans_metrics['AP']:.4f} | AUC: {trans_metrics['AUC']:.4f} | MRR: {trans_metrics['MRR']:.4f}")
        results["transductive_link_pred"] = trans_metrics
        model.restore_memory(memory_backup)
    
    # --- Inductive ---
    if len(inductive_test_data) > 0:
        logger.info("\n📊 Inductive Test (Cold Start):")
        induct_metrics = evaluate_link_prediction(
            model, inductive_test_data, full_neg_sampler,
            args.batch_size, args.n_neighbors, device
        )
        logger.info(f"  AP: {induct_metrics['AP']:.4f} | AUC: {induct_metrics['AUC']:.4f} | MRR: {induct_metrics['MRR']:.4f}")
        results["inductive_link_pred"] = induct_metrics
        model.restore_memory(memory_backup)
    
    # Save partial results
    with open(output_dir / "results_linkpred.json", 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n💾 Link prediction results saved to {output_dir / 'results_linkpred.json'}")
    
    # =================================================================
    # RANKING EVALUATION
    # =================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("RANKING EVALUATION")
    logger.info("=" * 70)
    logger.info(f"  Negatives per positive: {args.n_neg_eval}")
    logger.info(f"  Sample size: {args.eval_sample_size or 'Full'}")
    
    model.restore_memory(memory_backup)
    
    # --- Overall Test Ranking ---
    logger.info("\n📊 Overall Test Ranking:")
    start_time = time.time()
    test_ranking = evaluate_ranking(
        model=model,
        data=test_data,
        all_items=all_items,
        n_neighbors=args.n_neighbors,
        device=device,
        n_negatives=args.n_neg_eval,
        ranking_batch_size=args.ranking_batch_size,
        eval_sample_size=args.eval_sample_size
    )
    elapsed = time.time() - start_time
    
    logger.info(f"  Recall@10:  {test_ranking.recall_at_10:.4f}")
    logger.info(f"  Recall@20:  {test_ranking.recall_at_20:.4f}")
    logger.info(f"  NDCG@10:    {test_ranking.ndcg_at_10:.4f}")
    logger.info(f"  NDCG@20:    {test_ranking.ndcg_at_20:.4f}")
    logger.info(f"  MRR:        {test_ranking.mrr:.4f}")
    logger.info(f"  (Completed in {elapsed/60:.1f} minutes)")
    
    results["test_ranking"] = {
        "recall_at_10": test_ranking.recall_at_10,
        "recall_at_20": test_ranking.recall_at_20,
        "ndcg_at_10": test_ranking.ndcg_at_10,
        "ndcg_at_20": test_ranking.ndcg_at_20,
        "mrr": test_ranking.mrr
    }
    
    model.restore_memory(memory_backup)
    
    # --- Transductive Ranking ---
    if len(transductive_test_data) > 0:
        logger.info("\n📊 Transductive Ranking:")
        trans_ranking = evaluate_ranking(
            model=model,
            data=transductive_test_data,
            all_items=all_items,
            n_neighbors=args.n_neighbors,
            device=device,
            n_negatives=args.n_neg_eval,
            ranking_batch_size=args.ranking_batch_size,
            eval_sample_size=args.eval_sample_size
        )
        logger.info(f"  Recall@10:  {trans_ranking.recall_at_10:.4f}")
        logger.info(f"  NDCG@10:    {trans_ranking.ndcg_at_10:.4f}")
        logger.info(f"  MRR:        {trans_ranking.mrr:.4f}")
        
        results["transductive_ranking"] = {
            "recall_at_10": trans_ranking.recall_at_10,
            "recall_at_20": trans_ranking.recall_at_20,
            "ndcg_at_10": trans_ranking.ndcg_at_10,
            "ndcg_at_20": trans_ranking.ndcg_at_20,
            "mrr": trans_ranking.mrr
        }
        model.restore_memory(memory_backup)
    
    # --- Inductive Ranking ---
    if len(inductive_test_data) > 0:
        logger.info("\n📊 Inductive Ranking:")
        induct_ranking = evaluate_ranking(
            model=model,
            data=inductive_test_data,
            all_items=all_items,
            n_neighbors=args.n_neighbors,
            device=device,
            n_negatives=args.n_neg_eval,
            ranking_batch_size=args.ranking_batch_size,
            eval_sample_size=args.eval_sample_size
        )
        logger.info(f"  Recall@10:  {induct_ranking.recall_at_10:.4f}")
        logger.info(f"  NDCG@10:    {induct_ranking.ndcg_at_10:.4f}")
        logger.info(f"  MRR:        {induct_ranking.mrr:.4f}")
        
        results["inductive_ranking"] = {
            "recall_at_10": induct_ranking.recall_at_10,
            "recall_at_20": induct_ranking.recall_at_20,
            "ndcg_at_10": induct_ranking.ndcg_at_10,
            "ndcg_at_20": induct_ranking.ndcg_at_20,
            "mrr": induct_ranking.mrr
        }
    
    # =================================================================
    # SAVE FINAL RESULTS
    # =================================================================
    
    results["completed"] = True
    
    with open(output_dir / "results_full.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📁 Results saved to: {output_dir}")
    logger.info(f"   - results_linkpred.json (fast metrics)")
    logger.info(f"   - results_full.json (all metrics)")
    
    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<20} {'Overall':>10} {'Transductive':>12} {'Inductive':>10}")
    logger.info("-" * 54)
    
    # Link pred
    logger.info(f"{'AP':<20} {test_metrics['AP']:>10.4f} {trans_metrics.get('AP', 0):>12.4f} {induct_metrics.get('AP', 0):>10.4f}")
    logger.info(f"{'AUC':<20} {test_metrics['AUC']:>10.4f} {trans_metrics.get('AUC', 0):>12.4f} {induct_metrics.get('AUC', 0):>10.4f}")
    
    # Ranking
    logger.info("-" * 54)
    logger.info(f"{'Recall@10':<20} {test_ranking.recall_at_10:>10.4f} {trans_ranking.recall_at_10 if 'trans_ranking' in dir() else 0:>12.4f} {induct_ranking.recall_at_10 if 'induct_ranking' in dir() else 0:>10.4f}")
    logger.info(f"{'NDCG@10':<20} {test_ranking.ndcg_at_10:>10.4f} {trans_ranking.ndcg_at_10 if 'trans_ranking' in dir() else 0:>12.4f} {induct_ranking.ndcg_at_10 if 'induct_ranking' in dir() else 0:>10.4f}")
    logger.info(f"{'MRR':<20} {test_ranking.mrr:>10.4f} {trans_ranking.mrr if 'trans_ranking' in dir() else 0:>12.4f} {induct_ranking.mrr if 'induct_ranking' in dir() else 0:>10.4f}")


if __name__ == "__main__":
    main()


"""
MM-TGN Training Script

End-to-end training pipeline for Multimodal Temporal Graph Networks.

Features:
1. Temporal training with proper memory management
2. BCE loss for stability (BPR optional)
3. Comprehensive evaluation:
   - Link Prediction: AUC, AP, MRR
   - Ranking Metrics: Recall@K, NDCG@K (with negative sampling)
   - Transductive vs Inductive breakdown
4. TensorBoard logging
5. Checkpoint management
"""

import argparse
import logging
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from dataset import TemporalDataset, load_dataset, Data
from mmtgn import MMTGN, create_mmtgn, bpr_loss, bce_loss
from utils.utils import (
    RandEdgeSampler, 
    EarlyStopMonitor, 
    get_neighbor_finder
)
from utils.metrics import (
    compute_all_ranking_metrics,
    NegativeSamplerForEval,
    RankingMetrics
)


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="MM-TGN Training")
    
    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with ml_*.csv, ml_*.npy, node_map.json")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., ml-modern)")
    
    # Model Architecture
    parser.add_argument("--embedding-dim", type=int, default=172,
                        help="Embedding dimension for all components")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="Number of graph attention layers")
    parser.add_argument("--n-heads", type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="Number of temporal neighbors to sample")
    parser.add_argument("--memory-dim", type=int, default=172,
                        help="Memory vector dimension")
    parser.add_argument("--message-dim", type=int, default=100,
                        help="Message dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="L2 regularization")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    
    # Loss
    parser.add_argument("--loss", type=str, default="bce",
                        choices=["bpr", "bce"],
                        help="Loss function (bce=stable default, bpr=ranking-optimized)")
    
    # Features
    parser.add_argument("--use-memory", action="store_true", default=True,
                        help="Use TGN memory module")
    parser.add_argument("--no-memory", action="store_false", dest="use_memory",
                        help="Disable memory module")
    parser.add_argument("--use-hybrid", action="store_true", default=True,
                        help="Use hybrid node features (learnable users + projected items)")
    parser.add_argument("--no-hybrid", action="store_false", dest="use_hybrid",
                        help="Use raw features directly")
    parser.add_argument("--embedding-module", type=str, default="graph_attention",
                        choices=["graph_attention", "graph_sum", "identity", "time"],
                        help="Type of embedding module")
    
    # =================================================================
    # ABLATION STUDY ARGUMENTS
    # =================================================================
    
    # Experiment A: Node Feature Type (Vanilla vs SOTA)
    parser.add_argument("--node-feature-type", type=str, default="sota",
                        choices=["sota", "baseline", "random"],
                        help="""
                        Node feature source for ablation studies:
                        - sota: SOTA multimodal (Qwen2+SigLIP, 2688-dim)
                        - baseline: Baseline encoders (MiniLM+ResNet, 1536-dim)
                        - random: Learnable random embeddings (no content info - lower bound)
                        """)
    
    # Experiment B: Input Feature Dimension
    parser.add_argument("--input-feat-dim", type=str, default="auto",
                        help="""
                        Input feature dimension:
                        - auto: Detect from loaded features
                        - Integer: Override with specific dimension
                        Used when switching between sota (2688) and baseline (1536) features
                        """)
    
    # Experiment C: Fusion Mode
    parser.add_argument("--fusion-mode", type=str, default="film",
                        choices=["film", "concat", "none"],
                        help="""
                        Feature fusion strategy:
                        - film: FiLM conditioning (γ⊙x + β)
                        - concat: Simple concatenation
                        - none: No fusion (temporal channel only)
                        """)
    
    # Evaluation
    parser.add_argument("--n-neg-eval", type=int, default=100,
                        help="Number of negatives for ranking evaluation")
    parser.add_argument("--eval-ranking", action="store_true", default=True,
                        help="Compute ranking metrics (Recall@K, NDCG@K)")
    parser.add_argument("--no-eval-ranking", action="store_false", dest="eval_ranking",
                        help="Skip ranking metrics (faster)")
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name (auto-generated if not provided)")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 for TGN)")
    
    return parser.parse_args()


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_file: Optional[str] = None):
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def compute_link_pred_metrics(
    pos_probs: np.ndarray, 
    neg_probs: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard link prediction metrics.
    
    Returns dict with: AP, AUC, MRR
    """
    # For AP and AUC, we need binary labels
    labels = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
    probs = np.concatenate([pos_probs, neg_probs])
    
    ap = average_precision_score(labels, probs)
    auc = roc_auc_score(labels, probs)
    
    # Mean Reciprocal Rank (simple 1:1 comparison)
    mrr = 0.0
    for i, pos_p in enumerate(pos_probs):
        rank = 1 + (neg_probs > pos_p).sum()
        mrr += 1.0 / rank
    mrr /= len(pos_probs)
    
    return {"AP": ap, "AUC": auc, "MRR": mrr}


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: MMTGN,
    data: Data,
    negative_sampler: RandEdgeSampler,
    optimizer: torch.optim.Optimizer,
    loss_fn: str,
    batch_size: int,
    n_neighbors: int,
    device: str,
    logger: logging.Logger
) -> float:
    """
    Train for one epoch.
    
    Returns average loss.
    """
    model.train()
    
    n_batches = (len(data.sources) + batch_size - 1) // batch_size
    total_loss = 0.0
    
    # Process in chronological order (CRITICAL for TGN)
    indices = np.arange(len(data.sources))
    
    pbar = tqdm(range(0, len(indices), batch_size), desc="Training", leave=False)
    
    for batch_idx, start_idx in enumerate(pbar):
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]
        
        # Get batch data
        sources = data.sources[batch_indices]
        destinations = data.destinations[batch_indices]
        timestamps = data.timestamps[batch_indices]
        edge_idxs = data.edge_idxs[batch_indices]
        
        # Sample negatives
        _, negatives = negative_sampler.sample(len(sources))
        
        # Forward pass
        optimizer.zero_grad()
        
        pos_prob, neg_prob = model.compute_edge_probabilities(
            sources, destinations, negatives,
            timestamps, edge_idxs, n_neighbors
        )
        
        # Compute loss
        if loss_fn == "bpr":
            loss = bpr_loss(pos_prob, neg_prob)
        else:
            loss = bce_loss(pos_prob, neg_prob)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Detach memory gradients for TBPTT
        model.detach_memory()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / n_batches


@torch.no_grad()
@torch.no_grad()
def evaluate_link_prediction(
    model: MMTGN,
    data: Data,
    negative_sampler: RandEdgeSampler,
    batch_size: int,
    n_neighbors: int,
    device: str
) -> Dict[str, float]:
    """
    Evaluate model on link prediction task.
    
    Returns metrics dict with AP, AUC, MRR.
    
    Note: MRR here compares each positive against ALL negatives (global ranking),
    which is different from pair-wise MRR in evaluate_ranking. The ranking
    metrics (Recall@K, NDCG@K) are more meaningful for recommendation.
    """
    model.eval()
    
    all_pos_probs = []
    all_neg_probs = []
    
    indices = np.arange(len(data.sources))
    
    for start_idx in range(0, len(indices), batch_size):
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]
        
        sources = data.sources[batch_indices]
        destinations = data.destinations[batch_indices]
        timestamps = data.timestamps[batch_indices]
        edge_idxs = data.edge_idxs[batch_indices]
        
        # Sample negatives
        _, negatives = negative_sampler.sample(len(sources))
        
        pos_prob, neg_prob = model.compute_edge_probabilities(
            sources, destinations, negatives,
            timestamps, edge_idxs, n_neighbors
        )
        
        all_pos_probs.append(pos_prob.cpu().numpy())
        all_neg_probs.append(neg_prob.cpu().numpy())
    
    pos_probs = np.concatenate(all_pos_probs)
    neg_probs = np.concatenate(all_neg_probs)
    
    return compute_link_pred_metrics(pos_probs, neg_probs)


@torch.no_grad()
def evaluate_ranking(
    model: MMTGN,
    data: Data,
    all_items: np.ndarray,
    batch_size: int,
    n_neighbors: int,
    n_negatives: int,
    device: str,
    seed: int = 42
) -> RankingMetrics:
    """
    Evaluate model with ranking metrics (Recall@K, NDCG@K).
    
    Uses negative sampling strategy: for each positive, rank among N negatives.
    
    Args:
        model: MM-TGN model
        data: Evaluation data
        all_items: Array of all item IDs for negative sampling
        batch_size: Batch size
        n_neighbors: Number of neighbors for TGN
        n_negatives: Number of negatives per positive
        device: Device
        seed: Random seed for negative sampling
    
    Returns:
        RankingMetrics dataclass
    """
    model.eval()
    
    neg_sampler = NegativeSamplerForEval(all_items, n_negatives=n_negatives, seed=seed)
    
    all_pos_scores = []
    all_neg_scores = []  # Will be [n_samples, n_negatives]
    
    indices = np.arange(len(data.sources))
    
    # Save initial memory state for the entire evaluation
    # This prevents memory corruption across batches
    initial_memory_backup = model.backup_memory()
    
    for start_idx in tqdm(range(0, len(indices), batch_size), desc="Ranking eval", leave=False):
        end_idx = min(start_idx + batch_size, len(indices))
        batch_indices = indices[start_idx:end_idx]
        batch_size_actual = len(batch_indices)
        
        sources = data.sources[batch_indices]
        destinations = data.destinations[batch_indices]
        timestamps = data.timestamps[batch_indices]
        edge_idxs = data.edge_idxs[batch_indices]
        
        # Sample multiple negatives per positive
        neg_items = neg_sampler.sample_negatives(destinations)  # [batch, n_negatives]
        
        # Backup memory before this batch (to restore after all scoring)
        batch_memory_backup = model.backup_memory()
        
        # Get positive scores: user → positive_item
        _, negatives_dummy = RandEdgeSampler(sources, destinations).sample(batch_size_actual)
        pos_prob, _ = model.compute_edge_probabilities(
            sources, destinations, negatives_dummy,
            timestamps, edge_idxs, n_neighbors
        )
        all_pos_scores.append(pos_prob.cpu().numpy())
        
        # Restore memory after positive scoring (memory was updated as side effect)
        model.restore_memory(batch_memory_backup)
        
        # Get negative scores (need to score each negative)
        batch_neg_scores = np.zeros((batch_size_actual, n_negatives), dtype=np.float32)
        
        for neg_idx in range(n_negatives):
            neg_items_col = neg_items[:, neg_idx]
            
            # Backup memory state since we're scoring the same batch multiple times
            memory_backup = model.backup_memory()
            
            # Score user → negative_item
            # compute_edge_probabilities(src, dst, neg) returns:
            #   pos_prob = P(src → dst)  <-- We want this! (score of negative item)
            #   neg_prob = P(src → neg)  <-- This is wrong (it scores destinations again)
            # BUG FIX: Use pos_prob (first return), not neg_prob
            neg_item_prob, _ = model.compute_edge_probabilities(
                sources, neg_items_col, destinations,  # dst=neg_items, neg=original_pos
                timestamps, edge_idxs, n_neighbors
            )
            batch_neg_scores[:, neg_idx] = neg_item_prob.cpu().numpy()
            
            # Restore memory
            model.restore_memory(memory_backup)
        
        all_neg_scores.append(batch_neg_scores)
    
    # Restore initial memory state
    model.restore_memory(initial_memory_backup)
    
    pos_scores = np.concatenate(all_pos_scores)
    neg_scores_matrix = np.vstack(all_neg_scores)
    
    return compute_all_ranking_metrics(pos_scores, neg_scores_matrix)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(args):
    """Main training function."""
    
    # Setup
    set_seed(args.seed)
    
    # Create run name
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"mmtgn_{args.dataset}_{timestamp}"
    
    # Directories
    log_dir = Path(args.log_dir) / args.run_name
    save_dir = Path(args.save_dir) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging
    logger = setup_logging(str(save_dir / "train.log"))
    writer = SummaryWriter(log_dir)
    
    logger.info("=" * 70)
    logger.info("MM-TGN Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"Run: {args.run_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Loss: {args.loss}")
    logger.info(f"Ranking Eval: {args.eval_ranking} (n_neg={args.n_neg_eval})")
    
    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # =================================================================
    # DATA LOADING
    # =================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("DATA LOADING")
    logger.info("=" * 70)
    
    dataset = load_dataset(args.data_dir, args.dataset)
    logger.info(f"\n{dataset}")
    
    # Get splits
    train_data = dataset.train_data
    val_data = dataset.val_data
    test_data = dataset.test_data
    full_data = dataset.get_full_data()
    
    # Get transductive/inductive test splits
    transductive_test_data = dataset.get_transductive_test_data()
    inductive_test_data = dataset.get_inductive_test_data()
    
    logger.info(f"\n📊 Data Splits:")
    logger.info(f"  Train:       {len(train_data):,}")
    logger.info(f"  Validation:  {len(val_data):,}")
    logger.info(f"  Test Total:  {len(test_data):,}")
    logger.info(f"    - Transductive: {len(transductive_test_data):,}")
    logger.info(f"    - Inductive:    {len(inductive_test_data):,}")
    
    # Get all items for negative sampling
    all_items = dataset.get_all_items()
    logger.info(f"\n📊 Items for negative sampling: {len(all_items):,}")
    
    # Negative samplers
    train_neg_sampler = RandEdgeSampler(
        train_data.sources, train_data.destinations, seed=args.seed
    )
    full_neg_sampler = RandEdgeSampler(
        full_data.sources, full_data.destinations, seed=args.seed
    )
    
    # =================================================================
    # MODEL CREATION
    # =================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("MODEL CREATION")
    logger.info("=" * 70)
    
    # =================================================================
    # ABLATION STUDY CONFIGURATION
    # =================================================================
    
    # Experiment A: Node Feature Type
    node_feature_type = args.node_feature_type
    if node_feature_type == "random":
        # VANILLA BASELINE: Use learnable random embeddings (no content info)
        logger.info("\n🧪 ABLATION MODE: Random Features (Vanilla Baseline)")
        logger.info("   Items will use learnable nn.Embedding instead of SOTA features")
        logger.info("   This establishes the LOWER BOUND for semantic features")
        use_random_item_features = True
    elif node_feature_type == "baseline":
        # BASELINE ENCODERS: Use MiniLM + ResNet (1536-dim)
        logger.info("\n🧪 ABLATION MODE: Baseline Features (MiniLM + ResNet)")
        logger.info("   Using 1536-dim baseline features instead of 2688-dim SOTA")
        use_random_item_features = False
    else:  # sota (default)
        logger.info("\n🧪 Feature Mode: SOTA (Qwen2 + SigLIP, 2688-dim)")
        use_random_item_features = False
    
    # Experiment C: Fusion Mode
    fusion_mode = args.fusion_mode
    logger.info(f"   Fusion Mode: {fusion_mode}")
    
    # Determine structural_dim based on fusion mode
    # (Currently Channel 2 is not available, so always None)
    structural_dim = None  # Channel 2 bypass mode
    
    model = create_mmtgn(
        dataset=dataset,
        device=device,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_neighbors=args.n_neighbors,
        memory_dim=args.memory_dim,
        message_dim=args.message_dim,
        dropout=args.dropout,
        use_memory=args.use_memory,
        use_hybrid_features=args.use_hybrid,
        embedding_module_type=args.embedding_module,
        structural_dim=structural_dim,
        use_random_item_features=use_random_item_features  # Ablation: random baseline
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\n🏗️ Model Configuration:")
    logger.info(f"  Total Parameters:     {n_params:,}")
    logger.info(f"  Trainable Parameters: {n_trainable:,}")
    logger.info(f"  Embedding Dim:        {args.embedding_dim}")
    logger.info(f"  Memory:               {args.use_memory}")
    logger.info(f"  Hybrid Features:      {args.use_hybrid}")
    logger.info(f"  Embedding Module:     {args.embedding_module}")
    logger.info(f"  Node Feature Type:    {node_feature_type}")
    logger.info(f"  Fusion Mode:          {fusion_mode}")
    logger.info(f"  FiLM (Channel 2):     {'ENABLED' if structural_dim else 'BYPASS MODE'}")
    
    # =================================================================
    # OPTIMIZER
    # =================================================================
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )  # Note: 'verbose' removed in PyTorch 2.x
    
    early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=True)
    
    # =================================================================
    # TRAINING LOOP
    # =================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING")
    logger.info("=" * 70)
    logger.info(f"\n🚀 Starting Training...")
    logger.info(f"  Epochs:       {args.epochs}")
    logger.info(f"  Batch Size:   {args.batch_size}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Loss Function: {args.loss}")
    
    best_val_ap = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*70}")
        
        # Reset memory at start of each epoch
        model.reset_memory()
        
        # Training
        train_loss = train_epoch(
            model=model,
            data=train_data,
            negative_sampler=train_neg_sampler,
            optimizer=optimizer,
            loss_fn=args.loss,
            batch_size=args.batch_size,
            n_neighbors=args.n_neighbors,
            device=device,
            logger=logger
        )
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        
        # =============================================================
        # VALIDATION
        # =============================================================
        
        # Backup memory state before validation
        memory_backup = model.backup_memory()
        
        # Link prediction metrics
        val_metrics = evaluate_link_prediction(
            model=model,
            data=val_data,
            negative_sampler=full_neg_sampler,
            batch_size=args.batch_size,
            n_neighbors=args.n_neighbors,
            device=device
        )
        
        # Restore memory after validation
        model.restore_memory(memory_backup)
        
        val_ap = val_metrics["AP"]
        val_auc = val_metrics["AUC"]
        val_mrr = val_metrics["MRR"]
        
        logger.info(f"Val AP: {val_ap:.4f} | AUC: {val_auc:.4f} | MRR: {val_mrr:.4f}")
        
        writer.add_scalar("Metrics/val_AP", val_ap, epoch)
        writer.add_scalar("Metrics/val_AUC", val_auc, epoch)
        writer.add_scalar("Metrics/val_MRR", val_mrr, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_ap)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("LR", current_lr, epoch)
        
        # Save best model
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ap': val_ap,
                'args': vars(args)
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
            logger.info(f"💾 Saved best model (AP: {val_ap:.4f})")
        
        # Early stopping
        if early_stopper.early_stop_check(val_ap):
            logger.info(f"\n⏹️  Early stopping at epoch {epoch}")
            break
        
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch time: {epoch_time:.1f}s")
    
    # =================================================================
    # FINAL EVALUATION
    # =================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)
    
    # Load best model (weights_only=False for PyTorch 2.6+ compatibility)
    checkpoint = torch.load(save_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"\n✓ Loaded best model from epoch {checkpoint['epoch']}")
    
    # Reset memory and warm up
    model.reset_memory()
    
    logger.info("\n🔄 Building memory state from training data...")
    model.eval()
    with torch.no_grad():
        indices = np.arange(len(train_data.sources))
        for start_idx in tqdm(range(0, len(indices), args.batch_size), desc="Memory warmup"):
            end_idx = min(start_idx + args.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            sources = train_data.sources[batch_indices]
            destinations = train_data.destinations[batch_indices]
            timestamps = train_data.timestamps[batch_indices]
            edge_idxs = train_data.edge_idxs[batch_indices]
            
            _, negatives = train_neg_sampler.sample(len(sources))
            
            _ = model.compute_edge_probabilities(
                sources, destinations, negatives,
                timestamps, edge_idxs, args.n_neighbors
            )
    
    # Process validation data
    logger.info("🔄 Processing validation data...")
    with torch.no_grad():
        indices = np.arange(len(val_data.sources))
        for start_idx in range(0, len(indices), args.batch_size):
            end_idx = min(start_idx + args.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            sources = val_data.sources[batch_indices]
            destinations = val_data.destinations[batch_indices]
            timestamps = val_data.timestamps[batch_indices]
            edge_idxs = val_data.edge_idxs[batch_indices]
            
            _, negatives = full_neg_sampler.sample(len(sources))
            
            _ = model.compute_edge_probabilities(
                sources, destinations, negatives,
                timestamps, edge_idxs, args.n_neighbors
            )
    
    # =============================================================
    # TEST EVALUATION: OVERALL
    # =============================================================
    
    logger.info("\n" + "-" * 50)
    logger.info("TEST SET EVALUATION (Overall)")
    logger.info("-" * 50)
    
    test_lp_metrics = evaluate_link_prediction(
        model=model,
        data=test_data,
        negative_sampler=full_neg_sampler,
        batch_size=args.batch_size,
        n_neighbors=args.n_neighbors,
        device=device
    )
    
    logger.info(f"\n📊 Link Prediction Metrics:")
    logger.info(f"  AP:  {test_lp_metrics['AP']:.4f}")
    logger.info(f"  AUC: {test_lp_metrics['AUC']:.4f}")
    logger.info(f"  MRR: {test_lp_metrics['MRR']:.4f}")
    
    # Ranking metrics
    if args.eval_ranking:
        logger.info("\n📊 Ranking Metrics (100 negatives per positive):")
        
        memory_backup = model.backup_memory()
        
        test_ranking_metrics = evaluate_ranking(
            model=model,
            data=test_data,
            all_items=all_items,
            batch_size=args.batch_size,
            n_neighbors=args.n_neighbors,
            n_negatives=args.n_neg_eval,
            device=device,
            seed=args.seed
        )
        
        model.restore_memory(memory_backup)
        
        logger.info(f"  Recall@10: {test_ranking_metrics.recall_at_10:.4f}")
        logger.info(f"  Recall@20: {test_ranking_metrics.recall_at_20:.4f}")
        logger.info(f"  NDCG@10:   {test_ranking_metrics.ndcg_at_10:.4f}")
        logger.info(f"  NDCG@20:   {test_ranking_metrics.ndcg_at_20:.4f}")
    
    # =============================================================
    # TEST EVALUATION: TRANSDUCTIVE vs INDUCTIVE
    # =============================================================
    
    if len(transductive_test_data) > 0:
        logger.info("\n" + "-" * 50)
        logger.info("TEST SET EVALUATION (Transductive - Old Nodes)")
        logger.info("-" * 50)
        
        memory_backup = model.backup_memory()
        
        trans_lp_metrics = evaluate_link_prediction(
            model=model,
            data=transductive_test_data,
            negative_sampler=full_neg_sampler,
            batch_size=args.batch_size,
            n_neighbors=args.n_neighbors,
            device=device
        )
        
        model.restore_memory(memory_backup)
        
        logger.info(f"\n📊 Transductive Link Prediction ({len(transductive_test_data)} samples):")
        logger.info(f"  AP:  {trans_lp_metrics['AP']:.4f}")
        logger.info(f"  AUC: {trans_lp_metrics['AUC']:.4f}")
        logger.info(f"  MRR: {trans_lp_metrics['MRR']:.4f}")
    
    if len(inductive_test_data) > 0:
        logger.info("\n" + "-" * 50)
        logger.info("TEST SET EVALUATION (Inductive - Cold Start)")
        logger.info("-" * 50)
        
        memory_backup = model.backup_memory()
        
        induct_lp_metrics = evaluate_link_prediction(
            model=model,
            data=inductive_test_data,
            negative_sampler=full_neg_sampler,
            batch_size=args.batch_size,
            n_neighbors=args.n_neighbors,
            device=device
        )
        
        model.restore_memory(memory_backup)
        
        logger.info(f"\n📊 Inductive Link Prediction ({len(inductive_test_data)} samples):")
        logger.info(f"  AP:  {induct_lp_metrics['AP']:.4f}")
        logger.info(f"  AUC: {induct_lp_metrics['AUC']:.4f}")
        logger.info(f"  MRR: {induct_lp_metrics['MRR']:.4f}")
        
        # Log cold-start improvement (key hypothesis)
        if len(transductive_test_data) > 0:
            improvement = induct_lp_metrics['AP'] - trans_lp_metrics['AP']
            logger.info(f"\n🎯 Cold-Start Analysis:")
            logger.info(f"  Inductive vs Transductive AP Δ: {improvement:+.4f}")
    else:
        logger.info("\n⚠️ No inductive test samples (all nodes seen during training)")
    
    # =============================================================
    # SAVE RESULTS
    # =============================================================
    
    writer.add_scalar("Metrics/test_AP", test_lp_metrics['AP'], 0)
    writer.add_scalar("Metrics/test_AUC", test_lp_metrics['AUC'], 0)
    writer.add_scalar("Metrics/test_MRR", test_lp_metrics['MRR'], 0)
    
    if args.eval_ranking:
        writer.add_scalar("Metrics/test_Recall@10", test_ranking_metrics.recall_at_10, 0)
        writer.add_scalar("Metrics/test_Recall@20", test_ranking_metrics.recall_at_20, 0)
        writer.add_scalar("Metrics/test_NDCG@10", test_ranking_metrics.ndcg_at_10, 0)
        writer.add_scalar("Metrics/test_NDCG@20", test_ranking_metrics.ndcg_at_20, 0)
    
    # Build results dict
    results = {
        'best_epoch': best_epoch,
        'best_val_ap': float(best_val_ap),
        'test': {
            'overall': test_lp_metrics,
        },
        'args': vars(args)
    }
    
    if args.eval_ranking:
        results['test']['ranking'] = test_ranking_metrics.to_dict()
    
    if len(transductive_test_data) > 0:
        results['test']['transductive'] = trans_lp_metrics
    
    if len(inductive_test_data) > 0:
        results['test']['inductive'] = induct_lp_metrics
    
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    writer.close()
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {save_dir}")
    logger.info(f"TensorBoard logs: {log_dir}")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    args = get_args()
    train(args)

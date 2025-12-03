"""
MM-TGN Evaluation Metrics Module

Implements ranking metrics crucial for recommender systems research:
- Recall@K: Fraction of relevant items retrieved in top-K
- NDCG@K: Normalized Discounted Cumulative Gain
- MRR: Mean Reciprocal Rank
- Hit Rate@K: Binary indicator if relevant item is in top-K

For efficiency, uses negative sampling evaluation strategy:
- For each positive edge, sample N random negative items
- Rank positive among negatives
- Compute metrics based on this ranking
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RankingMetrics:
    """Container for ranking evaluation results."""
    recall_at_10: float
    recall_at_20: float
    ndcg_at_10: float
    ndcg_at_20: float
    mrr: float
    hit_rate_at_10: float
    hit_rate_at_20: float
    auc: float
    ap: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "Recall@10": self.recall_at_10,
            "Recall@20": self.recall_at_20,
            "NDCG@10": self.ndcg_at_10,
            "NDCG@20": self.ndcg_at_20,
            "MRR": self.mrr,
            "HR@10": self.hit_rate_at_10,
            "HR@20": self.hit_rate_at_20,
            "AUC": self.auc,
            "AP": self.ap
        }
    
    def __str__(self) -> str:
        return (
            f"Recall@10: {self.recall_at_10:.4f} | "
            f"Recall@20: {self.recall_at_20:.4f} | "
            f"NDCG@10: {self.ndcg_at_10:.4f} | "
            f"NDCG@20: {self.ndcg_at_20:.4f} | "
            f"MRR: {self.mrr:.4f}"
        )


def compute_recall_at_k(
    pos_scores: np.ndarray,
    neg_scores_matrix: np.ndarray,
    k: int
) -> float:
    """
    Compute Recall@K using negative sampling strategy.
    
    For each positive sample, we rank it among N negatives.
    Recall@K = fraction of positives ranked in top-K.
    
    Args:
        pos_scores: Positive edge scores [n_positives]
        neg_scores_matrix: Negative scores [n_positives, n_negatives]
        k: Top-K threshold
    
    Returns:
        Recall@K value
    """
    n_positives = len(pos_scores)
    hits = 0
    
    for i in range(n_positives):
        pos_score = pos_scores[i]
        neg_scores = neg_scores_matrix[i]
        
        # Rank = number of items with higher score + 1
        rank = (neg_scores > pos_score).sum() + 1
        
        if rank <= k:
            hits += 1
    
    return hits / n_positives


def compute_ndcg_at_k(
    pos_scores: np.ndarray,
    neg_scores_matrix: np.ndarray,
    k: int
) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).
    
    NDCG = DCG / IDCG
    DCG = sum(rel_i / log2(i + 1)) for i in 1..K
    
    For binary relevance (1 positive item), IDCG = 1 (best case: positive at rank 1)
    
    Args:
        pos_scores: Positive edge scores [n_positives]
        neg_scores_matrix: Negative scores [n_positives, n_negatives]
        k: Top-K threshold
    
    Returns:
        NDCG@K value
    """
    n_positives = len(pos_scores)
    total_ndcg = 0.0
    
    for i in range(n_positives):
        pos_score = pos_scores[i]
        neg_scores = neg_scores_matrix[i]
        
        # Rank of positive item (1-based)
        rank = int((neg_scores > pos_score).sum() + 1)
        
        if rank <= k:
            # DCG for binary relevance at position rank
            dcg = 1.0 / np.log2(rank + 1)
            # IDCG = 1.0 (best case: item at rank 1)
            idcg = 1.0
            total_ndcg += dcg / idcg
    
    return total_ndcg / n_positives


def compute_mrr(
    pos_scores: np.ndarray,
    neg_scores_matrix: np.ndarray
) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    MRR = (1/N) * sum(1/rank_i)
    
    Args:
        pos_scores: Positive edge scores [n_positives]
        neg_scores_matrix: Negative scores [n_positives, n_negatives]
    
    Returns:
        MRR value
    """
    n_positives = len(pos_scores)
    total_rr = 0.0
    
    for i in range(n_positives):
        pos_score = pos_scores[i]
        neg_scores = neg_scores_matrix[i]
        
        # Rank of positive item (1-based)
        rank = int((neg_scores > pos_score).sum() + 1)
        total_rr += 1.0 / rank
    
    return total_rr / n_positives


def compute_hit_rate_at_k(
    pos_scores: np.ndarray,
    neg_scores_matrix: np.ndarray,
    k: int
) -> float:
    """
    Compute Hit Rate@K (same as Recall@K for single positive).
    
    Args:
        pos_scores: Positive edge scores [n_positives]
        neg_scores_matrix: Negative scores [n_positives, n_negatives]
        k: Top-K threshold
    
    Returns:
        Hit Rate@K value
    """
    return compute_recall_at_k(pos_scores, neg_scores_matrix, k)


def compute_auc(
    pos_scores: np.ndarray,
    neg_scores_matrix: np.ndarray
) -> float:
    """
    Compute AUC (Area Under ROC Curve) using negative sampling.
    
    AUC = probability that a positive is ranked higher than a random negative.
    
    Args:
        pos_scores: Positive edge scores [n_positives]
        neg_scores_matrix: Negative scores [n_positives, n_negatives]
    
    Returns:
        AUC value
    """
    n_positives = len(pos_scores)
    n_negatives = neg_scores_matrix.shape[1]
    
    total_correct = 0
    total_pairs = 0
    
    for i in range(n_positives):
        pos_score = pos_scores[i]
        neg_scores = neg_scores_matrix[i]
        
        # Count pairs where positive > negative
        correct = (pos_score > neg_scores).sum()
        ties = (pos_score == neg_scores).sum()
        
        total_correct += correct + 0.5 * ties
        total_pairs += n_negatives
    
    return total_correct / total_pairs


def compute_ap(
    pos_scores: np.ndarray,
    neg_scores_matrix: np.ndarray
) -> float:
    """
    Compute Average Precision using negative sampling.
    
    For single positive item, AP = 1/rank if pos > all negs at higher ranks.
    
    Args:
        pos_scores: Positive edge scores [n_positives]
        neg_scores_matrix: Negative scores [n_positives, n_negatives]
    
    Returns:
        AP value
    """
    # For single positive per query, AP simplifies to reciprocal rank
    return compute_mrr(pos_scores, neg_scores_matrix)


def compute_all_ranking_metrics(
    pos_scores: np.ndarray,
    neg_scores_matrix: np.ndarray
) -> RankingMetrics:
    """
    Compute all ranking metrics at once.
    
    Args:
        pos_scores: Positive edge scores [n_positives]
        neg_scores_matrix: Negative scores [n_positives, n_negatives]
    
    Returns:
        RankingMetrics dataclass with all metrics
    """
    return RankingMetrics(
        recall_at_10=compute_recall_at_k(pos_scores, neg_scores_matrix, k=10),
        recall_at_20=compute_recall_at_k(pos_scores, neg_scores_matrix, k=20),
        ndcg_at_10=compute_ndcg_at_k(pos_scores, neg_scores_matrix, k=10),
        ndcg_at_20=compute_ndcg_at_k(pos_scores, neg_scores_matrix, k=20),
        mrr=compute_mrr(pos_scores, neg_scores_matrix),
        hit_rate_at_10=compute_hit_rate_at_k(pos_scores, neg_scores_matrix, k=10),
        hit_rate_at_20=compute_hit_rate_at_k(pos_scores, neg_scores_matrix, k=20),
        auc=compute_auc(pos_scores, neg_scores_matrix),
        ap=compute_ap(pos_scores, neg_scores_matrix)
    )


class NegativeSamplerForEval:
    """
    Negative sampler specifically for evaluation.
    
    Samples multiple negatives per positive for ranking evaluation.
    Ensures no collision with positive items.
    """
    
    def __init__(
        self,
        all_items: np.ndarray,
        n_negatives: int = 100,
        seed: int = 42
    ):
        """
        Args:
            all_items: Array of all item IDs in the dataset
            n_negatives: Number of negatives to sample per positive
            seed: Random seed
        """
        self.all_items = np.array(all_items)
        self.n_negatives = n_negatives
        self.rng = np.random.RandomState(seed)
    
    def sample_negatives(
        self,
        positive_items: np.ndarray,
        exclude_items: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Sample negatives for each positive item.
        
        Args:
            positive_items: Positive item IDs [n_positives]
            exclude_items: Optional items to exclude (e.g., user's history)
        
        Returns:
            Negative item IDs [n_positives, n_negatives]
        """
        n_positives = len(positive_items)
        negatives = np.zeros((n_positives, self.n_negatives), dtype=np.int64)
        
        for i in range(n_positives):
            # Items to exclude for this sample
            exclude = {positive_items[i]}
            if exclude_items is not None and i < len(exclude_items):
                exclude.add(exclude_items[i])
            
            # Sample until we have enough unique negatives
            candidates = self.all_items[~np.isin(self.all_items, list(exclude))]
            
            if len(candidates) >= self.n_negatives:
                idx = self.rng.choice(len(candidates), self.n_negatives, replace=False)
                negatives[i] = candidates[idx]
            else:
                # If not enough candidates, sample with replacement
                idx = self.rng.choice(len(candidates), self.n_negatives, replace=True)
                negatives[i] = candidates[idx]
        
        return negatives


def split_by_node_type(
    sources: np.ndarray,
    destinations: np.ndarray,
    pos_scores: np.ndarray,
    neg_scores_matrix: np.ndarray,
    transductive_nodes: set,
    inductive_nodes: set
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split evaluation data into transductive and inductive subsets.
    
    Transductive: Both source and destination seen during training.
    Inductive: At least one node is new (cold-start scenario).
    
    Args:
        sources: Source node IDs
        destinations: Destination node IDs
        pos_scores: Positive scores
        neg_scores_matrix: Negative scores matrix
        transductive_nodes: Set of nodes seen during training
        inductive_nodes: Set of new nodes (test only)
    
    Returns:
        Tuple of (transductive_data, inductive_data) dicts
    """
    n_samples = len(sources)
    
    trans_mask = np.array([
        sources[i] in transductive_nodes and destinations[i] in transductive_nodes
        for i in range(n_samples)
    ])
    
    induct_mask = np.array([
        sources[i] in inductive_nodes or destinations[i] in inductive_nodes
        for i in range(n_samples)
    ])
    
    transductive_data = {
        "pos_scores": pos_scores[trans_mask],
        "neg_scores_matrix": neg_scores_matrix[trans_mask]
    }
    
    inductive_data = {
        "pos_scores": pos_scores[induct_mask],
        "neg_scores_matrix": neg_scores_matrix[induct_mask]
    }
    
    return transductive_data, inductive_data


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    
    n_samples = 100
    n_negatives = 50
    
    # Simulate scores where positive generally > negative
    pos_scores = np.random.uniform(0.6, 0.9, n_samples)
    neg_scores_matrix = np.random.uniform(0.1, 0.7, (n_samples, n_negatives))
    
    metrics = compute_all_ranking_metrics(pos_scores, neg_scores_matrix)
    
    print("Test Metrics:")
    print(metrics)
    print("\nAll metrics dict:")
    for k, v in metrics.to_dict().items():
        print(f"  {k}: {v:.4f}")


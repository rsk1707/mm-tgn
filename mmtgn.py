"""
MM-TGN: Multimodal Temporal Graph Network

A dual-channel architecture that combines:
1. Temporal Channel: TGN backbone with memory and temporal attention
2. Content Channel: SOTA multimodal embeddings (text + image)
3. Structural Channel: (Optional) LightGCN/Spectral embeddings from teammate

Fusion via FiLM (Feature-wise Linear Modulation) or Gated mechanism.
"""

import logging
import numpy as np
import torch
from torch import nn
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any

from utils.utils import MergeLayer, NeighborFinder
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding import (
    get_embedding_module,
    HybridNodeFeatures,
    FiLMConditioner,
    UserStateFiLM,  # Novel: User-state modulation for cold-start
    GatedFusion,
    MultimodalProjector
)
from model.time_encoding import TimeEncode


class MMTGN(nn.Module):
    """
    Multimodal Temporal Graph Network.
    
    Extends the original TGN with:
    1. Hybrid node features (learnable users + projected item embeddings)
    2. FiLM conditioning for channel fusion
    3. BPR loss for ranking optimization
    
    Args:
        neighbor_finder: Temporal neighbor lookup
        node_features: Pre-computed node feature matrix (can be dummy if using hybrid)
        edge_features: Edge feature matrix
        device: torch device
        
        # Architecture
        n_layers: Number of graph attention layers
        n_heads: Attention heads
        embedding_dim: Working dimension for all embeddings
        dropout: Dropout rate
        
        # Memory
        use_memory: Whether to use TGN memory module
        memory_dimension: Dimension of memory vectors
        message_dimension: Dimension of messages
        memory_update_at_start: Update memory before or after computing embeddings
        
        # Multimodal
        num_users: Number of user nodes (for hybrid features)
        num_items: Number of item nodes
        item_features: SOTA item embeddings [num_items, feat_dim]
        use_hybrid_features: Use HybridNodeFeatures instead of raw features
        
        # Fusion
        use_film: Apply FiLM conditioning
        structural_dim: Dimension of Channel 2 embeddings (if available)
        use_random_item_features: Use learnable random embeddings instead of SOTA (ablation)
    """
    
    def __init__(
        self,
        neighbor_finder: NeighborFinder,
        node_features: np.ndarray,
        edge_features: np.ndarray,
        device: str,
        # Architecture
        n_layers: int = 2,
        n_heads: int = 2,
        embedding_dim: int = 172,
        dropout: float = 0.1,
        # Memory
        use_memory: bool = True,
        memory_dimension: int = 172,
        message_dimension: int = 100,
        memory_update_at_start: bool = True,
        memory_updater_type: str = "gru",
        message_function: str = "mlp",
        aggregator_type: str = "last",
        # Embedding module
        embedding_module_type: str = "graph_attention",
        # Multimodal
        num_users: Optional[int] = None,
        num_items: Optional[int] = None,
        item_features: Optional[np.ndarray] = None,
        use_hybrid_features: bool = True,
        use_random_item_features: bool = False,  # Ablation: vanilla baseline
        # Fusion
        use_film: bool = True,
        structural_dim: Optional[int] = None,
        # Time normalization
        mean_time_shift_src: float = 0,
        std_time_shift_src: float = 1,
        mean_time_shift_dst: float = 0,
        std_time_shift_dst: float = 1,
        # Other
        n_neighbors: int = 20,
        use_destination_embedding_in_message: bool = False,
        use_source_embedding_in_message: bool = False,
        dyrep: bool = False
    ):
        super(MMTGN, self).__init__()
        
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.n_layers = n_layers
        self.n_neighbors = n_neighbors
        self.embedding_dim = embedding_dim
        self.use_memory = use_memory
        self.use_hybrid_features = use_hybrid_features
        self.use_film = use_film
        self.dyrep = dyrep
        
        # Time normalization
        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst
        
        # Message passing settings
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        
        # =================================================================
        # NODE FEATURES SETUP (1-BASED INDEXING)
        # Index 0 is reserved for padding in TGN's neighbor finder
        # =================================================================
        
        if use_hybrid_features and (item_features is not None or use_random_item_features):
            assert num_users is not None and num_items is not None, \
                "num_users and num_items required for hybrid features"
            
            self.logger.info(f"Using HybridNodeFeatures: {num_users} users, {num_items} items")
            self.logger.info(f"  Index 0 = Padding, Users: [1, {num_users}], Items: [{num_users+1}, {num_users+num_items}]")
            
            if use_random_item_features:
                self.logger.info(f"  Mode: RANDOM (learnable item embeddings, ablation baseline)")
            else:
                self.logger.info(f"  Mode: SOTA (projected multimodal features)")
            
            self.hybrid_features = HybridNodeFeatures(
                num_users=num_users,
                num_items=num_items,
                item_features=item_features,  # None if use_random_item_features=True
                embedding_dim=embedding_dim,
                dropout=dropout,
                freeze_items=True,
                use_random_items=use_random_item_features  # Ablation flag
            )
            
            # Total nodes includes padding at index 0
            # 1-based: 0 (pad) + num_users + num_items
            n_nodes = num_users + num_items + 1
            self.node_raw_features = torch.zeros(n_nodes, embedding_dim, device=device)
            self.n_node_features = embedding_dim
            self.n_nodes = n_nodes
            
            # Store counts for reference
            self.num_users = num_users
            self.num_items = num_items
        else:
            # Standard mode: use provided features directly
            self.hybrid_features = None
            self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
            self.n_node_features = self.node_raw_features.shape[1]
            self.n_nodes = self.node_raw_features.shape[0]
            self.num_users = None
            self.num_items = None
        
        # Edge features
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
        self.n_edge_features = self.edge_raw_features.shape[1]
        
        # =================================================================
        # TIME ENCODER
        # =================================================================
        
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        
        # =================================================================
        # MEMORY MODULE
        # =================================================================
        
        self.memory = None
        if use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            
            raw_message_dimension = (
                2 * memory_dimension + 
                self.n_edge_features + 
                self.time_encoder.dimension
            )
            
            message_dim = message_dimension if message_function != "identity" else raw_message_dimension
            
            self.memory = Memory(
                n_nodes=self.n_nodes,
                memory_dimension=memory_dimension,
                input_dimension=message_dim,
                message_dimension=message_dim,
                device=device
            )
            
            self.message_aggregator = get_message_aggregator(
                aggregator_type=aggregator_type,
                device=device
            )
            
            self.message_function = get_message_function(
                module_type=message_function,
                raw_message_dimension=raw_message_dimension,
                message_dimension=message_dim
            )
            
            self.memory_updater = get_memory_updater(
                module_type=memory_updater_type,
                memory=self.memory,
                message_dimension=message_dim,
                memory_dimension=memory_dimension,
                device=device
            )
        
        # =================================================================
        # GRAPH EMBEDDING MODULE
        # =================================================================
        
        self.neighbor_finder = neighbor_finder
        self.embedding_module_type = embedding_module_type
        
        self.embedding_module = get_embedding_module(
            module_type=embedding_module_type,
            node_features=self.node_raw_features,
            edge_features=self.edge_raw_features,
            memory=self.memory,
            neighbor_finder=neighbor_finder,
            time_encoder=self.time_encoder,
            n_layers=n_layers,
            n_node_features=self.n_node_features,
            n_edge_features=self.n_edge_features,
            n_time_features=self.n_node_features,
            embedding_dimension=embedding_dim,
            device=device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory,
            n_neighbors=n_neighbors
        )
        
        # =================================================================
        # FILM FUSION (Optional - BYPASS MODE if structural_dim is None)
        # =================================================================
        # When structural embeddings (Channel 2) are not available,
        # FiLM acts as identity (pass-through) - no crash, just skip.
        
        self.film_conditioner = None
        self.structural_dim = structural_dim
        self.use_film = use_film
        
        if use_film and structural_dim is not None:
            self.logger.info(f"FiLM Conditioner enabled (structural_dim={structural_dim})")
            self.film_conditioner = FiLMConditioner(
                input_dim=embedding_dim,
                cond_dim=structural_dim,
                hidden_dim=embedding_dim
            )
        elif use_film:
            self.logger.info("FiLM enabled but structural_dim=None - BYPASS MODE (identity)")
        else:
            self.logger.info("FiLM disabled")
        
        # =================================================================
        # OUTPUT LAYERS
        # =================================================================
        
        # Link prediction head
        self.affinity_score = MergeLayer(
            self.n_node_features,
            self.n_node_features,
            self.n_node_features,
            1
        )
    
    def update_node_features(self):
        """Update node feature tensor from hybrid features (call each forward)."""
        if self.hybrid_features is not None:
            self.hybrid_features.invalidate_cache()
            all_features = self.hybrid_features.get_all_features()
            self.node_raw_features.data.copy_(all_features)
            self.embedding_module.node_features = self.node_raw_features
    
    def compute_temporal_embeddings(
        self,
        source_nodes: np.ndarray,
        destination_nodes: np.ndarray,
        negative_nodes: np.ndarray,
        edge_times: np.ndarray,
        edge_idxs: np.ndarray,
        n_neighbors: int = 20,
        structural_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute temporal embeddings for sources, destinations, and negatives.
        
        Returns:
            Tuple of (source_emb, dest_emb, neg_emb)
        """
        # Update hybrid features if using
        self.update_node_features()
        
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times, edge_times])
        
        memory = None
        time_diffs = None
        
        if self.use_memory:
            if self.memory_update_at_start:
                memory, last_update = self.get_updated_memory(
                    list(range(self.n_nodes)),
                    self.memory.messages
                )
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update
            
            # Compute time differences
            source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[source_nodes].long()
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
            
            destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[destination_nodes].long()
            destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            
            negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[negative_nodes].long()
            negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            
            time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs], dim=0)
        
        # Compute embeddings via TGN
        node_embedding = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs
        )
        
        # =================================================================
        # FILM FUSION with BYPASS MODE
        # =================================================================
        # If FiLM conditioner exists AND structural embeddings provided -> apply FiLM
        # Otherwise -> BYPASS (identity, no modification)
        # This ensures the model doesn't crash when Channel 2 is unavailable
        
        if self.film_conditioner is not None and structural_embeddings is not None:
            # Full FiLM conditioning: modulate embeddings with structural context
            node_embedding = self.film_conditioner(node_embedding, structural_embeddings)
        # else: BYPASS MODE - embeddings pass through unchanged
        
        # Split embeddings
        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples:2*n_samples]
        negative_node_embedding = node_embedding[2*n_samples:]
        
        # Update memory
        if self.use_memory:
            if self.memory_update_at_start:
                self.update_memory(positives, self.memory.messages)
                
                assert torch.allclose(
                    memory[positives], 
                    self.memory.get_memory(positives), 
                    atol=1e-5
                ), "Memory update mismatch"
                
                self.memory.clear_messages(positives)
            
            # Compute and store new messages
            unique_sources, source_id_to_messages = self.get_raw_messages(
                source_nodes, source_node_embedding,
                destination_nodes, destination_node_embedding,
                edge_times, edge_idxs
            )
            
            unique_destinations, destination_id_to_messages = self.get_raw_messages(
                destination_nodes, destination_node_embedding,
                source_nodes, source_node_embedding,
                edge_times, edge_idxs
            )
            
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, destination_id_to_messages)
            
            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]
                negative_node_embedding = memory[negative_nodes]
        
        return source_node_embedding, destination_node_embedding, negative_node_embedding
    
    def compute_edge_probabilities(
        self,
        source_nodes: np.ndarray,
        destination_nodes: np.ndarray,
        negative_nodes: np.ndarray,
        edge_times: np.ndarray,
        edge_idxs: np.ndarray,
        n_neighbors: int = 20,
        structural_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute link probabilities for positive and negative edges.
        
        Returns:
            (pos_probs, neg_probs) both of shape [batch_size]
        """
        n_samples = len(source_nodes)
        
        source_emb, dest_emb, neg_emb = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes,
            edge_times, edge_idxs, n_neighbors,
            structural_embeddings
        )
        
        score = self.affinity_score(
            torch.cat([source_emb, source_emb], dim=0),
            torch.cat([dest_emb, neg_emb], dim=0)
        ).squeeze(dim=0)
        
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]
        
        return pos_score.sigmoid(), neg_score.sigmoid()
    
    def update_memory(self, nodes, messages):
        """Aggregate and update memory for given nodes."""
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(nodes, messages)
        
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
        
        self.memory_updater.update_memory(
            unique_nodes, unique_messages, timestamps=unique_timestamps
        )
    
    def get_updated_memory(self, nodes, messages):
        """Get updated memory without modifying state."""
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(nodes, messages)
        
        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)
        
        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(
            unique_nodes, unique_messages, timestamps=unique_timestamps
        )
        
        return updated_memory, updated_last_update
    
    def get_raw_messages(
        self,
        source_nodes, source_node_embedding,
        destination_nodes, destination_node_embedding,
        edge_times, edge_idxs
    ):
        """Compute raw messages for memory update."""
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxs]
        
        source_memory = self.memory.get_memory(source_nodes) if not \
            self.use_source_embedding_in_message else source_node_embedding
        destination_memory = self.memory.get_memory(destination_nodes) if \
            not self.use_destination_embedding_in_message else destination_node_embedding
        
        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(
            source_time_delta.unsqueeze(dim=1)
        ).view(len(source_nodes), -1)
        
        source_message = torch.cat([
            source_memory, destination_memory,
            edge_features, source_time_delta_encoding
        ], dim=1)
        
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)
        
        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))
        
        return unique_sources, messages
    
    def set_neighbor_finder(self, neighbor_finder: NeighborFinder):
        """Update neighbor finder (for train/val/test transitions)."""
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
    
    def reset_memory(self):
        """Reset memory state (call at start of each epoch)."""
        if self.memory is not None:
            self.memory.__init_memory__()
    
    def backup_memory(self):
        """Backup memory state for validation."""
        if self.memory is not None:
            return self.memory.backup_memory()
        return None
    
    def restore_memory(self, backup):
        """Restore memory from backup."""
        if self.memory is not None and backup is not None:
            self.memory.restore_memory(backup)
    
    def detach_memory(self):
        """Detach memory gradients (for TBPTT)."""
        if self.memory is not None:
            self.memory.detach_memory()


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss.
    
    Maximizes the margin between positive and negative samples.
    Better suited for ranking tasks than BCE.
    
    Args:
        pos_scores: Positive edge scores [batch]
        neg_scores: Negative edge scores [batch]
    
    Returns:
        Scalar loss
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()


def bce_loss(
    pos_scores: torch.Tensor, 
    neg_scores: torch.Tensor,
    pos_weight: float = 1.0
) -> torch.Tensor:
    """
    Binary Cross-Entropy loss for link prediction.
    
    Args:
        pos_scores: Positive edge probabilities [batch]
        neg_scores: Negative edge probabilities [batch]
        pos_weight: Weight for positive samples
    
    Returns:
        Scalar loss
    """
    pos_loss = -pos_weight * torch.log(pos_scores + 1e-10).mean()
    neg_loss = -torch.log(1 - neg_scores + 1e-10).mean()
    return pos_loss + neg_loss


def contrastive_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0
) -> torch.Tensor:
    """
    Margin-based contrastive loss.
    
    Args:
        pos_scores: Positive edge scores [batch]
        neg_scores: Negative edge scores [batch]
        margin: Margin for separation
    
    Returns:
        Scalar loss
    """
    return torch.clamp(margin - pos_scores + neg_scores, min=0).mean()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_mmtgn(
    dataset,  # TemporalDataset
    device: str = "cuda",
    embedding_dim: int = 172,
    n_layers: int = 2,
    n_heads: int = 2,
    n_neighbors: int = 20,
    memory_dim: int = 172,
    message_dim: int = 100,
    dropout: float = 0.1,
    use_memory: bool = True,
    use_hybrid_features: bool = True,
    embedding_module_type: str = "graph_attention",
    structural_dim: Optional[int] = None,  # For Channel 2 integration
    use_random_item_features: bool = False  # Ablation: vanilla baseline
) -> MMTGN:
    """
    Factory function to create MMTGN from a TemporalDataset.
    
    Args:
        dataset: TemporalDataset instance
        device: torch device
        structural_dim: Dimension of Channel 2 embeddings (None = bypass FiLM)
        use_random_item_features: If True, use learnable random embeddings instead
                                  of SOTA features (ablation study baseline)
        ... (other hyperparameters)
    
    Returns:
        Configured MMTGN model
    
    Note: Uses 1-based indexing (Index 0 = padding for TGN neighbor masking)
    """
    from utils.utils import get_neighbor_finder
    
    # Build neighbor finder from full data
    full_data = dataset.get_full_data()
    neighbor_finder = get_neighbor_finder(full_data, uniform=False)
    
    # Compute time statistics
    time_stats = dataset.compute_time_statistics()
    
    # Get item features for hybrid mode
    # For ablation study: if use_random_item_features=True, we pass None
    # and HybridNodeFeatures will use learnable embeddings instead
    if use_random_item_features:
        item_features = None  # Signal to use random learnable embeddings
        logging.info("  - Item Features: RANDOM (learnable embeddings, ablation mode)")
    else:
        item_features = dataset.get_item_features_only() if use_hybrid_features else None
    
    # Log indexing info
    logging.info(f"Creating MMTGN with 1-based indexing:")
    logging.info(f"  - Padding index: 0")
    logging.info(f"  - Users: [1, {dataset.num_users}]")
    logging.info(f"  - Items: [{dataset.num_users + 1}, {dataset.num_users + dataset.num_items}]")
    logging.info(f"  - Total nodes: {dataset.n_nodes}")
    if structural_dim is None:
        logging.info(f"  - FiLM: BYPASS MODE (Channel 2 not provided)")
    else:
        logging.info(f"  - FiLM: ENABLED (structural_dim={structural_dim})")
    
    model = MMTGN(
        neighbor_finder=neighbor_finder,
        node_features=dataset.node_features,
        edge_features=dataset.edge_features,
        device=device,
        n_layers=n_layers,
        n_heads=n_heads,
        embedding_dim=embedding_dim,
        dropout=dropout,
        use_memory=use_memory,
        memory_dimension=memory_dim,
        message_dimension=message_dim,
        embedding_module_type=embedding_module_type,
        num_users=dataset.num_users if use_hybrid_features else None,
        num_items=dataset.num_items if use_hybrid_features else None,
        item_features=item_features,
        use_hybrid_features=use_hybrid_features,
        use_random_item_features=use_random_item_features,  # Ablation flag
        use_film=True,  # Always enable FiLM logic (will bypass if structural_dim=None)
        structural_dim=structural_dim,  # None = bypass mode
        mean_time_shift_src=time_stats[0],
        std_time_shift_src=time_stats[1],
        mean_time_shift_dst=time_stats[2],
        std_time_shift_dst=time_stats[3],
        n_neighbors=n_neighbors
    )
    
    return model.to(device)


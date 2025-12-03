"""
MM-TGN Embedding Module
Extended from original TGN with Multimodal Node Features support.

Key Components:
1. MultimodalNodeAdapter: Projects SOTA embeddings (text+image) to TGN dimension
2. HybridNodeFeatures: Combines learnable user embeddings with fixed item features
3. FiLMConditioner: Feature-wise Linear Modulation for fusion
"""

import torch
from torch import nn
import numpy as np
import math
from typing import Optional, Tuple

from model.temporal_attention import TemporalAttentionLayer


# =============================================================================
# MULTIMODAL NODE ADAPTER (NEW)
# =============================================================================

class MultimodalProjector(nn.Module):
    """
    Projects high-dimensional SOTA features (e.g., 2688-dim Qwen2+SigLIP) 
    to TGN's working dimension with nonlinear transformation.
    
    Architecture: Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(output_dim * 2, 512)  # Ensure sufficient capacity
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize with small weights for stable training
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class HybridNodeFeatures(nn.Module):
    """
    Unified node feature provider for MM-TGN.
    
    IMPORTANT: Uses 1-based indexing to match TGN conventions:
    - Index 0 = Padding (returns zero vector)
    - Users: indices [1, num_users] 
    - Items: indices [num_users + 1, num_users + num_items]
    
    Args:
        num_users: Number of user nodes
        num_items: Number of item nodes  
        item_features: Pre-computed SOTA features for items [num_items, feat_dim]
                       If None and use_random_items=True, uses learnable embeddings
        embedding_dim: Target dimension for TGN
        dropout: Dropout rate for projector
        freeze_items: Whether to freeze item features (recommended: True)
        use_random_items: If True, use learnable random embeddings for items
                          instead of SOTA features (ablation study baseline)
    """
    def __init__(
        self,
        num_users: int,
        num_items: int,
        item_features: Optional[np.ndarray],
        embedding_dim: int,
        dropout: float = 0.1,
        freeze_items: bool = True,
        use_random_items: bool = False  # Ablation: vanilla baseline
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_random_items = use_random_items
        # Total includes padding at index 0
        self.total_nodes = num_users + num_items + 1
        
        # User embeddings (learnable) - index 0 in embedding = user 1 in TGN
        # We allocate num_users embeddings for users at TGN indices [1, num_users]
        # Note: We do NOT use padding_idx here because we handle padding separately
        # via the is_padding mask in forward(). The user embedding table only has
        # num_users entries (indices 0 to num_users-1), corresponding to TGN IDs 1 to num_users.
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )
        nn.init.xavier_uniform_(self.user_embedding.weight)
        
        # Padding embedding (index 0) - always zero
        self.register_buffer(
            'padding_embedding',
            torch.zeros(1, embedding_dim)
        )
        
        # =================================================================
        # ITEM FEATURES: SOTA vs RANDOM (Ablation Study)
        # =================================================================
        
        if use_random_items:
            # ABLATION MODE: Learnable random embeddings (vanilla baseline)
            # This establishes the lower bound - no semantic content information
            self.item_embedding = nn.Embedding(
                num_embeddings=num_items,
                embedding_dim=embedding_dim
            )
            nn.init.xavier_uniform_(self.item_embedding.weight)
            
            # No projector or raw features needed
            self.item_projector = None
            self.item_features_raw = None
        else:
            # SOTA MODE: Project pre-computed multimodal features
            assert item_features is not None, \
                "item_features required when use_random_items=False"
            
            item_feat_dim = item_features.shape[1]
            self.register_buffer(
                'item_features_raw', 
                torch.from_numpy(item_features.astype(np.float32))
            )
            
            # Projector: SOTA dim -> TGN dim
            self.item_projector = MultimodalProjector(
                input_dim=item_feat_dim,
                output_dim=embedding_dim,
                dropout=dropout
            )
            
            if freeze_items:
                # Freeze the raw features (projector remains trainable)
                self.item_features_raw.requires_grad = False
            
            # No learnable item embedding in SOTA mode
            self.item_embedding = None
        
        # Cache for projected item features (computed once per forward)
        self._projected_items_cache = None
        self._cache_valid = False
    
    def invalidate_cache(self):
        """Call this at the start of each training step to recompute projections."""
        self._cache_valid = False
    
    def get_item_features(self) -> torch.Tensor:
        """
        Get item features (either projected SOTA or random learnable).
        Uses cache if valid (SOTA mode only).
        """
        if self.use_random_items:
            # Random mode: return learnable embeddings directly
            return self.item_embedding.weight
        else:
            # SOTA mode: return projected features (cached)
            if not self._cache_valid or self._projected_items_cache is None:
                self._projected_items_cache = self.item_projector(self.item_features_raw)
                self._cache_valid = True
            return self._projected_items_cache
    
    # Alias for backward compatibility
    def get_projected_items(self) -> torch.Tensor:
        """Alias for get_item_features() (backward compatibility)."""
        return self.get_item_features()
    
    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Get features for given node IDs.
        
        1-BASED INDEXING (TGN Convention):
        - Index 0 -> Padding (zero vector)
        - [1, num_users] -> Users
        - [num_users + 1, num_users + num_items] -> Items
        """
        device = node_ids.device
        batch_size = node_ids.shape[0]
        
        # Initialize output with zeros (handles padding automatically)
        features = torch.zeros(batch_size, self.embedding_dim, device=device)
        
        # Masks for different node types (1-based indexing)
        is_padding = node_ids == 0
        is_user = (node_ids >= 1) & (node_ids <= self.num_users)
        is_item = node_ids > self.num_users
        
        # Padding: already zeros, nothing to do
        
        # Get user features (TGN ID 1 -> embedding index 0)
        if is_user.any():
            user_tgn_ids = node_ids[is_user]
            user_local_ids = user_tgn_ids - 1  # Convert to 0-based embedding index
            features[is_user] = self.user_embedding(user_local_ids)
        
        # Get item features (TGN ID num_users+1 -> item index 0)
        if is_item.any():
            item_tgn_ids = node_ids[is_item]
            item_local_ids = item_tgn_ids - self.num_users - 1  # Convert to 0-based item index
            item_feats = self.get_item_features()
            features[is_item] = item_feats[item_local_ids]
        
        return features
    
    def get_all_features(self) -> torch.Tensor:
        """
        Get feature matrix for all nodes (for TGN initialization).
        
        Returns tensor of shape [total_nodes, embedding_dim]:
        - Row 0: Padding (zeros)
        - Rows 1 to num_users: User embeddings
        - Rows num_users+1 to num_users+num_items: Item embeddings
        """
        padding = self.padding_embedding  # [1, dim]
        user_feats = self.user_embedding.weight  # [num_users, dim]
        item_feats = self.get_item_features()  # [num_items, dim]
        return torch.cat([padding, user_feats, item_feats], dim=0)


class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Conditions one signal on another via learned scale (γ) and shift (β).
    
    Formula: output = γ(conditioning) ⊙ input + β(conditioning)
    
    Args:
        input_dim: Dimension of input features to be modulated
        cond_dim: Dimension of conditioning features
        hidden_dim: Hidden dimension for γ/β networks
    """
    def __init__(self, input_dim: int, cond_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        
        hidden_dim = hidden_dim or input_dim
        
        # γ (scale) network
        self.gamma_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # β (shift) network  
        self.beta_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize γ to 1 and β to 0 for identity initialization
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning.
        
        Args:
            x: Input features to modulate [batch, input_dim]
            conditioning: Conditioning signal [batch, cond_dim]
        
        Returns:
            Modulated features [batch, input_dim]
        """
        gamma = self.gamma_net(conditioning)
        beta = self.beta_net(conditioning)
        return gamma * x + beta


class UserStateFiLM(nn.Module):
    """
    User-State FiLM: Novel fusion strategy for MM-TGN.
    
    Uses User's TGN Memory State to modulate Item's Multimodal Features.
    This enables dynamic, user-specific interpretation of item content.
    
    Mathematical Formulation:
        h_adapted = γ(h_user) ⊙ h_item + β(h_user)
    
    Where:
        h_user: User's temporal memory state [batch, user_dim]
        h_item: Item's multimodal features [batch, item_dim]
        γ, β: Learned affine transformations
    
    Theoretical Motivation:
        The same item (e.g., "Action Movie") may appeal differently to users
        with different temporal states. A user who just watched 3 horror movies
        vs one who watched 3 comedies will perceive the action movie differently.
        This layer learns to adapt item representations to user context.
    
    Args:
        user_dim: Dimension of user memory state (e.g., 172 for TGN)
        item_dim: Dimension of item features (same as user_dim after projection)
        hidden_dim: Hidden layer dimension for γ/β networks
    """
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or user_dim
        
        # Ensure dimensions match for element-wise modulation
        self.user_dim = user_dim
        self.item_dim = item_dim
        
        # If dimensions differ, project user state to item dimension
        self.need_projection = (user_dim != item_dim)
        if self.need_projection:
            self.user_proj = nn.Linear(user_dim, item_dim)
        
        # γ (scale) network: how much to amplify each item feature dimension
        self.gamma_net = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, item_dim)
        )
        
        # β (shift) network: how much to add to each item feature dimension
        self.beta_net = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, item_dim)
        )
        
        # Initialize for identity at start (γ=1, β=0)
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)
    
    def forward(
        self,
        item_features: torch.Tensor,
        user_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Modulate item features based on user's temporal memory state.
        
        Args:
            item_features: Item embeddings [batch, item_dim]
            user_memory: User memory states [batch, user_dim]
        
        Returns:
            User-adapted item features [batch, item_dim]
        """
        # Project user state if dimensions differ
        if self.need_projection:
            user_state = self.user_proj(user_memory)
        else:
            user_state = user_memory
        
        # Compute modulation parameters
        gamma = self.gamma_net(user_state)  # [batch, item_dim]
        beta = self.beta_net(user_state)    # [batch, item_dim]
        
        # Apply FiLM: element-wise modulation
        return gamma * item_features + beta


class GatedFusion(nn.Module):
    """
    Gated fusion of multiple feature streams.
    
    Learns to weight contributions from temporal, content, and structural channels.
    """
    def __init__(self, dim: int, num_channels: int = 3):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(dim * num_channels, dim),
            nn.Sigmoid()
        )
        
        self.transform = nn.Sequential(
            nn.Linear(dim * num_channels, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Fuse multiple feature tensors.
        
        Args:
            *features: Variable number of [batch, dim] tensors
        
        Returns:
            Fused features [batch, dim]
        """
        combined = torch.cat(features, dim=-1)
        gate = self.gate(combined)
        transformed = self.transform(combined)
        
        # Gate controls how much of the transformed vs. first input to use
        return gate * transformed + (1 - gate) * features[0]


# =============================================================================
# ORIGINAL TGN EMBEDDING MODULES (Preserved)
# =============================================================================

class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    self.memory = memory
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return NotImplemented


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)

    self.use_memory = use_memory
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    source_node_features = self.node_features[source_nodes_torch, :]


    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features

    if n_layers == 0:
      return source_node_features
    else:

      source_node_conv_embeddings = self.compute_embedding(memory,
                                                           source_nodes,
                                                           timestamps,
                                                           n_layers=n_layers - 1,
                                                           n_neighbors=n_neighbors)

      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == 0

      source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)

      return source_embedding

  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                         n_edge_features, embedding_dimension)
                                         for _ in range(n_layers)])
    self.linear_2 = torch.nn.ModuleList(
      [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                       embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                   dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

    source_features = torch.cat([source_node_features,
                                 source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    return source_embedding


class NGCFGraphEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(NGCFGraphEmbedding, self).__init__(node_features=node_features,
                                                 edge_features=edge_features,
                                                 memory=memory,
                                                 neighbor_finder=neighbor_finder,
                                                 time_encoder=time_encoder, n_layers=n_layers,
                                                 n_node_features=n_node_features,
                                                 n_edge_features=n_edge_features,
                                                 n_time_features=n_time_features,
                                                 embedding_dimension=embedding_dimension,
                                                 device=device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory)
        # NGCF uses two learnable weight matrices for each layer.
        self.W1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension, embedding_dimension)
                                       for _ in range(n_layers)])
        self.W2 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension, embedding_dimension)
                                       for _ in range(n_layers)])
        
        # Activation function
        self.leaky_relu = torch.nn.LeakyReLU()

        

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        # First embedding transformation
        sum_embeddings = torch.sum(neighbor_embeddings, dim=1)
        sum_transformed = self.W1[n_layer - 1](sum_embeddings)
        
        # Second embedding transformation
        source_transformed = self.W2[n_layer - 1](source_node_features)
        
        # Bi-interaction pooling
        bi_interaction = sum_transformed * source_transformed
        
        # Apply activation function
        agg_embeddings = self.leaky_relu(bi_interaction)
        
        return agg_embeddings
    
    
def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)
  elif module_type == "graph_NGCF":
    return NGCFGraphEmbedding(node_features=node_features,
                                  edge_features=edge_features,
                                  memory=memory,
                                  neighbor_finder=neighbor_finder,
                                  time_encoder=time_encoder,
                                  n_layers=n_layers,
                                  n_node_features=n_node_features,
                                  n_edge_features=n_edge_features,
                                  n_time_features=n_time_features,
                                  embedding_dimension=embedding_dimension,
                                  device=device,
                                  n_heads=n_heads, dropout=dropout, use_memory=use_memory)
  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))

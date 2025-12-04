# MM-TGN: Multimodal Temporal Graph Network
## Complete Architecture & API Documentation

**Version**: 2.1 (Critical Bug Fixes)  
**Last Updated**: December 3, 2025  
**Research Goal**: Solve Cold Start and Concept Drift in Recommendation Systems  
**Environment**: Great Lakes HPC (University of Michigan)

---

## Table of Contents
1. [Research Hypothesis](#-research-hypothesis)
2. [System Architecture](#️-system-architecture-diagram)
3. [Indexing Standard](#-indexing-standard-critical)
4. [Complete CLI Reference](#-complete-cli-reference)
5. [Ready-to-Run Ablation Commands](#-ready-to-run-ablation-commands)
6. [Data Format Specifications](#-data-format-specifications)
7. [Core API Documentation](#-core-api-documentation)
8. [TensorBoard Setup](#-tensorboard-setup-great-lakes-hpc)
9. [File Dictionary](#-file-dictionary)
10. [Known Issues & Solutions](#-known-issues--solutions)
11. [Context Restoration](#-context-restoration-checklist)

---

## 🎯 Research Hypothesis

> **Multimodal features modulated by temporal context solve the Cold Start problem better than ID-only baselines.**

**Dual-Channel Architecture:**
- **Channel 1**: Temporal Sequential Dynamics (TGN) + SOTA Multimodal Features ✅
- **Channel 2**: Spectral/Structural Signal (LightGCN) - *In Development*

---

## 🏗️ System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ratings.csv ──┬──► tgn_formatter.py ──► ml_*.csv (Temporal Edges)    │
│                 │                              │                        │
│   enriched.csv ─┤                              ▼                        │
│        +        │                       ml_*.npy (Node Features)        │
│   posters/ ─────┘                              │                        │
│                 │                              │                        │
│                 ▼                              │                        │
│   generate_embeddings.py                       │                        │
│      │                                         │                        │
│      ├─► Qwen2-1.5B (Text: 1536-dim)          │                        │
│      └─► SigLIP-SO400M (Image: 1152-dim)      │                        │
│              │                                 │                        │
│              ▼                                 │                        │
│      Concatenate: 2688-dim SOTA Features ─────┘                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           MODEL ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     HybridNodeFeatures                           │   │
│  │  ┌─────────────┐    ┌────────────────────────────────────────┐  │   │
│  │  │ nn.Embedding │    │     MultimodalProjector                │  │   │
│  │  │   (Users)    │    │  SOTA (2688-dim) ──► TGN (172-dim)    │  │   │
│  │  │  10,200 × 172│    │  OR Random nn.Embedding (ablation)    │  │   │
│  │  └─────────────┘    └────────────────────────────────────────┘  │   │
│  │         │                        │                               │   │
│  │         └───────────┬────────────┘                               │   │
│  │                     ▼                                            │   │
│  │           [padding=0 | users | items]                            │   │
│  │              Full Node Feature Matrix                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         TGN Backbone                             │   │
│  │  ┌──────────────┐   ┌────────────────┐   ┌────────────────────┐ │   │
│  │  │   Memory     │   │   Temporal     │   │   Graph Attention  │ │   │
│  │  │   Module     │◄──│   Neighbor     │◄──│   Embedding        │ │   │
│  │  │  (GRU/LSTM)  │   │   Finder       │   │   (2 layers)       │ │   │
│  │  └──────────────┘   └────────────────┘   └────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    FiLM Fusion Layer                              │  │
│  │   Temporal Embeddings ───► FiLMConditioner ◄─── Structural       │  │
│  │         (172-dim)           (γ⊙x + β)           (Channel 2)      │  │
│  │              * BYPASS MODE if Channel 2 unavailable *            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      Prediction Head                              │  │
│  │            MergeLayer(user_emb, item_emb) ──► Score              │  │
│  │            Loss: BCE (default) or BPR (ranking)                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 Indexing Standard (CRITICAL)

### 1-Based Indexing with Padding at 0

```
Index 0:                    PADDING (zero vector, used for neighbor masking)
Indices [1, num_users]:     USER nodes
Indices [num_users+1, ...]: ITEM nodes
```

**Why 1-based?**
- TGN's `NeighborFinder` uses 0 to indicate "no neighbor" (padding)
- All node IDs in CSV files are 1-indexed
- Feature matrix row 0 is always zeros

**ML-Modern Example:**
```
num_users  = 10,200
num_items  = 21,969
total_nodes = 32,170 (including padding)

User "1"     → Index 1
User "10200" → Index 10,200
Item (first) → Index 10,201
Item (last)  → Index 32,169
```

---

## 📋 Complete CLI Reference

### Data Arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--data-dir` | `str` | - | ✅ Yes | Directory containing `ml_*.csv`, `ml_*.npy`, `node_map.json` |
| `--dataset` | `str` | - | ✅ Yes | Dataset name (e.g., `ml-modern`, `amazon-sports`) |

### Model Architecture Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--embedding-dim` | `int` | `172` | Embedding dimension for all components |
| `--n-layers` | `int` | `2` | Number of graph attention layers |
| `--n-heads` | `int` | `2` | Number of attention heads per layer |
| `--n-neighbors` | `int` | `15` | Number of temporal neighbors to sample |
| `--memory-dim` | `int` | `172` | TGN memory vector dimension |
| `--message-dim` | `int` | `100` | Message passing dimension |
| `--dropout` | `float` | `0.1` | Dropout rate |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch-size` | `int` | `200` | Training batch size |
| `--epochs` | `int` | `50` | Number of training epochs |
| `--lr` | `float` | `1e-4` | Learning rate |
| `--weight-decay` | `float` | `1e-5` | L2 regularization weight |
| `--patience` | `int` | `5` | Early stopping patience (epochs) |
| `--loss` | `str` | `bce` | Loss function: `bce` (stable) or `bpr` (ranking) |

### Feature & Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use-memory` | `flag` | `True` | Enable TGN memory module |
| `--no-memory` | `flag` | - | Disable TGN memory module |
| `--use-hybrid` | `flag` | `True` | Use HybridNodeFeatures (learnable users + projected items) |
| `--no-hybrid` | `flag` | - | Use raw features directly |
| `--embedding-module` | `str` | `graph_attention` | Options: `graph_attention`, `graph_sum`, `identity`, `time` |

### Ablation Study Arguments

| Argument | Type | Default | Choices | Description |
|----------|------|---------|---------|-------------|
| `--node-feature-type` | `str` | `sota` | `sota`, `baseline`, `random` | **Experiment A**: Feature source |
| `--input-feat-dim` | `str` | `auto` | `auto`, `<int>` | **Experiment B**: Override feature dimension |
| `--fusion-mode` | `str` | `film` | `film`, `concat`, `none` | **Experiment C**: Fusion strategy |

**Ablation Details:**
- `sota`: Qwen2-1.5B + SigLIP-SO400M (2688-dim) - **Our method**
- `baseline`: MiniLM + CLIP (1536-dim) - **Comparison baseline**
- `random`: Learnable `nn.Embedding` (172-dim) - **Lower bound, no content**

### Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--eval-ranking` | `flag` | `True` | Compute Recall@K, NDCG@K, MRR |
| `--no-eval-ranking` | `flag` | - | Skip ranking metrics (faster) |
| `--n-neg-eval` | `int` | `100` | Number of negatives per positive for ranking |
| `--eval-sample-size` | `int` | `None` | Subsample test set for faster ranking eval (e.g., `10000`). Use `None` for full evaluation (paper results). |

**Evaluation Speed Guide:**
| Mode | `--eval-sample-size` | `--n-neg-eval` | Est. Time |
|------|---------------------|----------------|-----------|
| **Development** | `10000` | `20` | ~3-5 min |
| **Validation** | `30000` | `50` | ~30 min |
| **Paper Results** | `None` (full) | `100` | ~6-8 hours |

### Logging Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--log-dir` | `str` | `runs` | TensorBoard log directory |
| `--save-dir` | `str` | `checkpoints` | Checkpoint save directory |
| `--run-name` | `str` | `None` | Custom run name (auto-generated: `mmtgn_<dataset>_<timestamp>`) |

### Other Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | `int` | `42` | Random seed for reproducibility |
| `--device` | `str` | `cuda` | Device: `cuda` or `cpu` |
| `--num-workers` | `int` | `0` | DataLoader workers (keep 0 for TGN) |

---

## 🚀 Ready-to-Run Ablation Commands

### Step 0: Environment Setup (Required First)

```bash
# Request GPU allocation on Great Lakes
salloc --account cse576f25s001_class \
       --partition gpu_mig40,gpu,spgpu \
       --nodes 1 --ntasks 1 --cpus-per-task 1 \
       --gpus 1 --mem 64G --time 06:00:00

# Activate environment
conda activate mmtgn
cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn
```

### Experiment 0a: Quick Smoke Test (FAST - 5 min)

```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --epochs 1 \
    --eval-ranking \
    --n-neg-eval 20 \
    --eval-sample-size 10000 \
    --run-name "smoke_test_fast"
```
**Expected**: ~5 min, verifies code runs without errors

### Experiment 0b: Full Smoke Test (1 Epoch - Full Evaluation)

```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --embedding-dim 172 \
    --n-layers 2 \
    --n-neighbors 15 \
    --batch-size 200 \
    --epochs 1 \
    --lr 1e-4 \
    --loss bce \
    --eval-ranking \
    --n-neg-eval 100 \
    --run-name "smoke_test_full"
```
**Expected**: ~25 min, Loss ~0.5-0.6, AP ~0.75-0.80

---

### Experiment A: Vanilla Baseline (Random Features - LOWER BOUND)

**Quick Test (verify vanilla learns after bug fix):**
```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --node-feature-type random \
    --fusion-mode none \
    --epochs 5 \
    --eval-ranking \
    --n-neg-eval 20 \
    --eval-sample-size 10000 \
    --run-name "vanilla_quick_test"
```
**Expected after fix**: Val AP should INCREASE (not stay at ~0.47). Target: AP > 0.55 by epoch 5.

**Full Experiment:**
```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --node-feature-type random \
    --fusion-mode none \
    --embedding-dim 172 \
    --n-layers 2 \
    --n-neighbors 15 \
    --batch-size 200 \
    --epochs 50 \
    --lr 1e-4 \
    --loss bce \
    --patience 5 \
    --eval-ranking \
    --n-neg-eval 100 \
    --run-name "ablation_A_vanilla_random"
```
**Purpose**: Establishes lower bound. No semantic content - pure collaborative filtering.

---

### Experiment B1: SOTA Features (Our Full Method)

**Quick Test (verify ranking eval works after memory fix):**
```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --node-feature-type sota \
    --epochs 1 \
    --eval-ranking \
    --n-neg-eval 20 \
    --eval-sample-size 10000 \
    --run-name "sota_ranking_test"
```
**Expected after fix**: Ranking evaluation should complete without "memory to time in past" error.

**Full Experiment:**
```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --node-feature-type sota \
    --fusion-mode film \
    --embedding-dim 172 \
    --n-layers 2 \
    --n-neighbors 15 \
    --batch-size 200 \
    --epochs 50 \
    --lr 1e-4 \
    --loss bce \
    --patience 5 \
    --eval-ranking \
    --n-neg-eval 100 \
    --run-name "experiment_B1_sota_film"
```
**Purpose**: Full multimodal pipeline with Qwen2 + SigLIP features.

---

### Experiment B2: Baseline Encoders (Weaker Encoders)

First, generate baseline features (if not done):
```bash
python data/script/generate_embeddings.py \
    --csv-path data/datasets/movielens-32m/movielens-modern/ml-modern/enriched.csv \
    --image-dir data/datasets/movielens-32m/movielens-modern/ml-modern-posters \
    --output-dir data/datasets/movielens-32m/features/baseline \
    --dataset-name ml-modern \
    --id-col movieId \
    --text-col overview \
    --text-model baseline \
    --image-model clip \
    --batch-size 32
```

Then format for TGN:
```bash
python data/script/tgn_formatter.py \
    --ratings-path data/datasets/movielens-32m/movielens-modern/ml-modern/ratings.csv \
    --features-dir data/datasets/movielens-32m/features/baseline \
    --output-dir data/processed_baseline \
    --dataset-name ml-modern \
    --text-model baseline \
    --image-model clip
```

Train:
```bash
python train_mmtgn.py \
    --data-dir data/processed_baseline \
    --dataset ml-modern \
    --node-feature-type sota \
    --fusion-mode film \
    --embedding-dim 172 \
    --n-layers 2 \
    --n-neighbors 15 \
    --batch-size 200 \
    --epochs 50 \
    --lr 1e-4 \
    --loss bce \
    --patience 5 \
    --eval-ranking \
    --n-neg-eval 100 \
    --run-name "experiment_B2_baseline_encoders"
```
**Purpose**: Compare SOTA vs baseline encoder quality.

---

### Experiment C1: Fusion Mode - FiLM (Default)

```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --node-feature-type sota \
    --fusion-mode film \
    --embedding-dim 172 \
    --epochs 50 \
    --eval-ranking \
    --run-name "experiment_C1_fusion_film"
```

### Experiment C2: Fusion Mode - Concatenation

```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --node-feature-type sota \
    --fusion-mode concat \
    --embedding-dim 172 \
    --epochs 50 \
    --eval-ranking \
    --run-name "experiment_C2_fusion_concat"
```

### Experiment C3: Fusion Mode - None (Temporal Only)

```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --node-feature-type sota \
    --fusion-mode none \
    --embedding-dim 172 \
    --epochs 50 \
    --eval-ranking \
    --run-name "experiment_C3_fusion_none"
```

---

### Experiment D: Loss Function Comparison

**D1: BCE Loss (Default, Stable)**
```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --loss bce \
    --epochs 50 \
    --eval-ranking \
    --run-name "experiment_D1_loss_bce"
```

**D2: BPR Loss (Ranking-Optimized)**
```bash
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --loss bpr \
    --epochs 50 \
    --eval-ranking \
    --run-name "experiment_D2_loss_bpr"
```

---

## 📦 Data Format Specifications

### Edge List CSV (`ml_<dataset>.csv`)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `u` | `int` | Source node ID (1-based) | `1` |
| `i` | `int` | Destination node ID (1-based) | `10205` |
| `ts` | `float` | Unix timestamp | `1609459200.0` |
| `label` | `int` | Edge label (always 1 for ratings) | `1` |
| `idx` | `int` | Edge index (1-based) | `1` |

**Example:**
```csv
u,i,ts,label,idx
1,10205,1609459200.0,1,1
1,10892,1609459500.0,1,2
2,10205,1609460000.0,1,3
```

### Node Features NPY (`ml_<dataset>.npy`)

- **Shape**: `(n_nodes, feature_dim)` = `(32170, 2688)` for ML-Modern SOTA
- **Row 0**: All zeros (padding)
- **Rows 1-10200**: User embeddings (projected or zeros)
- **Rows 10201-32169**: Item embeddings (SOTA multimodal)

### Node Map JSON (`node_map.json`)

```json
{
  "num_users": 10200,
  "num_items": 21969,
  "feature_dim": 2688,
  "padding_idx": 0,
  "user_range": [1, 10200],
  "item_range": [10201, 32169]
}
```

---

## 🔧 Core API Documentation

### `train_mmtgn.py`

**Main Function:**
```python
def train(args: argparse.Namespace) -> Dict[str, float]:
    """
    Main training loop for MM-TGN.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary of final evaluation metrics
    """
```

### `mmtgn.py`

**MMTGN Class:**
```python
class MMTGN(nn.Module):
    """
    Multimodal Temporal Graph Network.
    
    Args:
        neighbor_finder: NeighborFinder - Temporal neighbor lookup
        node_features: np.ndarray - Shape (n_nodes, feat_dim)
        edge_features: np.ndarray - Shape (n_edges, edge_dim)
        device: str - 'cuda' or 'cpu'
        n_layers: int - Number of graph attention layers (default: 2)
        n_heads: int - Attention heads (default: 2)
        embedding_dim: int - Working dimension (default: 172)
        dropout: float - Dropout rate (default: 0.1)
        use_memory: bool - Enable TGN memory (default: True)
        memory_dimension: int - Memory vector size (default: 172)
        num_users: int - Number of users (for hybrid features)
        num_items: int - Number of items (for hybrid features)
        item_features: np.ndarray - SOTA item embeddings
        use_hybrid_features: bool - Use HybridNodeFeatures (default: True)
        use_random_item_features: bool - Ablation: random embeddings (default: False)
        use_film: bool - Enable FiLM fusion (default: True)
        structural_dim: int - Channel 2 dimension (None = bypass)
    """
    
    def compute_temporal_embeddings(
        self,
        source_nodes: np.ndarray,
        destination_nodes: np.ndarray,
        negative_nodes: np.ndarray,
        edge_times: np.ndarray,
        edge_idxs: np.ndarray,
        n_neighbors: int = 20,
        structural_embeddings: Optional[torch.Tensor] = None,
        skip_memory_update: bool = False  # NEW: For evaluation
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute temporal embeddings for a batch.
        
        Args:
            skip_memory_update: If True, skip memory update step. Use for
                               negative scoring in evaluation to avoid 
                               "update memory to time in past" errors.
        
        Returns:
            (source_emb, dest_emb, neg_emb) - All shape [batch, embedding_dim]
        """
    
    def compute_edge_probabilities(
        self,
        source_nodes: np.ndarray,
        destination_nodes: np.ndarray,
        negative_nodes: np.ndarray,
        edge_times: np.ndarray,
        edge_idxs: np.ndarray,
        n_neighbors: int = 20,
        skip_memory_update: bool = False  # NEW: For evaluation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute link probabilities.
        
        Args:
            skip_memory_update: If True, skip memory update. Use for negative
                               scoring in evaluation to avoid timestamp conflicts.
        
        Returns:
            (pos_probs, neg_probs) - Both shape [batch]
        """
```

**Factory Function:**
```python
def create_mmtgn(
    dataset: TemporalDataset,
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
    structural_dim: Optional[int] = None,
    use_random_item_features: bool = False
) -> MMTGN:
    """Create MMTGN from TemporalDataset."""
```

**Loss Functions:**
```python
def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""

def bce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    """Binary Cross-Entropy loss."""
```

### `modules/embedding.py`

**HybridNodeFeatures:**
```python
class HybridNodeFeatures(nn.Module):
    """
    Unified node feature provider.
    
    Args:
        num_users: int - Number of user nodes
        num_items: int - Number of item nodes
        item_features: np.ndarray - SOTA features (num_items, feat_dim)
        embedding_dim: int - Target dimension (172)
        dropout: float - Projector dropout (0.1)
        freeze_items: bool - Freeze SOTA features (True)
        use_random_items: bool - Ablation mode (False)
    
    Methods:
        forward(node_ids: torch.Tensor) -> torch.Tensor:
            Get features for node IDs. Shape: (batch, embedding_dim)
        
        get_all_features() -> torch.Tensor:
            Full feature matrix. Shape: (total_nodes, embedding_dim)
    """
```

**FiLMConditioner:**
```python
class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation.
    
    Formula: output = γ(cond) ⊙ input + β(cond)
    
    Args:
        input_dim: int - Input feature dimension
        cond_dim: int - Conditioning feature dimension
        hidden_dim: int - Hidden layer dimension (optional)
    """
```

**UserStateFiLM:**
```python
class UserStateFiLM(nn.Module):
    """
    User-State FiLM for cold-start adaptation.
    
    Uses user's TGN memory to modulate item features.
    
    Formula: h_adapted = γ(h_user) ⊙ h_item + β(h_user)
    
    Args:
        user_dim: int - User memory dimension (172)
        item_dim: int - Item feature dimension (172)
        hidden_dim: int - Hidden dimension (optional)
    """
```

### `utils/metrics.py`

**RankingMetrics:**
```python
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
```

**Metric Functions:**
```python
def compute_recall_at_k(pos_scores: np.ndarray, neg_scores_matrix: np.ndarray, k: int) -> float:
    """Compute Recall@K."""

def compute_ndcg_at_k(pos_scores: np.ndarray, neg_scores_matrix: np.ndarray, k: int) -> float:
    """Compute NDCG@K."""

def compute_mrr(pos_scores: np.ndarray, neg_scores_matrix: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank."""

def compute_all_ranking_metrics(
    model: MMTGN,
    data: Data,
    neg_sampler: NegativeSamplerForEval,
    n_neighbors: int,
    batch_size: int,
    device: str
) -> RankingMetrics:
    """Compute all ranking metrics for a dataset split."""
```

---

## 📊 TensorBoard Setup (Great Lakes HPC)

### Log Locations
```
/scratch/.../mm-tgn/
├── runs/                          # TensorBoard event files
│   └── mmtgn_<dataset>_<timestamp>/
│       └── events.out.tfevents.*
├── checkpoints/                   # Model checkpoints
│   └── mmtgn_<dataset>_<timestamp>/
│       ├── best_model.pt
│       └── train.log
```

### Quick Start
```bash
# On compute node
cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn
./scripts/launch_tensorboard.sh &

# Or manually (IPv4 only - required for HPC):
tensorboard --logdir=runs --port=6006 --host=0.0.0.0 &
```

### SSH Tunnel (from LOCAL machine)
```bash
# Replace gl1509 with your compute node
ssh -L 6006:gl1509.arc-ts.umich.edu:6006 huseynli@greatlakes.arc-ts.umich.edu
```

Open: **http://localhost:6006**

### Management Commands
| Action | Command |
|--------|---------|
| Start | `./scripts/launch_tensorboard.sh &` |
| Check | `ps aux \| grep tensorboard` |
| Stop | `pkill -f tensorboard` |
| Custom port | `TENSORBOARD_PORT=6007 ./scripts/launch_tensorboard.sh &` |

---

## 📁 File Dictionary

### Core Model (`/mm-tgn/`)

| File | Lines | Responsibility |
|------|-------|---------------|
| `mmtgn.py` | ~700 | MMTGN model, FiLM fusion, loss functions, factory |
| `dataset.py` | ~500 | TemporalDataset, temporal splits, inductive tracking |
| `train_mmtgn.py` | ~900 | Training loop, evaluation, CLI arguments |

### Modules (`/mm-tgn/modules/`)

| File | Lines | Responsibility |
|------|-------|---------------|
| `embedding.py` | ~770 | HybridNodeFeatures, FiLM, UserStateFiLM, TGN embeddings |
| `memory.py` | ~100 | TGN Memory class |
| `memory_updater.py` | ~150 | GRU/RNN/LSTM memory updaters |
| `message_aggregator.py` | ~80 | Message aggregation (last, mean) |
| `message_function.py` | ~60 | Message computation (MLP, identity) |

### Data Pipeline (`/mm-tgn/data/script/`)

| File | Lines | Responsibility |
|------|-------|---------------|
| `generate_embeddings.py` | ~300 | Universal multimodal encoder |
| `tgn_formatter.py` | ~200 | ratings + features → TGN format |
| `movielens_scraper.py` | ~400 | TMDB API scraper |
| `movielens_subset.py` | ~200 | ML-Modern subset creation |

### Utilities (`/mm-tgn/utils/`)

| File | Lines | Responsibility |
|------|-------|---------------|
| `utils.py` | ~400 | NeighborFinder, RandEdgeSampler, EarlyStopMonitor |
| `metrics.py` | ~400 | Recall@K, NDCG@K, MRR, evaluation functions |

---

## 🔢 Dimension Reference

| Component | Dimension | Notes |
|-----------|-----------|-------|
| TGN Embedding | 172 | From TGN paper |
| TGN Memory | 172 | Matches embedding |
| SOTA Text (Qwen2) | 1536 | |
| SOTA Image (SigLIP) | 1152 | SO400M |
| **SOTA Combined** | **2688** | |
| Baseline Text (MiniLM) | 768 | |
| Baseline Image (CLIP) | 768 | ViT-L-14 |
| **Baseline Combined** | **1536** | |
| User Embedding | 172 | Learnable |
| Item Projected | 172 | After projector |

---

## 🚨 Known Issues & Solutions

| Issue | Solution | Status |
|-------|----------|--------|
| `torch.load` WeightsOnly error (PyTorch 2.6) | Use `weights_only=False` | ✅ Fixed |
| `ReduceLROnPlateau` verbose parameter | Remove `verbose=True` | ✅ Fixed |
| TensorBoard IPv6 binding error | Use `--host=0.0.0.0` | ✅ Fixed |
| Memory assertion "update to time in past" | Added `skip_memory_update=True` for negative scoring | ✅ Fixed |
| Vanilla model not learning (AP ≈ 0.47) | Fixed gradient flow: use direct assignment instead of `.data.copy_()` | ✅ Fixed |
| Gradient warning (non-leaf tensor) | Harmless, ignore | ⚠️ Cosmetic |
| PyG extension warnings | Fallback works | ⚠️ Non-blocking |

### Critical Bug Details

**Bug 1: Memory Update Assertion Error**
```
AssertionError: Trying to update memory to time in the past
```
- **Cause**: Negative scoring in `evaluate_ranking` used repeated timestamps
- **Fix**: Added `skip_memory_update=True` parameter to skip memory updates when scoring negatives
- **Location**: `mmtgn.py` (`compute_edge_probabilities`, `compute_temporal_embeddings`)

**Bug 2: Vanilla Model Not Learning**
- **Symptom**: Val AP stayed at ~0.47 (below random 0.5) for all epochs
- **Cause**: `.data.copy_()` broke gradient flow to learnable item embeddings
- **Fix**: Changed to direct tensor assignment in `update_node_features()`
- **Location**: `mmtgn.py` line ~302

```python
# BEFORE (broken - no gradients flow):
self.node_raw_features.data.copy_(all_features)

# AFTER (fixed - gradients preserved):
self.embedding_module.node_features = all_features
```

---

## 📊 Expected Metrics

### Smoke Test (1 Epoch)
| Metric | SOTA Expected | Vanilla Expected | Notes |
|--------|---------------|------------------|-------|
| Train Loss | 0.5 - 0.7 | 0.6 - 0.7 | Should decrease |
| Val AP | 0.75 - 0.85 | 0.55 - 0.70 | Better than 0.5 |
| Val AUC | 0.75 - 0.85 | 0.55 - 0.70 | Better than 0.5 |
| Time | ~25 min | ~20 min | On 1 GPU |

### Full Training (50 Epochs)
| Metric | SOTA Transductive | SOTA Inductive | Vanilla | Notes |
|--------|-------------------|----------------|---------|-------|
| Test AP | > 0.85 | > 0.80 | > 0.60 | SOTA >> Vanilla = success |
| Test AUC | > 0.85 | > 0.80 | > 0.60 | |
| Recall@10 | > 0.05 | > 0.02 | > 0.01 | Higher = better |
| NDCG@10 | > 0.03 | > 0.01 | > 0.01 | Higher = better |

### Key Comparisons (Research Validation)
| Comparison | Expected Outcome | What It Proves |
|------------|------------------|----------------|
| SOTA AP > Vanilla AP | +0.15 to +0.25 | Multimodal features help |
| SOTA Inductive > Vanilla Inductive | Larger gap | Cold-start hypothesis |
| SOTA > Baseline | +0.05 to +0.10 | Better encoders matter |

---

## 🔄 Context Restoration Checklist

```bash
# 1. Read this documentation
cat ARCHITECTURE.md

# 2. Request GPU
salloc --account cse576f25s001_class \
       --partition gpu_mig40,gpu,spgpu \
       --gpus 1 --mem 64G --time 06:00:00

# 3. Activate environment
conda activate mmtgn

# 4. Navigate to project
cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn

# 5. Verify data
ls -la data/processed/
# Should see: ml_ml-modern.csv, ml_ml-modern.npy, node_map.json

# 6. Run smoke test
python train_mmtgn.py --data-dir data/processed --dataset ml-modern --epochs 1

# 7. Start TensorBoard
./scripts/launch_tensorboard.sh &

# 8. SSH tunnel (from LOCAL machine)
# ssh -L 6006:gl1509.arc-ts.umich.edu:6006 huseynli@greatlakes.arc-ts.umich.edu

# 9. View TensorBoard
# Open: http://localhost:6006
```

---

## 📝 Changelog

### 2025-12-03 (v2.1) - Critical Bug Fixes
- **🐛 CRITICAL FIX: Memory Assertion Error**
  - Added `skip_memory_update` parameter to `compute_edge_probabilities()` and `compute_temporal_embeddings()`
  - Fixes `AssertionError: Trying to update memory to time in the past` during ranking evaluation
- **🐛 CRITICAL FIX: Vanilla Model Not Learning**
  - Fixed gradient flow for random item embeddings
  - Changed `.data.copy_()` to direct tensor assignment in `update_node_features()`
  - Vanilla model now learns properly (AP should improve from ~0.47)
- **⚡ NEW: `--eval-sample-size` argument**
  - Subsample test set for faster ranking evaluation during development
  - Use `--eval-sample-size 10000 --n-neg-eval 20` for ~75x faster evaluation
- **📚 Updated API documentation** with new parameters

### 2025-12-03 (v2.0)
- **Complete CLI Reference**: All arguments with types, defaults, descriptions
- **Ready-to-Run Commands**: Copy-paste commands for all ablation experiments
- **Data Format Specs**: Detailed CSV/NPY/JSON format documentation
- **Core API Documentation**: Function signatures and descriptions
- **Added Table of Contents**

### 2025-12-03 (v1.1)
- Added TensorBoard setup for Great Lakes HPC
- Added `scripts/launch_tensorboard.sh` with IPv4 fix
- Fixed `verbose` parameter in ReduceLROnPlateau
- Fixed `torch.load` for PyTorch 2.6

### 2025-12-03 (v1.0)
- Added ablation CLI arguments
- Implemented `UserStateFiLM`
- Fixed Amazon category parsing
- Created initial documentation

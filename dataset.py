"""
MM-TGN Dataset Module

Handles temporal data loading and splitting for TGN-style training.

Key Features:
1. Temporal Split: Chronological train/val/test (no data leakage)
2. Negative Sampling: Random negatives for link prediction
3. Data Containers: Structured storage for TGN consumption
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import json


@dataclass
class Data:
    """
    Container for temporal graph data in TGN format.
    
    All arrays are aligned by interaction index.
    """
    sources: np.ndarray          # Source node IDs (users)
    destinations: np.ndarray     # Destination node IDs (items)
    timestamps: np.ndarray       # Unix timestamps
    edge_idxs: np.ndarray        # Edge indices (1-based)
    labels: np.ndarray           # Interaction labels (ratings or binary)
    
    # Computed properties
    n_interactions: int = 0
    n_unique_nodes: int = 0
    
    def __post_init__(self):
        self.n_interactions = len(self.sources)
        self.n_unique_nodes = len(np.unique(np.concatenate([self.sources, self.destinations])))
    
    def __len__(self):
        return self.n_interactions


class TemporalDataset:
    """
    Temporal dataset manager for MM-TGN.
    
    Loads TGN-formatted data and provides temporal splits.
    
    Args:
        csv_path: Path to TGN-formatted CSV (u, i, ts, label, idx)
        features_path: Path to node features .npy file
        node_map_path: Path to node_map.json (from tgn_formatter.py)
        val_ratio: Fraction of data for validation (chronological)
        test_ratio: Fraction of data for testing (chronological)
        inductive_ratio: Fraction of nodes to hide for inductive evaluation
    """
    
    def __init__(
        self,
        csv_path: str,
        features_path: str,
        node_map_path: str,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        inductive_ratio: float = 0.1,
        randomize_features: bool = False  # For ablation studies
    ):
        self.csv_path = Path(csv_path)
        self.features_path = Path(features_path)
        self.node_map_path = Path(node_map_path)
        
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.inductive_ratio = inductive_ratio
        self.randomize_features = randomize_features
        
        # Load data
        self._load_data()
        self._load_features()
        self._load_node_map()
        
        # Create splits
        self._create_temporal_splits()
    
    def _load_data(self):
        """Load interaction data from CSV."""
        print(f"📂 Loading interactions from: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # Validate required columns
        required_cols = ['u', 'i', 'ts', 'label', 'idx']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")
        
        # Ensure chronological order
        df = df.sort_values('ts').reset_index(drop=True)
        
        self.sources = df['u'].values.astype(np.int64)
        self.destinations = df['i'].values.astype(np.int64)
        self.timestamps = df['ts'].values.astype(np.float64)
        self.labels = df['label'].values.astype(np.float32)
        self.edge_idxs = df['idx'].values.astype(np.int64)
        
        self.n_interactions = len(df)
        self.n_nodes = max(self.sources.max(), self.destinations.max()) + 1
        
        # =====================================================================
        # VALIDATE 1-BASED INDEXING
        # Index 0 should NOT appear in data (reserved for padding)
        # =====================================================================
        min_src = self.sources.min()
        min_dst = self.destinations.min()
        min_node = min(min_src, min_dst)
        
        if min_node == 0:
            print("   ⚠️  WARNING: Found node index 0 in data!")
            print("   ⚠️  This may indicate 0-based indexing. TGN expects 1-based indexing.")
            print("   ⚠️  Index 0 is reserved for padding (null neighbors).")
            # Count how many zeros
            n_zero_src = (self.sources == 0).sum()
            n_zero_dst = (self.destinations == 0).sum()
            print(f"   ⚠️  Zero indices: {n_zero_src} sources, {n_zero_dst} destinations")
        else:
            print(f"   ✓ 1-based indexing verified (min node ID = {min_node})")
        
        print(f"   ✓ Loaded {self.n_interactions:,} interactions")
        print(f"   ✓ {self.n_nodes:,} total nodes (including padding at 0)")
        print(f"   ✓ Node ID range: [{min_node}, {max(self.sources.max(), self.destinations.max())}]")
        print(f"   ✓ Edge ID range: [{self.edge_idxs.min()}, {self.edge_idxs.max()}]")
    
    def _load_features(self, use_mmap: bool = False):
        """
        Load pre-computed node features.
        
        Args:
            use_mmap: Use memory mapping for large files (reduces RAM usage)
        """
        print(f"📂 Loading features from: {self.features_path}")
        
        # Memory-efficient loading for large files
        if use_mmap:
            print("   ℹ️  Using memory-mapped loading (RAM efficient)")
            self.node_features = np.load(self.features_path, mmap_mode='r')
            # Need to copy for write access if needed
            self.node_features = np.array(self.node_features, dtype=np.float32)
        else:
            self.node_features = np.load(self.features_path).astype(np.float32)
        
        if self.randomize_features:
            print("   ⚠️  ABLATION MODE: Randomizing features!")
            self.node_features = np.random.randn(*self.node_features.shape).astype(np.float32)
        
        print(f"   ✓ Feature shape: {self.node_features.shape}")
        print(f"   ✓ Feature dtype: {self.node_features.dtype}")
        print(f"   ✓ Memory: {self.node_features.nbytes / 1e6:.1f} MB")
        
        # Validate alignment with 1-based indexing
        # Row 0 = padding, Row 1..N = actual nodes
        if self.node_features.shape[0] < self.n_nodes:
            print(f"   ⚠️  Warning: Feature matrix has fewer rows ({self.node_features.shape[0]}) than nodes ({self.n_nodes}). Padding...")
            pad_rows = self.n_nodes - self.node_features.shape[0]
            padding = np.zeros((pad_rows, self.node_features.shape[1]), dtype=np.float32)
            self.node_features = np.vstack([self.node_features, padding])
        
        # Validate that row 0 is zero (padding)
        if np.any(self.node_features[0] != 0):
            print("   ⚠️  Warning: Row 0 (padding) is not zero. This may cause issues.")
            print("   ℹ️  For 1-based indexing, Row 0 should be all zeros (padding vector)")
        else:
            print("   ✓ Row 0 (padding) is correctly zero")
        
        # Create edge features (empty for now, can be extended)
        # Edge indices are 1-based, so we need n_interactions + 1 rows
        self.edge_features = np.zeros((self.n_interactions + 1, 1), dtype=np.float32)
    
    def _load_node_map(self):
        """Load node mapping metadata."""
        print(f"📂 Loading node map from: {self.node_map_path}")
        
        with open(self.node_map_path, 'r') as f:
            self.node_map = json.load(f)
        
        self.num_users = self.node_map['num_users']
        self.num_items = self.node_map['num_items']
        self.feature_dim = self.node_map['feature_dim']
        
        # 1-based indexing info (Index 0 = padding)
        self.padding_idx = self.node_map.get('padding_idx', 0)
        self.user_id_range = self.node_map.get('user_id_range', [1, self.num_users])
        self.item_id_range = self.node_map.get('item_id_range', 
                                                [self.num_users + 1, self.num_users + self.num_items])
        
        print(f"   ✓ {self.num_users:,} users (indices {self.user_id_range[0]}-{self.user_id_range[1]})")
        print(f"   ✓ {self.num_items:,} items (indices {self.item_id_range[0]}-{self.item_id_range[1]})")
        print(f"   ✓ Feature dimension: {self.feature_dim}")
        print(f"   ✓ Padding index: {self.padding_idx}")
    
    def _create_temporal_splits(self):
        """
        Create chronological train/val/test splits.
        
        The key principle: NO FUTURE LEAKAGE.
        - Train: oldest interactions
        - Val: middle interactions  
        - Test: newest interactions
        """
        print("📊 Creating temporal splits...")
        
        # Compute split boundaries
        val_time_idx = int(self.n_interactions * (1 - self.val_ratio - self.test_ratio))
        test_time_idx = int(self.n_interactions * (1 - self.test_ratio))
        
        # Get timestamps at boundaries
        val_time = self.timestamps[val_time_idx]
        test_time = self.timestamps[test_time_idx]
        
        # Create masks
        train_mask = self.timestamps < val_time
        val_mask = (self.timestamps >= val_time) & (self.timestamps < test_time)
        test_mask = self.timestamps >= test_time
        
        print(f"   ✓ Train: {train_mask.sum():,} ({train_mask.mean():.1%})")
        print(f"   ✓ Val:   {val_mask.sum():,} ({val_mask.mean():.1%})")
        print(f"   ✓ Test:  {test_mask.sum():,} ({test_mask.mean():.1%})")
        
        # Store masks
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        
        # Create inductive split (nodes seen only in test)
        self._create_inductive_split()
    
    def _create_inductive_split(self):
        """
        Identify nodes for transductive vs inductive evaluation.
        
        CRITICAL FOR RESEARCH:
        - Transductive: Nodes seen during training (can use history)
        - Inductive: New nodes in test (cold-start scenario)
        
        This split is essential for evaluating MM-TGN's cold-start hypothesis:
        Multimodal features should help more for inductive (new) nodes.
        """
        # Collect nodes seen in each split
        self.train_nodes = set(self.sources[self.train_mask]) | set(self.destinations[self.train_mask])
        val_nodes = set(self.sources[self.val_mask]) | set(self.destinations[self.val_mask])
        test_nodes = set(self.sources[self.test_mask]) | set(self.destinations[self.test_mask])
        
        # Transductive: nodes seen in train (can be in val/test too)
        self.transductive_nodes = self.train_nodes
        
        # Inductive: nodes that appear ONLY in test (not in train or val)
        # These are the "cold start" nodes
        self.inductive_nodes_set = test_nodes - self.train_nodes - val_nodes
        self.inductive_nodes = np.array(list(self.inductive_nodes_set))
        
        # Create masks for test set
        test_sources = self.sources[self.test_mask]
        test_destinations = self.destinations[self.test_mask]
        
        # Inductive test: at least one node is new
        self.inductive_test_mask = (
            np.isin(test_sources, self.inductive_nodes) | 
            np.isin(test_destinations, self.inductive_nodes)
        )
        
        # Transductive test: both nodes were seen in training
        self.transductive_test_mask = ~self.inductive_test_mask
        
        # Separate into user and item inductive nodes
        inductive_users = self.inductive_nodes_set & set(range(1, self.num_users + 1))
        inductive_items = self.inductive_nodes_set & set(range(self.num_users + 1, self.num_users + self.num_items + 1))
        
        print(f"   ✓ Transductive nodes (seen in train): {len(self.transductive_nodes):,}")
        print(f"   ✓ Inductive nodes (new in test): {len(self.inductive_nodes):,}")
        print(f"      - Inductive users: {len(inductive_users):,}")
        print(f"      - Inductive items: {len(inductive_items):,}")
        print(f"   ✓ Test interactions:")
        print(f"      - Transductive: {self.transductive_test_mask.sum():,}")
        print(f"      - Inductive (cold-start): {self.inductive_test_mask.sum():,}")
    
    def get_data_split(self, split: str) -> Data:
        """
        Get data for a specific split.
        
        Args:
            split: One of 'train', 'val', 'test', 'full'
        
        Returns:
            Data object with filtered interactions
        """
        if split == 'train':
            mask = self.train_mask
        elif split == 'val':
            mask = self.val_mask
        elif split == 'test':
            mask = self.test_mask
        elif split == 'full':
            mask = np.ones(self.n_interactions, dtype=bool)
        else:
            raise ValueError(f"Unknown split: {split}")
        
        return Data(
            sources=self.sources[mask],
            destinations=self.destinations[mask],
            timestamps=self.timestamps[mask],
            edge_idxs=self.edge_idxs[mask],
            labels=self.labels[mask]
        )
    
    def get_full_data(self) -> Data:
        """Get all interactions (for building neighbor finder)."""
        return self.get_data_split('full')
    
    @property
    def train_data(self) -> Data:
        return self.get_data_split('train')
    
    @property
    def val_data(self) -> Data:
        return self.get_data_split('val')
    
    @property
    def test_data(self) -> Data:
        return self.get_data_split('test')
    
    def get_transductive_test_data(self) -> Data:
        """Get test interactions where both nodes were seen during training."""
        # First get test mask, then apply transductive mask within test
        test_indices = np.where(self.test_mask)[0]
        trans_indices = test_indices[self.transductive_test_mask]
        
        return Data(
            sources=self.sources[trans_indices],
            destinations=self.destinations[trans_indices],
            timestamps=self.timestamps[trans_indices],
            edge_idxs=self.edge_idxs[trans_indices],
            labels=self.labels[trans_indices]
        )
    
    def get_inductive_test_data(self) -> Data:
        """Get test interactions involving new (cold-start) nodes."""
        # First get test mask, then apply inductive mask within test
        test_indices = np.where(self.test_mask)[0]
        induct_indices = test_indices[self.inductive_test_mask]
        
        return Data(
            sources=self.sources[induct_indices],
            destinations=self.destinations[induct_indices],
            timestamps=self.timestamps[induct_indices],
            edge_idxs=self.edge_idxs[induct_indices],
            labels=self.labels[induct_indices]
        )
    
    def get_all_items(self) -> np.ndarray:
        """Get all unique item IDs (for negative sampling)."""
        # Items are at indices [num_users + 1, num_users + num_items] with 1-based indexing
        return np.arange(self.num_users + 1, self.num_users + self.num_items + 1)
    
    def compute_time_statistics(self) -> Tuple[float, float, float, float]:
        """
        Compute time difference statistics for TGN normalization.
        
        Returns:
            (mean_src, std_src, mean_dst, std_dst)
        """
        # Compute time differences between consecutive interactions
        sources_sorted_idx = np.argsort(self.sources)
        dests_sorted_idx = np.argsort(self.destinations)
        
        # Source time diffs
        src_times = self.timestamps[sources_sorted_idx]
        src_nodes = self.sources[sources_sorted_idx]
        src_diffs = []
        
        prev_node = -1
        prev_time = 0
        for node, time in zip(src_nodes, src_times):
            if node == prev_node:
                src_diffs.append(time - prev_time)
            prev_node = node
            prev_time = time
        
        # Destination time diffs
        dst_times = self.timestamps[dests_sorted_idx]
        dst_nodes = self.destinations[dests_sorted_idx]
        dst_diffs = []
        
        prev_node = -1
        prev_time = 0
        for node, time in zip(dst_nodes, dst_times):
            if node == prev_node:
                dst_diffs.append(time - prev_time)
            prev_node = node
            prev_time = time
        
        src_diffs = np.array(src_diffs) if src_diffs else np.array([0])
        dst_diffs = np.array(dst_diffs) if dst_diffs else np.array([0])
        
        return (
            float(np.mean(src_diffs)),
            float(np.std(src_diffs)) + 1e-6,
            float(np.mean(dst_diffs)),
            float(np.std(dst_diffs)) + 1e-6
        )
    
    def get_item_features_only(self) -> np.ndarray:
        """
        Get only item features (for multimodal adapter).
        
        With 1-based indexing:
        - Index 0 = Padding
        - Users: indices [1, num_users]
        - Items: indices [num_users + 1, num_users + num_items]
        
        Returns:
            Item features [num_items, feature_dim]
        """
        # Items start at index (num_users + 1) with 1-based indexing
        item_start = self.num_users + 1
        item_end = self.num_users + 1 + self.num_items
        return self.node_features[item_start:item_end]
    
    def __repr__(self) -> str:
        return (
            f"TemporalDataset(\n"
            f"  interactions={self.n_interactions:,},\n"
            f"  users={self.num_users:,},\n"
            f"  items={self.num_items:,},\n"
            f"  feature_dim={self.feature_dim},\n"
            f"  train={self.train_mask.sum():,},\n"
            f"  val={self.val_mask.sum():,},\n"
            f"  test={self.test_mask.sum():,}\n"
            f")"
        )


def load_dataset(
    data_dir: str,
    dataset_name: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> TemporalDataset:
    """
    Convenience function to load a dataset.
    
    Args:
        data_dir: Directory containing ml_{dataset_name}.csv, .npy, node_map.json
        dataset_name: Dataset identifier (e.g., 'ml-modern')
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
    
    Returns:
        Configured TemporalDataset
    """
    data_dir = Path(data_dir)
    
    return TemporalDataset(
        csv_path=str(data_dir / f"ml_{dataset_name}.csv"),
        features_path=str(data_dir / f"ml_{dataset_name}.npy"),
        node_map_path=str(data_dir / "node_map.json"),
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )


if __name__ == "__main__":
    # Quick test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    
    dataset = load_dataset(args.data_dir, args.dataset)
    print(dataset)
    
    # Test splits
    train = dataset.train_data
    print(f"\nTrain data: {len(train)} interactions")
    print(f"  First ts: {train.timestamps[0]}")
    print(f"  Last ts:  {train.timestamps[-1]}")
    
    # Time stats
    stats = dataset.compute_time_statistics()
    print(f"\nTime statistics:")
    print(f"  Source: mean={stats[0]:.2f}, std={stats[1]:.2f}")
    print(f"  Dest:   mean={stats[2]:.2f}, std={stats[3]:.2f}")


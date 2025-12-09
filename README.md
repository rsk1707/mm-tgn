<p align="center">
  <h1 align="center">🕸️ MM-TGN: Multimodal Temporal Graph Network</h1>
  <p align="center">
    <em>A Multimodal Temporal Graph Network for Sequential Recommendation with Cold-Start Capabilities</em>
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch"></a>
  <a href="https://pytorch-geometric.readthedocs.io/"><img src="https://img.shields.io/badge/PyG-2.0+-3C2179.svg" alt="PyTorch Geometric"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#-key-results">Results</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-datasets">Datasets</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 📋 Overview

**MM-TGN** addresses the **cold-start problem** in sequential recommendation by integrating state-of-the-art vision-language features into a temporal graph learning framework. Unlike traditional collaborative filtering methods that rely solely on interaction patterns, MM-TGN combines:

- **Temporal Modeling**: TGN's memory mechanism captures evolving user preferences
- **Multimodal Features**: Rich semantic content from item descriptions (Qwen2-1.5B) and images (SigLIP-SO400M)
- **Fusion Strategies**: MLP, FiLM, and Gated fusion for combining text and image embeddings

This enables effective recommendations for both **warm-start** (users/items seen during training) and **cold-start** (new users/items) scenarios.

---

## 🎯 Key Results

### Performance Comparison (Overall Test Set)

| Dataset | Model | AP | AUC | NDCG@20 | Recall@20 |
|---------|-------|-----|-----|---------|-----------|
| **Amazon-Cloth** | SASRec | 0.780 | 0.791 | 0.280 | 0.580 |
| | MMGCN | 0.663 | 0.711 | 0.465 | 0.842 |
| | Vanilla TGN | 0.516 | 0.501 | 0.270 | 0.925 |
| | **MM-TGN (Ours)** | **0.944** | **0.954** | **0.616** | **0.953** |
| **Amazon-Sports** | SASRec | 0.754 | 0.742 | 0.292 | 0.599 |
| | **MM-TGN (Ours)** | **0.839** | **0.857** | **0.391** | **0.766** |
| **ML-Modern** | SASRec | 0.883 | 0.883 | 0.489 | 0.855 |
| | **MM-TGN (Ours)** | 0.857 | 0.879 | 0.412 | 0.808 |

### Cold-Start (Inductive) Performance

| Dataset | Vanilla TGN AUC | MM-TGN AUC | Improvement |
|---------|-----------------|------------|-------------|
| Amazon-Cloth | 0.515 | **0.963** | +87% |
| Amazon-Sports | 0.514 | **0.868** | +69% |
| ML-Modern | 0.503 | **0.896** | +78% |

**Key Finding**: MM-TGN achieves **1.8x improvement** over ID-only baselines on cold-start items, demonstrating that multimodal features enable effective zero-shot generalization.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/rsk1707/mm-tgn.git
cd mm-tgn

# Create conda environment
conda create -n mmtgn python=3.9 -y
conda activate mmtgn

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train MM-TGN with MLP fusion on Amazon-Cloth
python train_mmtgn.py \
    --data-dir data/processed/amazon-cloth \
    --dataset amazon-cloth \
    --node-feature-type sota \
    --mm-fusion mlp \
    --epochs 50 \
    --batch-size 200 \
    --lr 1e-4 \
    --run-name cloth_sota_mlp

# Train Vanilla TGN (ID-only baseline)
python train_mmtgn.py \
    --data-dir data/processed/amazon-cloth \
    --dataset amazon-cloth \
    --node-feature-type random \
    --run-name cloth_vanilla
```

### Evaluation

```bash
# Evaluate with ranking metrics (100 negatives, 5000 samples)
python evaluate_mmtgn.py \
    --checkpoint checkpoints/cloth_sota_mlp/best_model.pt \
    --data-dir data/processed/amazon-cloth \
    --dataset amazon-cloth \
    --n-neg-eval 100 \
    --eval-sample-size 5000 \
    --seed 42
```

### SLURM (HPC Cluster)

```bash
# Submit training job
sbatch jobs/train_cloth_sota.sh

# Submit evaluation job (after training completes)
sbatch jobs/eval_cloth_sota.sh
```

---

## 🏗️ Architecture

<p align="center">
  <img src="docs/architecture.png" alt="MM-TGN Architecture" width="800">
</p>

### Components

1. **Multimodal Encoder**
   - Text: Qwen2-1.5B (1536-dim)
   - Image: SigLIP-SO400M (1152-dim)
   - Combined: 2688-dim → 172-dim (TGN working dimension)

2. **Fusion Strategies**
   - **MLP**: `h = MLP(concat(text, image))` - symmetric combination
   - **FiLM**: `h = γ(text) ⊙ proj(image) + β(text)` - text modulates image
   - **Gated**: `h = g ⊙ proj(text) + (1-g) ⊙ proj(image)` - learned attention

3. **TGN Backbone**
   - Memory Module: Per-node memory vectors updated after each interaction
   - Message Function: MLP-based message computation
   - Message Aggregator: Last message (most recent)
   - Memory Updater: GRU cell
   - Embedding Module: 2-layer Graph Attention

4. **Training**
   - Loss: BPR (Bayesian Personalized Ranking)
   - Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
   - Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
   - Early Stopping: Patience 5 on validation AP

---

## 📊 Datasets

We evaluate on three datasets with chronological 70/15/15 train/val/test splits:

| Dataset | Interactions | Users | Items | Avg User Degree | Avg Item Degree |
|---------|--------------|-------|-------|-----------------|-----------------|
| Amazon-Sports | 217,539 | 31,111 | 12,742 | 7.0 | 17.1 |
| Amazon-Cloth | 509,723 | 68,313 | 23,694 | 7.5 | 21.5 |
| ML-Modern | 1,000,000 | 10,200 | 21,969 | 98.0 | 45.5 |

### Data Preparation

1. **Amazon Datasets**: Raw data from [Amazon Reviews](https://amazon-reviews-2023.github.io/) + multimodal features from [MM-Graph](https://github.com/westlake-repl/MM-Graph)

2. **MovieLens**: Subset of [MovieLens-32M](https://grouplens.org/datasets/movielens/) augmented with TMDB posters and plot summaries

```bash
# Generate embeddings (requires GPU)
python data/script/generate_embeddings.py \
    --dataset amazon-cloth \
    --encoder-type sota

# Format for TGN
python data/script/tgn_formatter.py \
    --dataset amazon-cloth
```

---

## 📁 Project Structure

```
mm-tgn/
├── train_mmtgn.py          # Main training script
├── evaluate_mmtgn.py       # Evaluation script
├── mmtgn.py                # MM-TGN model architecture
├── dataset.py              # Data loading & preprocessing
│
├── modules/                # Neural network modules
│   ├── embedding.py        # Fusion heads (MLP, FiLM, Gated)
│   ├── memory.py           # TGN memory module
│   ├── memory_updater.py   # GRU-based memory updater
│   ├── message_aggregator.py
│   └── message_function.py
│
├── model/                  # TGN components
│   ├── temporal_attention.py
│   ├── time_encoding.py
│   └── tgn.py
│
├── utils/                  # Utilities
│   ├── metrics.py          # Recall@K, NDCG@K, MRR, AP, AUC
│   └── utils.py            # NeighborFinder, samplers
│
├── data/
│   └── script/             # Data processing scripts
│       ├── generate_embeddings.py
│       ├── tgn_formatter.py
│       ├── export_splits.py
│       └── export_eval_samples.py
│
├── jobs/                   # SLURM job scripts
│   ├── train_*.sh          # Training jobs
│   ├── eval_*.sh           # Evaluation jobs
│   └── submit_all_*.sh     # Batch submission
│
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Configuration

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embedding-dim` | 172 | Node embedding dimension |
| `--n-layers` | 2 | Graph attention layers |
| `--n-heads` | 2 | Attention heads |
| `--n-neighbors` | 15 | Temporal neighbors to sample |
| `--memory-dim` | 172 | Memory vector dimension |
| `--message-dim` | 100 | Message dimension |
| `--dropout` | 0.1 | Dropout rate |
| `--batch-size` | 200 | Training batch size |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 50 | Maximum epochs |
| `--patience` | 5 | Early stopping patience |
| `--loss` | bpr | Loss function (bpr/bce) |

### Feature Configuration

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--node-feature-type` | `sota`, `random` | Feature source |
| `--mm-fusion` | `mlp`, `film`, `gated` | Fusion strategy |

---

## 📈 Evaluation Protocol

We follow a rigorous evaluation protocol for fair comparison:

- **Split**: Chronological 70/15/15 (train/val/test)
- **Negative Sampling**: 100 random negatives per positive
- **Evaluation Sample**: Fixed 5,000 test interactions (seed=42)
- **Metrics**: AP, AUC, MRR, Recall@K, NDCG@K, Hit@K (K ∈ {10, 20})
- **Groups**: Overall, Transductive (seen nodes), Inductive (cold-start)

---

## 🧪 Reproducing Results

### Full Ablation Study

```bash
# 1. Train all variants
sbatch jobs/train_cloth_vanilla.sh    # Vanilla TGN
sbatch jobs/train_cloth_sota.sh       # MM-TGN (MLP)
sbatch jobs/train_cloth_sota_film.sh  # MM-TGN (FiLM)
sbatch jobs/train_cloth_sota_gated.sh # MM-TGN (Gated)

# 2. Evaluate (after training completes)
sbatch jobs/eval_cloth_vanilla.sh
sbatch jobs/eval_cloth_sota.sh
sbatch jobs/eval_cloth_sota_film.sh
sbatch jobs/eval_cloth_sota_gated.sh

# 3. Results saved to:
# checkpoints/<run_name>/results_full.json
```

### Expected Training Time

| Dataset | Training | Evaluation | GPU |
|---------|----------|------------|-----|
| Amazon-Cloth | ~3-4 hours | ~2-4 hours | A100 (40GB) |
| Amazon-Sports | ~2-3 hours | ~2-3 hours | A100 (40GB) |
| ML-Modern | ~4-5 hours | ~3-4 hours | A100 (40GB) |

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{mmtgn2025,
  title={MM-TGN: A Multimodal Temporal Graph Network for Sequential Recommendation},
  author={Huseynli, Murad and Kulshrestha, Aarya and Ravichandran, Srikrishnan},
  journal={University of Michigan CSE 576 Course Project},
  year={2025}
}
```

---

## 🤝 Acknowledgments

- [TGN](https://github.com/twitter-research/tgn) - Original Temporal Graph Network implementation
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural network library
- [Qwen2](https://huggingface.co/Qwen) and [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384) - Multimodal encoders
- University of Michigan Great Lakes HPC for computational resources

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ at the University of Michigan</sub>
</p>

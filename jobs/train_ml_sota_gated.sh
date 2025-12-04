#!/bin/bash
#SBATCH --job-name=mmtgn_ml_gated
#SBATCH --account=cse576f25s001_class
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_ml_sota_gated_%j.out
#SBATCH --error=logs/train_ml_sota_gated_%j.err

# ==============================================================================
# MM-TGN Training: MovieLens SOTA + Gated Fusion
# Experiment: Gated fusion (learned attention weights for text vs image)
# ==============================================================================

echo "🚀 MM-TGN Training: ML-Modern SOTA + Gated Fusion"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmtgn

mkdir -p logs

python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --node-feature-type sota \
    --mm-fusion gated \
    --embedding-dim 172 \
    --n-layers 2 \
    --n-heads 2 \
    --n-neighbors 15 \
    --batch-size 200 \
    --epochs 50 \
    --lr 1e-4 \
    --loss bce \
    --patience 5 \
    --eval-ranking \
    --n-neg-eval 100 \
    --run-name "ml_sota_gated_$(date +%Y%m%d)"

echo ""
echo "✅ Training complete!"


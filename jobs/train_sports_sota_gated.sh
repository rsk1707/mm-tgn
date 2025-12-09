#!/bin/bash
#SBATCH --job-name=mmtgn_sports_gated
#SBATCH --account=cse576f25s001_class
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_sports_sota_gated_%j.out
#SBATCH --error=logs/train_sports_sota_gated_%j.err

# ==============================================================================
# MM-TGN Training: Amazon Sports SOTA Features (Gated Fusion)
# Experiment: Learned attention weights for modality combination
# ==============================================================================

echo "🚀 MM-TGN Training: Amazon Sports SOTA (Gated Fusion)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmtgn

mkdir -p logs

python train_mmtgn.py \
    --data-dir data/processed/amazon-sports \
    --dataset amazon-sports \
    --node-feature-type sota \
    --mm-fusion gated \
    --embedding-dim 172 \
    --n-layers 2 \
    --n-heads 2 \
    --n-neighbors 15 \
    --batch-size 200 \
    --epochs 50 \
    --lr 1e-4 \
    --loss bpr \
    --patience 5 \
    --no-eval-ranking \
    --n-neg-eval 100 \
    --run-name "sports_sota_gated_$(date +%Y%m%d)"

echo ""
echo "✅ Training complete!"


#!/bin/bash
#SBATCH --job-name=mmtgn_sota
#SBATCH --account=cse576f25s001_class
#SBATCH --partition=spgpu,gpu_mig40,gpu
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/train_sota_%j.out
#SBATCH --error=logs/train_sota_%j.err

# 1. Setup Environment
module load python-anaconda3
source ~/.bashrc
conda activate mmtgn

# Navigate to project root
cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn

# Ensure logs directory exists
mkdir -p logs

echo "🚀 Starting SOTA Training Run..."
echo "Date: $(date)"
echo "Node: $(hostname)"

# 2. Run the Main Training Command
# Added --eval-sample-size 10000 to speed up evaluation loops
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
    --loss bpr \
    --patience 5 \
    --eval-ranking \
    --n-neg-eval 100 \
    --eval-sample-size 150000 \
    --run-name "sota_full_run"

echo "✅ Training Complete."

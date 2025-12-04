#!/bin/bash
#SBATCH --job-name=mmtgn_smoke
#SBATCH --account=cse576f25s001_class
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# ==============================================================================
# MM-TGN Smoke Test
# Quick verification that everything works (~45 minutes total)
# - Training: ~25 min (1 epoch)
# - Evaluation: ~15 min (with small n_neg=20, sample=5000)
# - Transductive/Inductive eval: ~5 min
# ==============================================================================

echo "🧪 MM-TGN Smoke Test"
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
    --mm-fusion mlp \
    --epochs 1 \
    --batch-size 200 \
    --eval-ranking \
    --n-neg-eval 20 \
    --eval-sample-size 5000 \
    --run-name "smoke_test_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "✅ Smoke test complete!"


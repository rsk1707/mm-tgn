#!/bin/bash
#SBATCH --job-name=eval_cloth_film
#SBATCH --account=cse576f25s001_class
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_cloth_sota_film_%j.out
#SBATCH --error=logs/eval_cloth_sota_film_%j.err

# ==============================================================================
# MM-TGN Evaluation: Amazon Cloth - SOTA Features (FiLM Fusion)
# Full ranking evaluation on test set
# ==============================================================================

echo "📊 MM-TGN Evaluation: Amazon Cloth SOTA (FiLM)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmtgn

# Find the latest SOTA FiLM checkpoint
CHECKPOINT_DIR=$(ls -td checkpoints/cloth_sota_film_* 2>/dev/null | head -1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "❌ No SOTA FiLM checkpoint found!"
    exit 1
fi

echo "📂 Checkpoint: $CHECKPOINT_DIR"
echo ""

python evaluate_mmtgn.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --data-dir data/processed/amazon-cloth \
    --dataset amazon-cloth \
    --node-feature-type sota \
    --mm-fusion film \
    --batch-size 200 \
    --n-neighbors 15 \
    --n-neg-eval 100 \
    --eval-sample-size 5000 \
    --seed 42

echo ""
echo "✅ Evaluation complete!"


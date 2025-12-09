#!/bin/bash
#SBATCH --job-name=eval_ml_sota_gated
#SBATCH --account=cse576f25s001_class
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_ml_sota_gated_%j.out
#SBATCH --error=logs/eval_ml_sota_gated_%j.err

# ==============================================================================
# MM-TGN Evaluation: MovieLens SOTA (Gated Fusion)
# Separate evaluation job for comprehensive ranking metrics
# ==============================================================================

echo "📊 MM-TGN Evaluation: ML-Modern SOTA (Gated)"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmtgn

mkdir -p logs

# Find the latest Gated checkpoint
CHECKPOINT_DIR=$(ls -td checkpoints/ml_sota_gated_* 2>/dev/null | head -1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "❌ No SOTA (Gated) checkpoint found!"
    echo "   Run train_ml_sota_gated.sh first"
    exit 1
fi

CHECKPOINT="$CHECKPOINT_DIR/best_model.pt"
echo "📦 Using checkpoint: $CHECKPOINT"

python evaluate_mmtgn.py \
    --checkpoint "$CHECKPOINT" \
    --data-dir data/processed \
    --dataset ml-modern \
    --node-feature-type sota \
    --mm-fusion gated \
    --n-neg-eval 100 \
    --eval-sample-size 5000 \
    --batch-size 200 \
    --ranking-batch-size 100 \
    --seed 42

echo ""
echo "✅ Evaluation complete!"
echo "📁 Results saved to: $CHECKPOINT_DIR/"


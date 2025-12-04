#!/bin/bash
# ==============================================================================
# Submit All ML-Modern Experiments
# ==============================================================================
# Usage: ./jobs/submit_all_ml.sh
#
# This submits all ablation experiments for MovieLens:
# 1. Vanilla (random features) - Lower bound
# 2. SOTA + MLP fusion - Default
# 3. SOTA + FiLM fusion - Text modulates image
# 4. SOTA + Gated fusion - Learned attention
# ==============================================================================

cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn

echo "📦 Submitting ML-Modern Ablation Experiments"
echo "============================================="
echo ""

# Create logs directory
mkdir -p logs

# 1. Vanilla baseline
echo "1/4: Vanilla (random features)..."
JOB1=$(sbatch jobs/train_ml_vanilla.sh | awk '{print $4}')
echo "     Submitted: Job $JOB1"

# 2. SOTA + MLP (default)
echo "2/4: SOTA + MLP fusion..."
JOB2=$(sbatch jobs/train_ml_sota.sh | awk '{print $4}')
echo "     Submitted: Job $JOB2"

# 3. SOTA + FiLM
echo "3/4: SOTA + FiLM fusion..."
JOB3=$(sbatch jobs/train_ml_sota_film.sh | awk '{print $4}')
echo "     Submitted: Job $JOB3"

# 4. SOTA + Gated
echo "4/4: SOTA + Gated fusion..."
JOB4=$(sbatch jobs/train_ml_sota_gated.sh | awk '{print $4}')
echo "     Submitted: Job $JOB4"

echo ""
echo "============================================="
echo "✅ All jobs submitted!"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f logs/train_ml_*.out"
echo ""
echo "Results will be saved to:"
echo "  checkpoints/<run_name>/results.json"


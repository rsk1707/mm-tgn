#!/bin/bash
# ==============================================================================
# Submit All MovieLens EVALUATION Jobs
# 
# This script submits evaluation jobs for all trained models.
# Run this AFTER training jobs have completed.
#
# The evaluation jobs will:
#   1. Load the trained checkpoint (best_model.pt)
#   2. Warm up TGN memory with train+val data
#   3. Compute link prediction metrics (AP, AUC, MRR)
#   4. Compute ranking metrics (Recall@K, NDCG@K, HR@K, MRR)
#   5. Evaluate transductive and inductive splits separately
# ==============================================================================

cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn

echo "========================================"
echo "MM-TGN Evaluation Suite"
echo "========================================"
echo ""
echo "Date: $(date)"
echo ""

# Check if jobs directory exists
if [ ! -d "jobs" ]; then
    echo "❌ jobs/ directory not found!"
    exit 1
fi

# Check if checkpoints exist
echo "📂 Checking for trained checkpoints..."
echo ""

VANILLA_CKPT=$(ls -td checkpoints/ml_vanilla_* 2>/dev/null | head -1)
SOTA_CKPT=$(ls -td checkpoints/ml_sota_mlp_* 2>/dev/null | head -1)
FILM_CKPT=$(ls -td checkpoints/ml_sota_film_* 2>/dev/null | head -1)
GATED_CKPT=$(ls -td checkpoints/ml_sota_gated_* 2>/dev/null | head -1)

echo "Found checkpoints:"
[ -n "$VANILLA_CKPT" ] && echo "  ✅ Vanilla: $VANILLA_CKPT" || echo "  ❌ Vanilla: NOT FOUND"
[ -n "$SOTA_CKPT" ] && echo "  ✅ SOTA+MLP: $SOTA_CKPT" || echo "  ❌ SOTA+MLP: NOT FOUND"
[ -n "$FILM_CKPT" ] && echo "  ✅ SOTA+FiLM: $FILM_CKPT" || echo "  ❌ SOTA+FiLM: NOT FOUND"
[ -n "$GATED_CKPT" ] && echo "  ✅ SOTA+Gated: $GATED_CKPT" || echo "  ❌ SOTA+Gated: NOT FOUND"
echo ""

# ==============================================================================
# SUBMIT EVALUATION JOBS
# ==============================================================================

echo "========================================"
echo "Submitting EVALUATION jobs..."
echo "========================================"
echo ""

JOB_COUNT=0

# Vanilla evaluation
if [ -n "$VANILLA_CKPT" ] && [ -f "$VANILLA_CKPT/best_model.pt" ]; then
    echo "1/4 Submitting: Vanilla Evaluation"
    JOB_EVAL_VANILLA=$(sbatch jobs/eval_ml_vanilla.sh | awk '{print $4}')
    echo "    Job ID: $JOB_EVAL_VANILLA"
    echo "    Checkpoint: $VANILLA_CKPT"
    ((JOB_COUNT++))
else
    echo "1/4 SKIPPED: Vanilla (no checkpoint found)"
    JOB_EVAL_VANILLA=""
fi

# SOTA+MLP evaluation
if [ -n "$SOTA_CKPT" ] && [ -f "$SOTA_CKPT/best_model.pt" ]; then
    echo "2/4 Submitting: SOTA+MLP Evaluation"
    JOB_EVAL_SOTA=$(sbatch jobs/eval_ml_sota.sh | awk '{print $4}')
    echo "    Job ID: $JOB_EVAL_SOTA"
    echo "    Checkpoint: $SOTA_CKPT"
    ((JOB_COUNT++))
else
    echo "2/4 SKIPPED: SOTA+MLP (no checkpoint found)"
    JOB_EVAL_SOTA=""
fi

# SOTA+FiLM evaluation
if [ -n "$FILM_CKPT" ] && [ -f "$FILM_CKPT/best_model.pt" ]; then
    echo "3/4 Submitting: SOTA+FiLM Evaluation"
    JOB_EVAL_FILM=$(sbatch jobs/eval_ml_sota_film.sh | awk '{print $4}')
    echo "    Job ID: $JOB_EVAL_FILM"
    echo "    Checkpoint: $FILM_CKPT"
    ((JOB_COUNT++))
else
    echo "3/4 SKIPPED: SOTA+FiLM (no checkpoint found)"
    JOB_EVAL_FILM=""
fi

# SOTA+Gated evaluation
if [ -n "$GATED_CKPT" ] && [ -f "$GATED_CKPT/best_model.pt" ]; then
    echo "4/4 Submitting: SOTA+Gated Evaluation"
    JOB_EVAL_GATED=$(sbatch jobs/eval_ml_sota_gated.sh | awk '{print $4}')
    echo "    Job ID: $JOB_EVAL_GATED"
    echo "    Checkpoint: $GATED_CKPT"
    ((JOB_COUNT++))
else
    echo "4/4 SKIPPED: SOTA+Gated (no checkpoint found)"
    JOB_EVAL_GATED=""
fi

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo ""
echo "✅ Submitted $JOB_COUNT/4 evaluation jobs"
echo ""

if [ -n "$JOB_EVAL_VANILLA" ]; then
    echo "  Vanilla:     Job $JOB_EVAL_VANILLA"
fi
if [ -n "$JOB_EVAL_SOTA" ]; then
    echo "  SOTA+MLP:    Job $JOB_EVAL_SOTA"
fi
if [ -n "$JOB_EVAL_FILM" ]; then
    echo "  SOTA+FiLM:   Job $JOB_EVAL_FILM"
fi
if [ -n "$JOB_EVAL_GATED" ]; then
    echo "  SOTA+Gated:  Job $JOB_EVAL_GATED"
fi

echo ""
echo "========================================"
echo "MONITORING"
echo "========================================"
echo ""
echo "Check job status:  squeue -u \$USER"
echo "View eval logs:    tail -f logs/eval_ml_*_<jobid>.out"
echo ""
echo "Results will be saved to:"
echo "  checkpoints/<run_name>/results_linkpred.json  (fast metrics)"
echo "  checkpoints/<run_name>/results_full.json     (all metrics)"
echo ""

# Build cancel command
CANCEL_JOBS=""
[ -n "$JOB_EVAL_VANILLA" ] && CANCEL_JOBS="$CANCEL_JOBS $JOB_EVAL_VANILLA"
[ -n "$JOB_EVAL_SOTA" ] && CANCEL_JOBS="$CANCEL_JOBS $JOB_EVAL_SOTA"
[ -n "$JOB_EVAL_FILM" ] && CANCEL_JOBS="$CANCEL_JOBS $JOB_EVAL_FILM"
[ -n "$JOB_EVAL_GATED" ] && CANCEL_JOBS="$CANCEL_JOBS $JOB_EVAL_GATED"

if [ -n "$CANCEL_JOBS" ]; then
    echo "Cancel all:        scancel$CANCEL_JOBS"
fi
echo ""
echo "🌙 Good night! Results will be ready when you wake up."
echo ""


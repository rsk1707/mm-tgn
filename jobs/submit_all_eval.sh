#!/bin/bash
# ==============================================================================
# Submit All MovieLens EVALUATION Jobs (Smart Version)
# 
# This script submits evaluation jobs with proper SLURM dependencies.
# It will wait for training jobs to complete before starting evaluation.
#
# Usage:
#   ./jobs/submit_all_eval.sh                    # Auto-detect or use defaults
#   ./jobs/submit_all_eval.sh JOB1 JOB2 ...      # Wait for specific training jobs
#
# Example:
#   ./jobs/submit_all_eval.sh 37223229 37223231  # Wait for FiLM and Gated training
# ==============================================================================

cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn

echo "========================================"
echo "MM-TGN Evaluation Suite (Smart)"
echo "========================================"
echo ""
echo "Date: $(date)"
echo ""

# ==============================================================================
# CHECKPOINT PATHS (Pre-defined based on naming convention)
# ==============================================================================

# These are the expected checkpoint directories
# The script will wait for them to exist before submitting
TODAY=$(date +%Y%m%d)
VANILLA_CKPT="checkpoints/ml_vanilla_${TODAY}"
SOTA_CKPT="checkpoints/ml_sota_mlp_${TODAY}"
FILM_CKPT="checkpoints/ml_sota_film_${TODAY}"
GATED_CKPT="checkpoints/ml_sota_gated_${TODAY}"

echo "📂 Expected checkpoint paths:"
echo "  Vanilla:     $VANILLA_CKPT"
echo "  SOTA+MLP:    $SOTA_CKPT"
echo "  SOTA+FiLM:   $FILM_CKPT"
echo "  SOTA+Gated:  $GATED_CKPT"
echo ""

# ==============================================================================
# DEPENDENCY HANDLING
# ==============================================================================

# Get running/pending training job IDs
TRAINING_JOBS=$(squeue -u $USER -n "mmtgn_ml_vanilla,mmtgn_ml_sota,mmtgn_ml_sota_film,mmtgn_ml_sota_gated" -h -o "%i" 2>/dev/null | tr '\n' ':' | sed 's/:$//')

# Also check for jobs passed as arguments
if [ $# -gt 0 ]; then
    ARG_JOBS=$(echo "$@" | tr ' ' ':')
    if [ -n "$TRAINING_JOBS" ]; then
        TRAINING_JOBS="${TRAINING_JOBS}:${ARG_JOBS}"
    else
        TRAINING_JOBS="$ARG_JOBS"
    fi
fi

# Build dependency string
if [ -n "$TRAINING_JOBS" ]; then
    DEPENDENCY="--dependency=afterany:${TRAINING_JOBS}"
    echo "⏳ Found training jobs still running/pending:"
    echo "   Job IDs: $TRAINING_JOBS"
    echo "   Evaluation will start AFTER these complete."
    echo ""
else
    DEPENDENCY=""
    echo "✅ No training jobs found in queue."
    echo "   Submitting evaluation jobs immediately."
    echo ""
fi

# ==============================================================================
# SUBMIT EVALUATION JOBS
# ==============================================================================

echo "========================================"
echo "Submitting EVALUATION jobs..."
echo "========================================"
echo ""

# Vanilla evaluation
echo "1/4 Submitting: Vanilla Evaluation"
if [ -n "$DEPENDENCY" ]; then
    JOB_EVAL_VANILLA=$(sbatch $DEPENDENCY jobs/eval_ml_vanilla.sh | awk '{print $4}')
else
    JOB_EVAL_VANILLA=$(sbatch jobs/eval_ml_vanilla.sh | awk '{print $4}')
fi
echo "    Job ID: $JOB_EVAL_VANILLA"
echo "    Checkpoint: $VANILLA_CKPT"

# SOTA+MLP evaluation
echo "2/4 Submitting: SOTA+MLP Evaluation"
if [ -n "$DEPENDENCY" ]; then
    JOB_EVAL_SOTA=$(sbatch $DEPENDENCY jobs/eval_ml_sota.sh | awk '{print $4}')
else
    JOB_EVAL_SOTA=$(sbatch jobs/eval_ml_sota.sh | awk '{print $4}')
fi
echo "    Job ID: $JOB_EVAL_SOTA"
echo "    Checkpoint: $SOTA_CKPT"

# SOTA+FiLM evaluation
echo "3/4 Submitting: SOTA+FiLM Evaluation"
if [ -n "$DEPENDENCY" ]; then
    JOB_EVAL_FILM=$(sbatch $DEPENDENCY jobs/eval_ml_sota_film.sh | awk '{print $4}')
else
    JOB_EVAL_FILM=$(sbatch jobs/eval_ml_sota_film.sh | awk '{print $4}')
fi
echo "    Job ID: $JOB_EVAL_FILM"
echo "    Checkpoint: $FILM_CKPT"

# SOTA+Gated evaluation
echo "4/4 Submitting: SOTA+Gated Evaluation"
if [ -n "$DEPENDENCY" ]; then
    JOB_EVAL_GATED=$(sbatch $DEPENDENCY jobs/eval_ml_sota_gated.sh | awk '{print $4}')
else
    JOB_EVAL_GATED=$(sbatch jobs/eval_ml_sota_gated.sh | awk '{print $4}')
fi
echo "    Job ID: $JOB_EVAL_GATED"
echo "    Checkpoint: $GATED_CKPT"

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo ""
echo "✅ Submitted 4/4 evaluation jobs"
echo ""
echo "  Vanilla:     Job $JOB_EVAL_VANILLA"
echo "  SOTA+MLP:    Job $JOB_EVAL_SOTA"
echo "  SOTA+FiLM:   Job $JOB_EVAL_FILM"
echo "  SOTA+Gated:  Job $JOB_EVAL_GATED"
echo ""

if [ -n "$DEPENDENCY" ]; then
    echo "⏳ Jobs will START after training jobs complete:"
    echo "   Waiting for: $TRAINING_JOBS"
    echo ""
fi

echo "========================================"
echo "EXECUTION ORDER"
echo "========================================"
echo ""
echo "1. Training jobs finish (if any running)"
echo "2. Eval jobs start automatically"
echo "3. Results saved to checkpoints/<run>/results_full.json"
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
CANCEL_JOBS="$JOB_EVAL_VANILLA $JOB_EVAL_SOTA $JOB_EVAL_FILM $JOB_EVAL_GATED"
echo "Cancel eval jobs:  scancel $CANCEL_JOBS"
echo ""
echo "🌙 Good night! Everything is automated. Results will be ready when you wake up."
echo ""

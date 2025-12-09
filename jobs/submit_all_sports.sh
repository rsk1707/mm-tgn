#!/bin/bash
# ==============================================================================
# Smart Submission Script for Amazon Sports Experiments
# Submits all training jobs, then queues evaluation jobs with dependencies
# ==============================================================================

echo "🚀 Amazon Sports: Submitting All Experiments"
echo "=============================================="
echo "Date: $(date)"
echo ""

cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn

# Make all scripts executable
chmod +x jobs/train_sports_*.sh jobs/eval_sports_*.sh

# ==============================================================================
# PHASE 1: Submit Training Jobs
# ==============================================================================
echo "📋 PHASE 1: Submitting Training Jobs..."
echo ""

# Submit training jobs and capture job IDs
TRAIN_VANILLA_JID=$(sbatch --parsable jobs/train_sports_vanilla.sh)
echo "   ✅ Vanilla training: Job $TRAIN_VANILLA_JID"

TRAIN_SOTA_JID=$(sbatch --parsable jobs/train_sports_sota.sh)
echo "   ✅ SOTA MLP training: Job $TRAIN_SOTA_JID"

TRAIN_FILM_JID=$(sbatch --parsable jobs/train_sports_sota_film.sh)
echo "   ✅ SOTA FiLM training: Job $TRAIN_FILM_JID"

TRAIN_GATED_JID=$(sbatch --parsable jobs/train_sports_sota_gated.sh)
echo "   ✅ SOTA Gated training: Job $TRAIN_GATED_JID"

echo ""

# ==============================================================================
# PHASE 2: Submit Evaluation Jobs with Dependencies
# ==============================================================================
echo "📋 PHASE 2: Queueing Evaluation Jobs (will start after training)..."
echo ""

# Evaluation jobs depend on corresponding training jobs
EVAL_VANILLA_JID=$(sbatch --parsable --dependency=afterok:$TRAIN_VANILLA_JID jobs/eval_sports_vanilla.sh)
echo "   ✅ Vanilla eval: Job $EVAL_VANILLA_JID (depends on $TRAIN_VANILLA_JID)"

EVAL_SOTA_JID=$(sbatch --parsable --dependency=afterok:$TRAIN_SOTA_JID jobs/eval_sports_sota.sh)
echo "   ✅ SOTA MLP eval: Job $EVAL_SOTA_JID (depends on $TRAIN_SOTA_JID)"

EVAL_FILM_JID=$(sbatch --parsable --dependency=afterok:$TRAIN_FILM_JID jobs/eval_sports_sota_film.sh)
echo "   ✅ SOTA FiLM eval: Job $EVAL_FILM_JID (depends on $TRAIN_FILM_JID)"

EVAL_GATED_JID=$(sbatch --parsable --dependency=afterok:$TRAIN_GATED_JID jobs/eval_sports_sota_gated.sh)
echo "   ✅ SOTA Gated eval: Job $EVAL_GATED_JID (depends on $TRAIN_GATED_JID)"

echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================
echo "=============================================="
echo "📊 SUBMISSION SUMMARY"
echo "=============================================="
echo ""
echo "Training Jobs (8 hrs each):"
echo "  - Vanilla:     $TRAIN_VANILLA_JID"
echo "  - SOTA MLP:    $TRAIN_SOTA_JID"
echo "  - SOTA FiLM:   $TRAIN_FILM_JID"
echo "  - SOTA Gated:  $TRAIN_GATED_JID"
echo ""
echo "Evaluation Jobs (queued, 8 hrs each):"
echo "  - Vanilla:     $EVAL_VANILLA_JID → after $TRAIN_VANILLA_JID"
echo "  - SOTA MLP:    $EVAL_SOTA_JID → after $TRAIN_SOTA_JID"
echo "  - SOTA FiLM:   $EVAL_FILM_JID → after $TRAIN_FILM_JID"
echo "  - SOTA Gated:  $EVAL_GATED_JID → after $TRAIN_GATED_JID"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Results in: checkpoints/sports_*/results_full.json"
echo ""
echo "✅ All jobs submitted!"


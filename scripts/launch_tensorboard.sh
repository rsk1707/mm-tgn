#!/bin/bash
# =============================================================================
# TensorBoard Launch Script for Great Lakes HPC
# =============================================================================
#
# USAGE:
#   Option A (Interactive): ./scripts/launch_tensorboard.sh
#   Option B (Background):  ./scripts/launch_tensorboard.sh &
#
# THEN, on your LOCAL machine, set up SSH tunnel:
#   ssh -L 6006:<COMPUTE_NODE>:6006 huseynli@greatlakes.arc-ts.umich.edu
#
# Finally, open in browser: http://localhost:6006
#
# =============================================================================

# Configuration
PORT=${TENSORBOARD_PORT:-6006}
LOGDIR="${TENSORBOARD_LOGDIR:-/scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn/runs}"

# Get current hostname (compute node name)
HOSTNAME=$(hostname)

echo "=============================================================="
echo "🚀 TensorBoard Launcher for MM-TGN"
echo "=============================================================="
echo ""
echo "📍 Compute Node: $HOSTNAME"
echo "📁 Log Directory: $LOGDIR"
echo "🔌 Port: $PORT"
echo ""

# Check if log directory exists
if [ ! -d "$LOGDIR" ]; then
    echo "❌ Error: Log directory not found: $LOGDIR"
    echo "   Run training first to generate logs."
    exit 1
fi

# Count event files
NUM_RUNS=$(find "$LOGDIR" -name "events.out.tfevents.*" 2>/dev/null | wc -l)
echo "📊 Found $NUM_RUNS TensorBoard event file(s)"
echo ""

# Activate conda environment
echo "🔧 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate mmtgn

# Check if tensorboard is available
if ! command -v tensorboard &> /dev/null; then
    echo "❌ TensorBoard not found. Installing..."
    pip install tensorboard
fi

echo ""
echo "=============================================================="
echo "📋 TO VIEW TENSORBOARD:"
echo "=============================================================="
echo ""
echo "1. On your LOCAL machine, open a new terminal and run:"
echo ""
echo "   ssh -L ${PORT}:${HOSTNAME}:${PORT} huseynli@greatlakes.arc-ts.umich.edu"
echo ""
echo "2. Then open in your browser:"
echo ""
echo "   http://localhost:${PORT}"
echo ""
echo "=============================================================="
echo ""
echo "Starting TensorBoard... (Press Ctrl+C to stop)"
echo ""

# Launch TensorBoard (force IPv4 only - fixes "unsupported address family ::" error)
tensorboard --logdir="$LOGDIR" --port="$PORT" --host=0.0.0.0 --reload_interval=30


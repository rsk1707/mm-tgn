#!/bin/bash
#SBATCH --job-name=gen_embeddings
#SBATCH --account=cse576f25s001_class
#SBATCH --partition=gpu_mig40,gpu,spgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/embeddings_%j.out
#SBATCH --error=logs/embeddings_%j.err

# ================================================================
# Generate SOTA Embeddings for Amazon Datasets
# Uses: Qwen2-1.5B (text) + SigLIP (image) = 2688 dim
# ================================================================

echo "🚀 Starting Embedding Generation..."
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# Setup
cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmtgn

# Create output directories
mkdir -p data/datasets/amazon-sports/features/sota
mkdir -p data/datasets/amazon-cloth/features/sota
mkdir -p logs

echo "========================================"
echo "1/2: Amazon-Sports (12,742 items)"
echo "========================================"

python data/script/generate_embeddings.py \
    --csv-path data/datasets/amazon-sports/sports-5core-metadata.csv \
    --image-dir data/datasets/amazon-sports/sports-5core-images \
    --output-dir data/datasets/amazon-sports/features/sota \
    --dataset-name amazon-sports \
    --id-col asin \
    --text-col description \
    --text-model efficient \
    --image-model siglip \
    --batch-size 32

if [ $? -eq 0 ]; then
    echo "✅ Amazon-Sports embeddings complete!"
else
    echo "❌ Amazon-Sports failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "2/2: Amazon-Cloth (23,694 items)"
echo "========================================"

python data/script/generate_embeddings.py \
    --csv-path data/datasets/amazon-cloth/cloth-5core-metadata.csv \
    --image-dir data/datasets/amazon-cloth/cloth-5core-images \
    --output-dir data/datasets/amazon-cloth/features/sota \
    --dataset-name amazon-cloth \
    --id-col asin \
    --text-col description \
    --text-model efficient \
    --image-model siglip \
    --batch-size 32

if [ $? -eq 0 ]; then
    echo "✅ Amazon-Cloth embeddings complete!"
else
    echo "❌ Amazon-Cloth failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "All Done!"
echo "========================================"
echo "Output files:"
ls -la data/datasets/amazon-sports/features/sota/
ls -la data/datasets/amazon-cloth/features/sota/


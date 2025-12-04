#!/bin/bash
#SBATCH --job-name=mmtgn_vanilla
#SBATCH --account=cse576f25s001_class
#SBATCH --partition=spgpu,gpu_mig40,gpu
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/train_vanilla_%j.out

module load python-anaconda3
source ~/.bashrc
conda activate mmtgn
cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn

# NOTICE: --node-feature-type random and --fusion-mode none
python train_mmtgn.py \
    --data-dir data/processed \
    --dataset ml-modern \
    --node-feature-type random \
    --fusion-mode none \
    --embedding-dim 172 \
    --n-layers 2 \
    --batch-size 200 \
    --epochs 50 \
    --loss bpr \
    --eval-ranking \
    --n-neg-eval 100 \
    --run-name "vanilla_baseline"

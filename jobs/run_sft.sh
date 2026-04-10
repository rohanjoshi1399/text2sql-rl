#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

# SFT warm-up — must reach >=70% EX on dev before GRPO
# Usage: sbatch jobs/run_sft.sh

set -euo pipefail
mkdir -p logs checkpoints

module load anaconda3/2024.06
source activate texttosql

export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=${HF_TOKEN:-""}
export WANDB_PROJECT=sql-grpo

cd ~/text2sql-rl

echo "=== SFT Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -m src.training.sft --config configs/sft.yaml

echo ""
echo "=== Evaluating SFT checkpoint ==="
python -m src.eval.run_eval \
    --model checkpoints/sft/best \
    --mode model \
    --split dev \
    --output results

echo "=== Done ==="

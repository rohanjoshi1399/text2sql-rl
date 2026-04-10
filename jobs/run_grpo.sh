#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err

# GRPO training from SFT checkpoint
# Usage: sbatch jobs/run_grpo.sh
# Prerequisites: SFT checkpoint at checkpoints/sft/best with >=70% EX

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

# Verify SFT checkpoint exists
if [ ! -d "checkpoints/sft/best" ]; then
    echo "ERROR: SFT checkpoint not found at checkpoints/sft/best"
    echo "Run SFT first: sbatch jobs/run_sft.sh"
    exit 1
fi

echo "=== GRPO Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -m src.training.grpo \
    --config configs/grpo.yaml \
    --warm-start checkpoints/sft/best

# Only evaluate if training completed (best model saved)
if [ -d "checkpoints/grpo/best" ]; then
    echo ""
    echo "=== Evaluating GRPO checkpoint ==="
    python -m src.eval.run_eval \
        --model checkpoints/grpo/best \
        --mode model \
        --split dev \
        --output results
fi

echo "=== Done ==="

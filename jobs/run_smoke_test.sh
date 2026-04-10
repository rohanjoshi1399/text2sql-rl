#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke_test_%j.out
#SBATCH --error=logs/smoke_test_%j.err

# Quick smoke test — verifies GRPOTrainer + LoRA + rewards work end-to-end
# Usage: sbatch jobs/run_smoke_test.sh

set -euo pipefail
mkdir -p logs

module load anaconda3/2024.06
source activate texttosql

export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=${HF_TOKEN:-""}

cd ~/text2sql-rl

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -m src.training.grpo --smoke-test

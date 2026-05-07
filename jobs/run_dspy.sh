#!/bin/bash
#SBATCH --job-name=dspy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80GB
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=logs/dspy_%j.out
#SBATCH --error=logs/dspy_%j.err

# DSPy MIPROv2 prompt optimization
# Task model: Llama 3.1 8B (SQL generation)
# Optimizer model: Gemma 4 31B (proposes better prompts)
# Usage: sbatch jobs/run_dspy.sh

set -euo pipefail
mkdir -p logs checkpoints

module load anaconda3/2024.06
source activate texttosql

export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before submitting}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

cd ~/text2sql-rl

echo "=== DSPy MIPROv2 Optimization ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python -m src.prompts.optimize \
    --task-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --opt-model google/gemma-4-31b-it \
    --optimizer miprov2 \
    --trainset-size 200 \
    --output checkpoints/dspy_llama_gemma4

echo "=== Done ==="

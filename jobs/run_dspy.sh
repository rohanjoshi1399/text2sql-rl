#!/bin/bash
#SBATCH --job-name=dspy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/dspy_%j.out
#SBATCH --error=logs/dspy_%j.err

# DSPy MIPROv2 prompt optimization
# Launches a vLLM server for local Llama 8B inference
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

# Uses local transformers for inference — no vLLM needed
python -m src.prompts.optimize \
    --task-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --opt-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --optimizer miprov2 \
    --trainset-size 200 \
    --output checkpoints/dspy_optimized

echo "=== Done ==="

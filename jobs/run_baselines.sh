#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

# Run zero-shot and 5-shot baselines on Spider dev
# Usage: sbatch jobs/run_baselines.sh

set -euo pipefail
mkdir -p logs results

module load anaconda3/2024.06
source activate texttosql

export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=${HF_TOKEN:-""}

cd ~/text2sql-rl

echo "=== Zero-Shot Baseline ==="
python -m src.eval.run_eval \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --mode zero-shot \
    --split dev \
    --output results

echo ""
echo "=== 5-Shot Baseline ==="
python -m src.eval.run_eval \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --mode few-shot \
    -k 5 \
    --split dev \
    --output results

echo ""
echo "=== Error Analysis ==="
python -m src.eval.error_analysis \
    --predictions results/eval_dev_zero-shot.json

echo "=== Done ==="

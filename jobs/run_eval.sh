#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# One-off evaluation of an arbitrary checkpoint on Spider dev.
#
# Usage:
#   sbatch jobs/run_eval.sh <model_path> [mode] [output_name]
#
# Examples:
#   sbatch jobs/run_eval.sh checkpoints/grpo/checkpoint-150
#   sbatch jobs/run_eval.sh checkpoints/grpo/checkpoint-150 model grpo_ckpt150
#
# mode defaults to "model" (LoRA checkpoint or full weights).
# output_name controls the results JSON filename suffix
#   (results/eval_dev_<output_name>.json). Defaults to "model".

set -euo pipefail
mkdir -p logs results

MODEL_PATH="${1:?usage: sbatch jobs/run_eval.sh <model_path> [mode] [output_name]}"
MODE="${2:-model}"
OUT_NAME="${3:-model}"

module load anaconda3/2024.06
source activate texttosql

export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=${HF_TOKEN:-""}

cd ~/text2sql-rl

if [ ! -d "$MODEL_PATH" ] && [[ "$MODEL_PATH" != *"/"* ]]; then
    echo "ERROR: model path not found: $MODEL_PATH"
    exit 1
fi

echo "=== Evaluation ==="
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Model: $MODEL_PATH"
echo "Mode:  $MODE"
echo "Out:   results/eval_dev_${OUT_NAME}.json"
echo ""

python -m src.eval.run_eval \
    --model "$MODEL_PATH" \
    --mode "$MODE" \
    --split dev \
    --name "$OUT_NAME" \
    --output results

echo ""
echo "=== Done ==="

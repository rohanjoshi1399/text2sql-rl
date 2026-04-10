#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# Preprocessing — no GPU needed, just runs gold SQL against SQLite
# Usage: sbatch jobs/run_preprocess.sh

set -euo pipefail
mkdir -p logs

module load anaconda3/2024.06
source activate texttosql

export HF_HOME=/scratch/$USER/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd ~/text2sql-rl

echo "=== Preprocessing Spider training data ==="
python -m src.data.preprocess \
    --data-dir data/spider_data/spider_data \
    --timeout 5 \
    --output data/spider_train_filtered

echo "=== Done ==="

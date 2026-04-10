#!/bin/bash
# Setup script for NEU Explorer cluster
# Run this ONCE on an interactive GPU node:
#   srun --partition=gpu-interactive --gres=gpu:a100:1 --mem=64GB --time=01:00:00 --pty /bin/bash
#   bash scripts/setup_cluster.sh

set -euo pipefail

echo "=== Setting up text2sql-rl environment on Explorer ==="

# Load anaconda
module load anaconda3/2024.06

# Create conda environment (skip if already exists)
if ! conda env list | grep -q "texttosql"; then
    conda create -n texttosql python=3.11 -y
    echo "Created conda environment 'texttosql'"
fi
source activate texttosql

# Setup HF cache on scratch (home quota is limited)
export HF_HOME=/scratch/$USER/hf_cache
mkdir -p $HF_HOME
echo "HF_HOME set to $HF_HOME"

# Install PyTorch with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn (requires CUDA headers, may take a few minutes)
pip install flash-attn --no-build-isolation || echo "WARNING: flash-attn install failed. Training will work but slower."

# Install project dependencies
pip install -r requirements.txt

# Verify
echo ""
echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
python -c "from trl import GRPOTrainer, GRPOConfig; print('TRL GRPOTrainer: OK')"
python -c "from peft import LoraConfig; print('PEFT LoRA: OK')"
python -c "import sqlparse; print('sqlparse: OK')"
python -c "import dspy; print('DSPy: OK')"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Set your HF_TOKEN: export HF_TOKEN=<your-token>"
echo "  2. Run preprocessing: python -m src.data.preprocess"
echo "  3. Smoke test: python -m src.training.grpo --smoke-test"
echo ""
echo "For batch jobs, use: sbatch jobs/<job_script>.sh"

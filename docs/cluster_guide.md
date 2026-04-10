# NEU Explorer Cluster Guide

## 1. Connecting

### Prerequisites
- NEU VPN active (GlobalProtect → `vpn.northeastern.edu`)
- SSH client (built into Windows Terminal, PowerShell, or VS Code)

### SSH into the login node
```bash
ssh <your-username>@login.explorer.northeastern.edu
```

The login node is for file management and job submission only — **never run training or inference here** (no GPUs).

### Upload files from local machine
```bash
# Single file
scp "C:\path\to\file.py" <your-username>@login.explorer.northeastern.edu:~/text2sql-rl/path/to/file.py

# Entire directory
scp -r "C:\path\to\dir" <your-username>@login.explorer.northeastern.edu:~/text2sql-rl/dir

# Large datasets — compress first to avoid dropped connections
tar -czf data.tar.gz data/
scp data.tar.gz <your-username>@login.explorer.northeastern.edu:~/text2sql-rl/
# Then on cluster: tar -xzf data.tar.gz && rm data.tar.gz
```

> **Note:** `rsync` is not available on Windows natively. Use `scp` or install Git Bash (which includes rsync).

---

## 2. Environment Setup (One-Time)

### Request an interactive GPU node
```bash
srun --partition=gpu --gres=gpu:1 --mem=32GB --time=00:30:00 --pty /bin/bash
```
Wait for the prompt to change from `login-XX` to a compute node hostname (e.g., `d1028`).

### Create conda environment
```bash
module load anaconda3/2024.06
conda create -n texttosql python=3.11 -y
source activate texttosql

pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.1 trl==0.14.0 datasets peft accelerate bitsandbytes dspy sqlparse wandb nltk sentencepiece protobuf pytest
```

### Set HuggingFace cache on scratch (home has limited quota)
```bash
export HF_HOME=/scratch/$USER/hf_cache
mkdir -p $HF_HOME
```

### Login to HuggingFace (for gated models like Llama)
```bash
huggingface-cli login --token <your-hf-token>
```

> **Important:** The login saves to the login node's home directory. If compute nodes don't share the same home filesystem, hardcode `HF_TOKEN` in your job scripts instead.

### Verify installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from trl import GRPOTrainer; print('TRL OK')"
python -m pytest tests/test_rewards.py -v
```

### Exit the interactive node when done
```bash
exit
```

---

## 3. Checking GPU Availability

### See all GPU nodes and their state
```bash
sinfo -p gpu --format="%P %G %T %D"
```

Output columns:
- **GRES**: GPU type and count (e.g., `gpu:a100:4` = 4x A100)
- **STATE**: `idle` (all GPUs free), `mixed` (some GPUs free), `allocated` (full), `drained` (offline)

### Available GPU types on Explorer

| GPU | VRAM | Compute | bfloat16 | Best for |
|-----|------|---------|----------|----------|
| V100-PCIE | 16 GB | 7.0 | No | Too small for Llama 8B |
| V100-SXM2 | 32 GB | 7.0 | No | SFT (4-bit), DSPy, baselines |
| T4 | 16 GB | 7.5 | No | Too small for Llama 8B |
| A100 | 40/80 GB | 8.0 | Yes | GRPO, SFT (full precision) |
| H200 | 80+ GB | 9.0 | Yes | GRPO, anything |

### Check your queued/running jobs
```bash
squeue --me          # basic status
squeue --me -l       # detailed (shows REASON for pending)
squeue --me --start  # estimated start time
```

### Common REASON values for pending jobs
- `Priority` — other jobs are ahead in the queue
- `Resources` — no matching GPUs available
- `QOSMaxJobsPerUser` — you've hit the concurrent job limit

### Tips for faster scheduling
- Request **less time** (shorter jobs get priority): `--time=04:00:00` instead of `08:00:00`
- Request **less memory**: `--mem=32GB` instead of `64GB`
- Request **any GPU** (`--gres=gpu:1`) instead of a specific type
- Target **idle nodes**: if `sinfo` shows idle V100s, request `--gres=gpu:v100-sxm2:1`

---

## 4. Running Jobs

### Interactive mode (`srun`)
Your terminal stays connected. If SSH drops, the job dies.
```bash
srun --partition=gpu --gres=gpu:v100-sxm2:1 --mem=32GB --time=02:00:00 --pty /bin/bash

# Then activate environment and run:
module load anaconda3/2024.06
source activate texttosql
export HF_HOME=/scratch/$USER/hf_cache
cd ~/text2sql-rl
python -m src.training.grpo --smoke-test
```

### Batch mode (`sbatch`)
Job runs in the background — safe to disconnect.
```bash
sbatch jobs/run_sft.sh
# Returns: Submitted batch job 1234567
```

### Monitoring batch jobs
```bash
squeue --me                          # check status
tail -f logs/sft_1234567.out         # watch live output
cat logs/sft_1234567.err             # check errors
scancel 1234567                      # cancel a job
sacct --jobs=1234567 --format=JobID,State,ExitCode,MaxRSS,Elapsed  # post-mortem
```

---

## 5. Key Environment Variables

Always set these in job scripts or before running:
```bash
export HF_HOME=/scratch/$USER/hf_cache
export HF_TOKEN="<your-hf-token>"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
```

---

## 6. Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `CUDA not available` | On login node (no GPU) | Submit as batch job or use `srun` for interactive GPU |
| `CUDA out of memory` | Model too large for GPU | Use 4-bit quantization (auto-detected for V100) or request A100 |
| `HF 401 Unauthorized` | Token not set on compute node | Hardcode `HF_TOKEN` in job script |
| `Job stuck in PENDING` | No GPUs free or low priority | Request any GPU (`--gres=gpu:1`), reduce time/memory |
| `torch version mismatch` | pip installed wrong version | `pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121` |
| `Connection dropped during scp` | Large files over unstable SSH | Compress first with `tar -czf`, upload single archive |
| `vLLM breaks torch` | vLLM pulls incompatible PyTorch | Don't install vLLM. Reinstall torch if broken. |

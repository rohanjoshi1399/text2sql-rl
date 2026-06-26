# SELECT * FROM EXPERIENCE

Training Text-to-SQL agents with GRPO reinforcement learning and DSPy prompt optimization on Spider.

## Overview

This project investigates how much reinforcement learning (GRPO) improves SQL generation accuracy over zero-shot and supervised fine-tuning baselines, and whether prompt optimization (DSPy) provides complementary gains. We train Llama 3.1 8B Instruct on the Spider benchmark using execution-based rewards from live SQLite databases.

**Research questions:**
1. How much does GRPO improve execution accuracy over zero-shot and SFT baselines?
2. Can DSPy prompt optimization approach GRPO's gains without weight updates?
3. Do DSPy + GRPO gains stack, or are they redundant?

## Project Structure

```
text2sql-rl/
├── configs/                  # Training and evaluation configs
│   ├── sft.yaml
│   ├── grpo.yaml
│   └── eval.yaml
├── src/
│   ├── data/                 # Data loading and preprocessing
│   │   ├── spider_loader.py  # Load Spider JSONs + DDL from SQLite
│   │   └── preprocess.py     # Filter empty-result queries (8659 → 7040)
│   ├── rewards/              # Reward functions for GRPO
│   │   ├── execution.py      # SQL execution against in-memory SQLite DBs
│   │   ├── syntax.py         # sqlparse validation + SELECT keyword check
│   │   ├── schema_coverage.py # Table/column F1 scoring
│   │   └── composite.py      # Two-phase combined rewards with pre-multiplied weights
│   ├── training/             # SFT and GRPO training scripts
│   │   ├── sft.py            # LoRA SFT with auto 4-bit on V100
│   │   ├── grpo.py           # GRPO from SFT checkpoint
│   │   └── utils.py          # Checkpoint resume, Slurm signal handling
│   ├── prompts/              # DSPy signatures and optimization
│   │   ├── signatures.py     # Text2SQL and Text2SQLWithReasoning
│   │   └── optimize.py       # LocalTransformersLM + MIPROv2 optimization
│   └── eval/                 # Evaluation pipeline
│       ├── run_eval.py       # Zero-shot, few-shot, and model evaluation
│       └── error_analysis.py # Failure categorization (10 error types)
├── jobs/                     # Slurm job scripts
│   ├── run_sft.sh
│   ├── run_grpo.sh
│   ├── run_dspy.sh
│   ├── run_baselines.sh
│   ├── run_preprocess.sh
│   └── run_smoke_test.sh
├── scripts/
│   ├── generate_charts.py    # Result visualization (7 charts)
├── tests/
│   └── test_rewards.py       # 73 unit tests for reward functions
├── docs/
│   ├── architecture.md       # System architecture and component docs
│   ├── execution_guide.md    # Step-by-step cluster execution guide
│   └── cluster_guide.md      # NEU Explorer cluster setup
├── WHY.md                    # Design decision log with rationale
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.11
- Linux (for `signal.SIGALRM` based SQL timeout; Windows uses threading fallback)
- CUDA 12.1

### Cluster Installation (NEU Explorer)

```bash
module load anaconda3/2024.06
conda create -n texttosql python=3.11 -y
source activate texttosql

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Set model cache to scratch (home quota is limited)
export HF_HOME=/scratch/$USER/hf_cache
```

### Pinned Dependencies

These exact versions are tested and validated on the cluster. Do not upgrade without re-running smoke tests.

| Package | Version | Why pinned |
|---------|---------|------------|
| torch | 2.5.1+cu121 | Matches cluster CUDA 12.1 |
| transformers | 4.47.1 | Compatible with TRL 0.14.0 |
| trl | 0.14.0 | GRPO smoke test validated; newer versions break API |
| dspy | 3.1.3 | MIPROv2 stable |

**Do NOT install vllm** -- it pulls incompatible PyTorch wheels and breaks the environment.

### Environment Variables

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=sql-grpo
export HF_TOKEN=<your-huggingface-token>     # For Llama 3.1 8B gated access
```

### Verify Installation

```bash
# Smoke test: confirm GRPO + LoRA compatibility
python -m src.training.grpo --smoke-test

# Smoke test: confirm reward functions
python -m pytest tests/test_rewards.py -v
```

## Usage

### 1. Preprocess Spider Data

```bash
# Filter empty-result and broken gold queries (8659 → 7040)
python -m src.data.preprocess
```

### 2. Run Baselines

```bash
# Zero-shot and few-shot evaluation on Spider dev (1034 examples)
python -m src.eval.run_eval --mode zero-shot --split dev --output results
python -m src.eval.run_eval --mode few-shot --split dev --output results
```

### 3. SFT Warm-up

```bash
# On cluster via Slurm (auto-resumes from latest checkpoint)
sbatch jobs/run_sft.sh

# Must reach >=70% EX on Spider dev before proceeding to GRPO
```

### 4. GRPO Training

```bash
# Requires SFT checkpoint at checkpoints/sft/best
sbatch jobs/run_grpo.sh
```

### 5. DSPy Prompt Optimization

```bash
# MIPROv2 with separate task/optimizer models
sbatch jobs/run_dspy.sh
```

### 6. Evaluation

```bash
# Evaluate any checkpoint
python -m src.eval.run_eval --model checkpoints/sft/best --mode model --split dev --output results
```

## Key Configuration

### SFT (`configs/sft.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lora_rank` | 32 | All 7 linear modules (q/k/v/o + gate/up/down) |
| `learning_rate` | 2e-4 | Cosine schedule, warmup 5% |
| `batch_size` | 2 x 8 grad_accum = 16 effective | |
| `epochs` | 3 | Best model around epoch 2 (overfitting after) |
| `optimizer` | paged_adamw_8bit | Memory-efficient for LoRA |

### GRPO (`configs/grpo.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `beta` | 0.001 | KL penalty -- 0.1 over-constrains policy |
| `num_generations` | 6 | Group size G; 4-8 is the sweet spot |
| `lora_rank` | 32 | Same LoRA config as SFT |
| `max_completion_length` | 512 | SQL queries are short |
| `max_grad_norm` | 0.1 | Aggressive clipping for RL stability |
| `temperature` | 0.8 | Prevents entropy collapse |
| `learning_rate` | 5e-6 | Much lower than SFT for RL stability |

## Results

All evaluation on Spider dev set (1,034 examples). Difficulty breakdown uses Spider's official `eval_hardness` classifier (248 easy / 446 medium / 174 hard / 166 extra-hard).

### Overall Execution Accuracy

| Condition | EX | EM | Correct/Total |
|-----------|----|----|---------------|
| Zero-shot | 67.8% | 17.0% | 701/1034 |
| 5-shot | 66.3% | 23.3% | 686/1034 |
| DSPy MIPROv2 (self-opt) | 58.6%* | — | — |
| SFT + LoRA (epoch 3) | 71.1% | 31.4% | 735/1034 |
| **GRPO + LoRA (1 epoch)** | **72.1%** | **33.3%** | **745/1034** |

*DSPy 58.6% is MIPROv2's best-trial internal metric, not a full 1,034-example eval.*

### By Difficulty (Spider Official Hardness)

| Condition | Easy (248) | Medium (446) | Hard (174) | Extra (166) |
|-----------|-----------|-------------|-----------|------------|
| Zero-shot | 85.5% | 66.6% | 62.6% | 50.0% |
| 5-shot | 87.5% | 65.3% | 59.2% | 45.2% |
| SFT (epoch 3) | 87.9% | 72.0% | 56.3% | **59.0%** |
| **GRPO (1 epoch)** | **88.7%** | **74.7%** | **60.9%** | 51.8% |

### Key Findings

- **GRPO beats SFT (+1.0 pp overall)**: 49 queries GRPO gets right that SFT misses; 39 regressions. Net +10.
- **Hard-query recovery**: SFT regressed on hard queries (−6.3 pp vs zero-shot). GRPO recovers +4.6 pp, reaching 60.9% — within 1.7 pp of zero-shot.
- **Extra-hard tradeoff**: GRPO drifts away from SFT's memorized UNION/INTERSECT/EXCEPT templates (−7.2 pp, net −12 examples). Hard gains come at set-operator cost.
- **SFT hard-query regression**: SFT memorizes common multi-join template shapes, losing compositional flexibility for novel combinations. GRPO's execution reward is indifferent to structural form, which restores this generalization.
- **5-shot hurts Hard queries**: −3.4 pp on Hard vs zero-shot. Demos from different schemas prime incorrect join patterns.
- **DSPy self-optimization underperforms**: A model cannot effectively optimize its own prompts at 8B scale (58.6% vs 67.8% zero-shot).

*Test set evaluation held in reserve — to be run once across all conditions.*

## Team

- Rohan Joshi
- Haridhar Pulivarthy
- Krushna Sharma

## References

- [Spider](https://yale-lily.github.io/spider) -- Cross-domain Text-to-SQL benchmark
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300) -- Group Relative Policy Optimization
- [DSPy](https://dspy.ai) -- Programmatic prompt optimization
- [Dr. GRPO](https://arxiv.org/abs/2503.20783) -- Length-bias correction for GRPO

## License

MIT

# SELECT * FROM EXPERIENCE

Training Text-to-SQL agents with GRPO reinforcement learning and DSPy prompt optimization on Spider.

## Overview

This project investigates how much reinforcement learning (GRPO) improves SQL generation accuracy over zero-shot and supervised fine-tuning baselines, and whether prompt optimization (DSPy) provides complementary gains. We train Llama 3.1 8B Instruct on the Spider benchmark using execution-based rewards from live SQLite databases.

**Research questions:**
1. How much does GRPO improve execution accuracy over zero-shot and SFT baselines?
2. Can DSPy prompt optimization approach GRPO's gains without weight updates?
3. Do DSPy + GRPO gains stack, or are they redundant?

**Expected trajectory:** Zero-shot (~61%) → Few-shot/DSPy (~71%) → SFT (~80%) → GRPO (~85%+)

## Project Structure

```
sql-grpo/
├── configs/                  # Training and evaluation configs
│   ├── sft.yaml
│   ├── grpo.yaml
│   └── eval.yaml
├── src/
│   ├── data/                 # Data loading and preprocessing
│   │   ├── spider_loader.py
│   │   └── preprocess.py     # Filter empty-result queries, timeout violations
│   ├── rewards/              # Reward functions
│   │   ├── execution.py      # SQL execution against SQLite DBs
│   │   ├── syntax.py         # sqlparse validation
│   │   ├── schema_coverage.py
│   │   └── composite.py      # Combined reward with configurable weights
│   ├── training/             # SFT and GRPO training scripts
│   │   ├── sft.py
│   │   └── grpo.py
│   ├── prompts/              # DSPy signatures and optimization
│   │   ├── signatures.py
│   │   └── optimize.py
│   └── eval/                 # Evaluation pipeline
│       ├── run_eval.py
│       └── error_analysis.py
├── scripts/
│   ├── setup_cluster.sh      # RC cluster environment setup
│   ├── download_spider.sh    # Download dataset + SQLite DBs
│   └── run_baselines.sh      # Zero-shot and few-shot evaluation
├── notebooks/
│   └── analysis.ipynb        # Results visualization and comparison tables
├── tests/
│   └── test_rewards.py       # Unit tests for reward functions
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- Linux (Ubuntu 22.04+)
- CUDA 12.8 with an A100 80GB (or 40GB with Unsloth)

### Installation

```bash
git clone https://github.com/<your-org>/sql-grpo.git
cd sql-grpo

# Create environment
python -m venv .venv && source .venv/activate

# Install dependencies
pip install -r requirements.txt

# Download Spider dataset and SQLite databases
bash scripts/download_spider.sh

# Set environment variables (copy and edit)
cp .env.example .env
```

### Environment Variables

```bash
# .env
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TOKENIZERS_PARALLELISM=false
WANDB_PROJECT=sql-grpo
HF_TOKEN=<your-huggingface-token>     # For Llama 3.1 8B gated access
```

### Verify Installation

```bash
# Smoke test: confirm GRPO + LoRA + vLLM compatibility
python -m src.training.grpo --smoke-test

# Smoke test: confirm reward function on labeled examples
python -m pytest tests/test_rewards.py -v
```

## Usage

### 1. Preprocess Spider Data

```bash
# Remove empty-result queries and timeout violations
python -m src.data.preprocess --data-dir data/spider --timeout 5
```

### 2. Run Baselines

```bash
# Zero-shot and few-shot evaluation
python -m src.eval.run_eval --model meta-llama/Meta-Llama-3.1-8B-Instruct --mode zero-shot
python -m src.eval.run_eval --model meta-llama/Meta-Llama-3.1-8B-Instruct --mode few-shot --k 5
```

### 3. SFT Warm-up

```bash
python -m src.training.sft --config configs/sft.yaml
# Must reach ≥70% EX on Spider dev before proceeding to GRPO
```

### 4. GRPO Training

```bash
python -m src.training.grpo --config configs/grpo.yaml --warm-start checkpoints/sft-best/
```

### 5. DSPy Prompt Optimization

```bash
python -m src.prompts.optimize --optimizer miprov2 --task-model llama-8b --opt-model gemini-flash
```

### 6. Full Evaluation

```bash
# Evaluate any checkpoint against Spider dev (or test for final numbers only)
python -m src.eval.run_eval --model checkpoints/grpo-best/ --split dev --output results/
```

## Key Configuration

The GRPO config (`configs/grpo.yaml`) reflects validated hyperparameters from the literature:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `beta` | 0.001 | KL penalty — 0.1 is too high |
| `num_generations` | 6 | Group size G; 4–8 is the sweet spot |
| `loss_type` | `dr_grpo` | Removes length bias |
| `lora_rank` | 32 | 16 is adequate; 32 slightly better for SQL |
| `max_completion_len` | 512 | SQL queries are short |
| `max_grad_norm` | 0.1 | Aggressive clipping for stability |
| `generation_temp` | 0.8 | Prevents entropy collapse |

## Results

| Condition | Spider Dev EX | Spider Test EX |
|-----------|---------------|----------------|
| Zero-shot | — | — |
| 5-shot | — | — |
| DSPy (MIPROv2) | — | — |
| SFT (LoRA) | — | — |
| GRPO (ours) | — | — |
| DSPy + GRPO | — | — |

*Table will be populated as experiments complete.*

## Team

- Haridhar Pulivarthy
- Krushna Sharma
- Rohan Joshi

## References

- [Agent Lightning](https://github.com/microsoft/agent-lightning) — RL orchestration framework with Spider recipe
- [Spider](https://yale-lily.github.io/spider) — Cross-domain Text-to-SQL benchmark
- [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization
- [DSPy](https://dspy.ai) — Programmatic prompt optimization
- [SkyRL-SQL](https://arxiv.org/abs/2507.07849) — Multi-turn RL for Text-to-SQL
- [Dr. GRPO](https://arxiv.org/abs/2503.20783) — Length-bias correction for GRPO

## License

MIT

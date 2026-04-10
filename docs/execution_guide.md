# Project Execution Guide

## Pipeline Overview

```
[Local]  Preprocess data ──→ Upload to cluster
                                    │
                                    ▼
[Cluster] Smoke test ──→ Baselines ──→ SFT ──→ GRPO
                              │                   │
                              │         [Parallel] │
                              │                   ▼
                              └──────── DSPy ──→ DSPy + GRPO combined
                                                   │
                                                   ▼
                                            Final evaluation
```

---

## Step-by-Step Execution Order

### Step 0: Preprocess (Local, no GPU)
**Already done.** Filters ~1,600 empty-result queries from training data.
```bash
python -m src.data.preprocess --data-dir data/spider_data/spider_data --timeout 5 --output data/spider_train_filtered
```
- **Time:** ~5 minutes
- **GPU:** None
- **Output:** `data/spider_train_filtered/` (7,040 examples from 8,659 original)

---

### Step 1: Smoke Test (Cluster, any GPU)
Verifies GRPOTrainer + LoRA + reward functions work end-to-end. **Run this first on any GPU.**
```bash
sbatch jobs/run_smoke_test.sh
# Or interactively:
srun --partition=gpu --gres=gpu:1 --mem=32GB --time=00:30:00 --pty /bin/bash
python -m src.training.grpo --smoke-test
```

| Detail | Value |
|--------|-------|
| **Time** | ~5-10 minutes |
| **GPU** | Any (V100 32GB+, auto-quantizes if needed) |
| **Output** | `checkpoints/smoke_test/` (throwaway) |
| **Success criteria** | Prints "Smoke test PASSED" |

---

### Step 2: Baselines (Cluster, any 32GB+ GPU)
**Already done.** Zero-shot and 5-shot evaluation on Spider dev.
```bash
sbatch jobs/run_baselines.sh
```

| Detail | Value |
|--------|-------|
| **Time** | ~2-4 hours |
| **GPU** | V100-SXM2 32GB or better |
| **Output** | `results/eval_dev_zero-shot.json`, `results/eval_dev_few-shot.json` |
| **Results obtained** | Zero-shot: **67.8% EX**, 5-shot: **66.3% EX** |

---

### Step 3: SFT Training (Cluster, V100 32GB+ or A100)
Supervised fine-tuning warm-up. Must reach ≥70% EX before GRPO.
```bash
sbatch jobs/run_sft.sh
```

| Detail | Value |
|--------|-------|
| **Time** | ~4-8 hours (3 epochs on 7,040 examples) |
| **GPU** | V100-SXM2 32GB (4-bit QLoRA) or A100 (bf16 LoRA) |
| **Memory** | ~20GB on V100, ~24GB on A100 |
| **Output** | `checkpoints/sft/best/` |
| **Success criteria** | ≥70% EX on Spider dev (auto-evaluated at end) |
| **Config** | `configs/sft.yaml` |

**Key settings:**
- Batch size 2 × grad_accum 8 = effective batch 16
- Learning rate: 2e-4 with cosine schedule
- LoRA rank 32 on all attention + MLP layers
- Auto 4-bit quantization on V100

**If OOM:** Reduce `max_seq_length` from 1024 to 768 in `configs/sft.yaml`.

---

### Step 4: GRPO Training (Cluster, A100 40GB+ required)
The main RL training. **Requires SFT checkpoint from Step 3.**
```bash
sbatch jobs/run_grpo.sh
```

| Detail | Value |
|--------|-------|
| **Time** | ~8-24 hours (2 epochs, depends on convergence) |
| **GPU** | **A100 40GB minimum** (6 generations per prompt) |
| **Memory** | ~35-40GB |
| **Output** | `checkpoints/grpo/best/` |
| **Prerequisite** | `checkpoints/sft/best/` must exist with ≥70% EX |
| **Config** | `configs/grpo.yaml` |

**Key settings:**
- `num_generations: 6` (group size G)
- `beta: 0.001` (KL penalty)
- `temperature: 0.8` (prevents entropy collapse)
- `max_completion_length: 512`
- Phase 2 composite rewards (execution + syntax + schema + format)

**Cannot run on V100** — 6 generations × Llama 8B doesn't fit in 32GB even with 4-bit.

**If OOM on A100 40GB:** Reduce `num_generations` from 6 to 4 in `configs/grpo.yaml`.

**Monitor for reward hacking:**
- Watch W&B (or log output) for reward curve
- If reward plateaus at 0, SFT checkpoint was too weak
- If reward spikes to max quickly, check for entropy collapse

---

### Step 5: DSPy Optimization (Cluster, V100 32GB+ or A100)
Prompt optimization — independent of SFT/GRPO. **Can run in parallel with Step 3.**
```bash
sbatch jobs/run_dspy.sh
```

| Detail | Value |
|--------|-------|
| **Time** | ~2-4 hours |
| **GPU** | V100-SXM2 32GB (float16) or A100 (bfloat16) |
| **Memory** | ~18GB (inference only) |
| **Output** | `checkpoints/dspy_optimized.json` |
| **Config** | CLI args in `jobs/run_dspy.sh` |

**Alternative: Run locally with APIs** (no GPU, after API quotas reset):
```bash
set GEMINI_API_KEY=<key>
python -m src.prompts.optimize --task-model gemini/gemini-2.0-flash --opt-model gemini/gemini-2.0-flash --optimizer bootstrap --trainset-size 50 --output checkpoints/dspy_optimized.json
```

---

### Step 6: Final Evaluation (Cluster, any 32GB+ GPU)
Evaluate all conditions on Spider dev (and test for final numbers).
```bash
# SFT checkpoint
python -m src.eval.run_eval --model checkpoints/sft/best --mode model --split dev --output results

# GRPO checkpoint
python -m src.eval.run_eval --model checkpoints/grpo/best --mode model --split dev --output results

# Error analysis
python -m src.eval.error_analysis --predictions results/eval_dev_model.json
```

| Detail | Value |
|--------|-------|
| **Time** | ~1-2 hours per condition |
| **GPU** | V100-SXM2 32GB or better |

**Test set (final numbers only — run once):**
```bash
python -m src.eval.run_eval --model checkpoints/grpo/best --mode model --split test --output results
```

---

## Job-to-GPU Mapping

| Job | Min GPU | Recommended GPU | Can use V100 32GB? | SBATCH flag |
|-----|---------|----------------|---------------------|-------------|
| Smoke test | Any | V100-SXM2 | Yes (4-bit) | `--gres=gpu:1` |
| Baselines | V100 32GB | V100-SXM2 | Yes | `--gres=gpu:v100-sxm2:1` |
| SFT | V100 32GB | A100 | Yes (4-bit QLoRA) | `--gres=gpu:v100-sxm2:1` |
| GRPO | **A100 40GB** | A100 80GB / H200 | **No** | `--gres=gpu:a100:1` |
| DSPy | V100 32GB | V100-SXM2 | Yes (float16) | `--gres=gpu:v100-sxm2:1` |
| Evaluation | V100 32GB | V100-SXM2 | Yes | `--gres=gpu:v100-sxm2:1` |

---

## Time Estimates

| Step | Wall time | GPU hours | Can parallelize? |
|------|-----------|-----------|-----------------|
| Preprocess | 5 min | 0 (CPU) | N/A |
| Smoke test | 10 min | 0.2 | N/A |
| Baselines | 3 hours | 3 | N/A |
| SFT | 6 hours | 6 | — |
| GRPO | 16 hours | 16 | After SFT |
| DSPy | 3 hours | 3 | Parallel with SFT/GRPO |
| Final eval | 2 hours × N conditions | 2-10 | After all training |
| **Total** | **~30 hours** | **~28-38** | |

---

## Expected Results

| Condition | Spider Dev EX (expected) | Spider Dev EX (actual) |
|-----------|--------------------------|------------------------|
| Zero-shot | ~61% | **67.8%** |
| 5-shot | ~71% | **66.3%** |
| DSPy (MIPROv2) | ~71% | pending |
| SFT (LoRA) | ~80% | pending |
| GRPO | ~85%+ | pending |
| DSPy + GRPO | ~87%+ | pending |

---

## Dependency Graph

```
preprocess ─────────────────────────────────┐
                                            │
smoke_test ─── (verifies pipeline works) ───┤
                                            │
baselines ──── (zero-shot, few-shot) ───────┤
                                            │
sft ─────────── (must reach ≥70% EX) ──────┤──→ grpo ──→ eval_grpo
                                            │
dspy ────────── (parallel, independent) ────┤──→ eval_dspy
                                            │
                                            └──→ eval_combined (DSPy prompt + GRPO weights)
                                                      │
                                                      └──→ error_analysis
                                                      └──→ final test set eval
```

---

## Uploading Code Changes

After making local changes, upload the modified files:
```bash
# Upload specific files
scp "C:\path\to\text2sql-rl\src\training\sft.py" <your-username>@login.explorer.northeastern.edu:~/text2sql-rl/src/training/sft.py

# Upload entire src directory
scp -r "C:\path\to\text2sql-rl\src" <your-username>@login.explorer.northeastern.edu:~/text2sql-rl/src
```

---

## Package Versions (Tested Working)

```
torch==2.5.1+cu121
transformers==4.47.1
trl==0.14.0
peft (latest compatible)
datasets (latest)
dspy==3.1.3
sqlparse==0.5.5
bitsandbytes (latest)
accelerate (latest)
```

**Do NOT install vLLM** — it pulls incompatible PyTorch versions and breaks the environment.

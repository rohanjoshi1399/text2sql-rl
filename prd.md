# PRD: SELECT * FROM Experience
### Training Text-to-SQL Agents with Prompt and Weight Optimization
**Version:** 1.0 | **Team:** 3 members (A, B, C) | **Timeline:** 12 weeks | **Compute:** University RC Cluster (A100 80GB)

---

## 1. Project Overview

### 1.1 Problem Statement
LLMs fail consistently on complex SQL generation — multi-table joins, nested subqueries, and ambiguous schema references. Prompt engineering is brittle across schemas; supervised fine-tuning requires labeled data and doesn't teach error recovery. This project investigates whether reinforcement learning can systematically close the gap, and specifically, how much of that improvement requires updating model weights versus optimizing the prompt alone.

### 1.2 Core Research Questions
| ID | Question | Method |
|----|----------|--------|
| RQ1 | How much does GRPO improve EX over zero-shot and SFT baselines? | GRPO fine-tuning |
| RQ2 | Can DSPy prompt optimization approach GRPO's gains without weight updates? | DSPy MIPROv2 |
| RQ3 | Do DSPy + GRPO gains stack, or are they redundant? | Combined evaluation |

### 1.3 Expected Performance Trajectory (Llama 3.1 8B on Spider)
```
Zero-shot (~61%) → Few-shot/DSPy (~71%) → SFT (~80%) → GRPO (~85%+)
```

---

## 2. Technical Architecture

### 2.1 Model
- **Primary:** `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Fallback:** `meta-llama/Llama-3.2-3B-Instruct` (if VRAM constrained)
- **Why Instruct variant:** Mandatory for GRPO — base models rarely produce a useful learning signal on SQL without chat formatting

### 2.2 Agent Design
```
NL Question + DB Schema
        ↓
   LitAgent (Agent Lightning)
        ↓ (up to 5 turns, single-turn first)
   ┌─────────────────────────────────┐
   │  Tool 1: schema_lookup(db)      │  → relevant tables/columns
   │  Tool 2: execute_sql(query)     │  → result set or error message
   │  Tool 3: self_correct(error)    │  → refined query
   └─────────────────────────────────┘
        ↓
   Final SQL Output
```
**Implementation note:** Start with single-turn generation (no tool loop). Add multi-turn only in Week 8 if ahead of schedule. SkyRL-SQL showed single-turn RL generalizes well to multi-turn at test time.

### 2.3 MDP Formulation
- **State:** NL question + DB schema + working memory (tool call history, partial query, error messages)
- **Actions:** `schema_lookup(db)` | `execute_sql(query)` | `self_correct(error)` | `emit_final_sql()`
- **Turn cap:** 5 tool calls per episode (prevents unbounded rollouts)
- **Termination:** `emit_final_sql()` called, or turn cap reached

### 2.4 Reward Function

#### Phase 1 (Weeks 3–6): Simple reward
```python
R = execution_accuracy  # binary 0/1
  + 0.2 * syntax_valid  # sqlparse parses without error
```

#### Phase 2 (Weeks 7–9): Full composite reward
```python
R = 1.0 * execution_accuracy          # result set matches gold — dominant signal
  + 0.2 * execution_success           # query runs without error
  + 0.2 * syntax_valid                # sqlparse validates structure
  + 0.2 * schema_coverage_f1          # F1 over tables+columns vs. gold
  + 0.1 * format_compliance           # output follows expected format tags
```
**Important:** Process rewards default to 0 if SQL fails to execute. Schema coverage F1 is computed over table+column names only — not structural matching, which is too noisy.

#### Reward computation
Both generated and ground-truth SQL are executed against Spider's SQLite databases at training time. Result sets are compared directly — no pre-stored outputs. A hard 5-second timeout per query prevents runaway joins from stalling the training loop.

### 2.5 GRPO Configuration
```python
model               = "meta-llama/Meta-Llama-3.1-8B-Instruct"
lora_rank           = 32          # 16 is adequate; 32 is marginally better for SQL
lora_alpha          = 32
lora_dropout        = 0.0
target_modules      = ["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"]
learning_rate       = 5e-6        # increase to 1e-5 for LoRA only
beta                = 0.001       # NOT 0.1 — would over-constrain policy
num_generations     = 6           # G=4–8; 6 balances diversity vs. VRAM
max_completion_len  = 512         # SQL queries are short
loss_type           = "dr_grpo"   # removes length bias (Dr. GRPO paper)
epsilon             = 0.2         # clip ratio
max_grad_norm       = 0.1         # aggressive clipping improves stability
optimizer           = "paged_adamw_8bit"
generation_temp     = 0.8
```

### 2.6 Prompt Optimization (DSPy, replacing APO)
- **Framework:** DSPy with MIPROv2 optimizer
- **Signature:** `question, schema -> sql_query`
- **Optimizer model:** External API (Gemini Flash or GPT-4o-mini) as the optimizer; Llama 3.1 8B as the task model
- **Why not APO:** APO (ProTeGi) was validated only on classification tasks, never SQL. DSPy has a working Text-to-SQL implementation at 7B scale.
- **Convergence criterion:** Stop when EX improvement across two consecutive iterations < 1%

---

## 3. Dataset

### 3.1 Spider
- **Source:** `taoyds/spider` (HuggingFace) or GitHub
- **Size:** 10,181 questions, 5,693 unique SQL queries, 200 databases, 138 domains
- **Splits:** Train (8,659 / 146 DBs) | Dev (1,034 / 20 DBs) | Test (2,147 / 40 DBs)
- **Difficulty:** Easy / Medium / Hard / Extra Hard
- **Evaluation:** `test-suite-sql-eval` (taoyds/test-suite-sql-eval) — executes against multiple DB instances for robust accuracy
- **Databases:** SQLite files ship with dataset — used for live execution at training time
- **Preprocessing required:** Remove ~1,700 samples returning empty result sets (following Arctic-Text2SQL-R1). Remove any queries exceeding 5-second execution. This is non-negotiable for reward signal quality.

### 3.2 Evaluation Protocol
- **Dev set:** Used for all intermediate evaluation during training
- **Test set:** Held out — touched only for final reported numbers
- **Never:** Train and test on the same split

### 3.3 Stretch Goal: BIRD
- 12,751 question-SQL pairs, 95 databases (33.4 GB)
- Llama 3.1 8B achieves ~32% zero-shot — large improvement headroom
- Only pursue if ahead of schedule after Week 9

---

## 4. Full Component Stack

### 4.1 Core Libraries
| Component | Library | Version |
|-----------|---------|---------|
| Orchestration | `agentlightning` | v0.3.0 |
| RL training | `trl` (GRPOTrainer) | ≥0.23.1 |
| Inference | `vllm` | 0.10.2 |
| LoRA | `peft` | Latest stable |
| LLM loading | `transformers` | Latest stable |
| Memory optimization | `unsloth` | Optional; recommended for VRAM headroom |
| Quantization | `bitsandbytes` | 4-bit if VRAM constrained |
| Flash attention | `flash-attn` | ≥2.6.3 |
| Deep learning | `torch` | 2.8.0, CUDA 12.8 |
| Distributed | `accelerate` | For multi-GPU if available |

### 4.2 SQL & Evaluation
| Component | Library | Notes |
|-----------|---------|-------|
| SQL execution | `sqlite3` | Python built-in |
| SQL parsing | `sqlparse` | Syntax validation reward |
| Timeout control | `signal` module | Hard 5s per query |
| Safety | Read-only DB connections | Copy DB to memory per episode |
| Evaluation | `test-suite-sql-eval` | Official Spider evaluation |
| NL toolkit | `nltk` | Required by eval script |

### 4.3 Prompt Optimization & Tracking
| Component | Library | Notes |
|-----------|---------|-------|
| Prompt optimization | `dspy` | MIPROv2 or BootstrapFewShot |
| Experiment tracking | `wandb` | TRL native integration via `report_to="wandb"` |
| Data loading | `datasets` | HuggingFace |

### 4.4 Environment Variables
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # prevents VRAM spikes
TOKENIZERS_PARALLELISM=false                        # avoids deadlocks in DataLoader
```

---

## 5. Known Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Agent Lightning CUDA incompatibility with RC cluster | Medium | High | Dedicate Week 1 to smoke testing; fallback: TRL + Unsloth |
| TRL GRPO + vLLM + LoRA compatibility bug (issue #2698) | Low-Medium | High | Verify with TRL ≥0.23.1; Unsloth as workaround |
| Reward hacking — agent learns trivial SQL | Medium | High | Execution accuracy as dominant signal; filter empty-result training queries |
| Reward sparsity — all samples score 0 early | High | High | SFT warm-up to ≥70% EX before GRPO (non-negotiable) |
| Entropy/mode collapse | Medium | Medium | Monitor entropy via W&B; Dr. GRPO loss; temperature ≥0.6 |
| GRPO cold-start failure (no SFT) | High if skipped | High | Never skip SFT phase |
| VRAM OOM on A100 80GB | Low | High | Reduce `num_generations` to 4; enable gradient checkpointing; use Unsloth |
| DSPy optimizer model (8B) too weak for MIPROv2 | Medium | Medium | Use external API (Gemini Flash) as optimizer LLM |
| Spider too easy (not a risk for 8B) | Low | Low | 61% zero-shot → 85%+ RL trajectory provides clear story |

---

## 6. Task Breakdown

### Phase 0: Setup (Week 1–2)
#### Person A — Infrastructure
- [ ] **T-A01:** Request RC cluster access; verify A100 80GB availability and CUDA version
- [ ] **T-A02:** Install Agent Lightning v0.3.0 with full dependency chain (`torch 2.8.0`, `vllm 0.10.2`, `verl 0.5.0`, `flash-attn`)
- [ ] **T-A03:** If Agent Lightning fails on cluster CUDA, install TRL ≥0.23.1 + Unsloth fallback stack
- [ ] **T-A04:** Download and cache Llama 3.1 8B Instruct weights on cluster filesystem
- [ ] **T-A05:** Verify GRPO + LoRA + vLLM compatibility (run minimal smoke test from TRL docs)
- [ ] **T-A06:** Set up W&B project and shared experiment dashboard

#### Person B — Data & SQL Environment
- [ ] **T-B01:** Download Spider dataset (train/dev/test splits + all SQLite DB files)
- [ ] **T-B02:** Build SQL execution sandbox: read-only SQLite connections, 5-second timeout enforcement, in-memory DB copy per episode
- [ ] **T-B03:** Pre-filter training data: remove empty-result queries (~1,700 samples); remove queries exceeding 5s execution time
- [ ] **T-B04:** Install and verify `test-suite-sql-eval`; confirm it reproduces published Spider EX scores on a gold sample
- [ ] **T-B05:** Write `compute_reward()` — Phase 1 simple version: `execution_accuracy + 0.2 * syntax_valid`
- [ ] **T-B06:** Unit test reward function on 20 hand-labeled examples (correct SQL, wrong SQL, non-executing SQL, empty result SQL)

#### Person C — Baseline Evaluation
- [ ] **T-C01:** Write Spider data loader: batch NL questions with schema context formatted for Llama chat template
- [ ] **T-C02:** Run zero-shot evaluation on Spider dev set (full 1,034 examples); record EX by difficulty tier
- [ ] **T-C03:** Run 3-shot and 5-shot evaluation; record EX by difficulty tier
- [ ] **T-C04:** Document baseline results in W&B; create comparison table template
- [ ] **T-C05:** Identify 50 failure examples from zero-shot eval stratified by difficulty — used to guide SFT data selection

**Phase 0 exit criteria:** Agent Lightning (or TRL fallback) installs cleanly, SQL sandbox produces correct rewards on test cases, zero-shot baseline EX is recorded.

---

### Phase 1: SFT Warm-up (Weeks 3–4)
#### Person A — Training
- [ ] **T-A07:** Configure SFT training: LoRA rank 32, alpha 32, all linear layers, `paged_adamw_8bit`, lr=2e-4
- [ ] **T-A08:** Run SFT on Spider train split (filtered); monitor train loss and dev EX per epoch
- [ ] **T-A09:** Evaluate SFT checkpoint on Spider dev — must reach ≥70% EX before proceeding to GRPO
- [ ] **T-A10:** If <70% EX: increase training epochs or add schema-linking supervision signals; do not proceed to GRPO until threshold met
- [ ] **T-A11:** Save and version best SFT checkpoint; push to cluster storage

#### Person B — Reward Engineering
- [ ] **T-B07:** Build schema coverage F1 scorer using `sqlglot` or `sqlparse`; extract table+column names from both generated and gold SQL; compute precision, recall, F1
- [ ] **T-B08:** Implement full composite reward function (Phase 2 version) with all five components; validate on labeled examples
- [ ] **T-B09:** Build reward diagnostics: log per-component reward breakdown per episode to W&B; identify which component fires most/least

#### Person C — DSPy Setup
- [ ] **T-C06:** Install DSPy; configure Llama 3.1 8B as the task model (via SGLang or Ollama)
- [ ] **T-C07:** Configure external API (Gemini Flash or GPT-4o-mini) as the DSPy optimizer model
- [ ] **T-C08:** Define DSPy signature: `question: str, schema: str -> sql_query: str`
- [ ] **T-C09:** Run BootstrapFewShot optimizer on Spider train subset (200 examples); record baseline EX improvement
- [ ] **T-C10:** Document DSPy-optimized prompt vs. hand-crafted baseline in comparison table

**Phase 1 exit criteria:** SFT model achieves ≥70% EX on Spider dev; DSPy baseline running; composite reward function validated.

---

### Phase 2: GRPO Training (Weeks 5–7)
#### Person A — GRPO Training Loop
- [ ] **T-A12:** Configure GRPOTrainer starting from SFT checkpoint (non-negotiable warm start); set `beta=0.001`, `num_generations=6`, `loss_type="dr_grpo"`, `max_grad_norm=0.1`
- [ ] **T-A13:** Run first GRPO experiment with simple reward (Phase 1 version); monitor reward curve, entropy curve, and EX on dev set every 100 steps
- [ ] **T-A14:** Verify no reward hacking: manually inspect top-reward SQL outputs; confirm they are semantically correct, not trivially constructed
- [ ] **T-A15:** If entropy collapses: increase temperature, reduce learning rate, or switch to `beta=0.01`
- [ ] **T-A16:** Switch to composite reward (Phase 2 version) once simple-reward training is stable; compare dev EX curves

#### Person B — Reward Monitoring & Ablations
- [ ] **T-B10:** Monitor reward component breakdown per training step in W&B; flag if syntax reward saturates before EX improves (indicates reward hacking)
- [ ] **T-B11:** Run ablation: GRPO with execution-only reward vs. composite reward; record both dev EX curves
- [ ] **T-B12:** Sample 100 failed episodes at step 500; categorize failure modes: wrong table, wrong column, wrong join condition, wrong aggregation, non-executing SQL
- [ ] **T-B13:** Implement batch composition logic: ensure each training batch contains a mix of Easy/Medium/Hard examples to prevent gradient collapse when all easy examples solve correctly

#### Person C — DSPy MIPROv2 + Evaluation
- [ ] **T-C11:** Run DSPy MIPROv2 optimizer (more powerful than BootstrapFewShot); stop when EX improvement < 1% across two iterations
- [ ] **T-C12:** Record DSPy EX by difficulty tier on Spider dev; enter in comparison table
- [ ] **T-C13:** Evaluate SFT model (no RL) on full Spider dev; record by difficulty tier as intermediate baseline
- [ ] **T-C14:** Set up automated evaluation script: given any checkpoint, compute full Spider dev EX breakdown in <10 minutes

**Phase 2 exit criteria:** GRPO model shows improvement over SFT baseline on dev set; DSPy optimization complete; no reward hacking detected.

---

### Phase 3: Combined Evaluation & Ablations (Weeks 8–9)
#### Person A — Combined Run
- [ ] **T-A17:** Apply best DSPy-optimized prompt to GRPO-tuned model; evaluate on Spider dev
- [ ] **T-A18:** Compare: zero-shot vs. few-shot vs. DSPy vs. SFT vs. GRPO vs. DSPy+GRPO; populate full comparison table
- [ ] **T-A19:** Run hyperparameter sensitivity: vary `num_generations` (4, 6, 8); vary `beta` (0.0, 0.001, 0.01); record effect on dev EX
- [ ] **T-A20 (stretch):** Add multi-turn tool loop (schema_lookup + execute_sql + self_correct); compare single-turn vs. multi-turn GRPO

#### Person B — BIRD Stretch Goal
- [ ] **T-B14 (stretch):** Download BIRD dataset and SQLite databases
- [ ] **T-B15 (stretch):** Run zero-shot and best GRPO model on BIRD dev; record EX
- [ ] **T-B16:** Build difficulty-tier breakdown chart: EX improvement from zero-shot to GRPO, faceted by Easy/Medium/Hard/Extra Hard

#### Person C — Error Analysis
- [ ] **T-C15:** Sample 200 failures from GRPO model; manually categorize into: wrong table selection, wrong column, wrong join type, wrong aggregation/grouping, correct SQL wrong result (data issue), non-executing SQL
- [ ] **T-C16:** For each failure category: measure how much each method (zero-shot / SFT / DSPy / GRPO) reduces its frequency
- [ ] **T-C17:** Identify queries where DSPy helps but GRPO doesn't (and vice versa) — key evidence for RQ3

**Phase 3 exit criteria:** All five conditions evaluated on Spider dev; error taxonomy complete; RQ3 answered.

---

### Phase 4: Final Evaluation & Report (Weeks 10–12)
#### Person A — Final Numbers
- [ ] **T-A21:** Run best model (DSPy + GRPO) on Spider **test set** — first and only time test set is touched
- [ ] **T-A22:** Record final EX by difficulty tier on test set; compare to published baselines (SFT at 79.9%, MARS-SQL at 89.75%)
- [ ] **T-A23:** Clean up codebase: remove debug code, add README, document reproducibility steps, pin dependency versions
- [ ] **T-A24:** Push final code and model checkpoints to shared repo

#### Person B — Figures & Visualizations
- [ ] **T-B17:** Training curve plot: EX vs. training step for GRPO (with and without SFT warm-up if ablated)
- [ ] **T-B18:** Comparison bar chart: all 5 conditions, faceted by difficulty tier
- [ ] **T-B19:** Reward component breakdown over training: stacked area chart showing how each reward component evolves
- [ ] **T-B20:** Failure taxonomy chart: stacked bar showing proportion of each failure type per method

#### Person C — Report Writing
- [ ] **T-C18:** Write final report (8–10 pages, NeurIPS format): Introduction, Related Work, Method, Experiments, Analysis, Conclusion
- [ ] **T-C19:** Related Work: cite Reasoning-SQL, Arctic-Text2SQL-R1, SkyRL-SQL, PaVeRL-SQL, CRL — the 5 most relevant 2025 RL+Text-to-SQL papers
- [ ] **T-C20:** Write analysis section addressing all three RQs with quantitative evidence
- [ ] **T-C21:** Limitations section: single benchmark, 8B model only, Spider not fully representative of production SQL complexity
- [ ] **T-C22:** Prepare final presentation slides from updated slide content

**Phase 4 exit criteria:** Final report submitted; code repo reproducible; test set numbers match or exceed SFT baseline by ≥5 EX points.

---

## 7. Comparison Table (Target)

| Condition | Spider Dev EX | Spider Test EX | Notes |
|-----------|-------------|--------------|-------|
| Zero-shot | ~61% | — | Baseline |
| 5-shot | ~70% | — | Baseline |
| DSPy (MIPROv2) | ~71–73% | — | RQ2 upper bound |
| SFT (LoRA) | ~80% | ~79% | Published: Databricks |
| GRPO (ours) | ~85%+ | ~85%+ | Primary result |
| DSPy + GRPO (ours) | ~85–87%? | — | RQ3 answer |

---

## 8. What to Cut If Behind Schedule

Cut in this order — never cut the bottom three:

1. Multi-turn tool loop (T-A20) — single-turn generalizes well
2. BIRD stretch goal (T-B14/15) — Spider is sufficient
3. Hyperparameter sensitivity sweeps (T-A19) — use defaults
4. Composite reward → use execution + syntax only throughout
5. DSPy MIPROv2 → use BootstrapFewShot only
6. Error analysis depth → reduce to top 2 failure categories

**Never cut:** SFT warm-up, basic GRPO training, Spider dev/test evaluation, W&B tracking.

---

## 9. Deliverables

| Deliverable | Owner | Due |
|-------------|-------|-----|
| Working training pipeline (Agent Lightning or TRL) | A | Week 4 |
| Reproducible code repo with README | A | Week 11 |
| Spider evaluation script | B+C | Week 2 |
| Reward function (simple + composite) | B | Week 4 |
| DSPy-optimized prompt | C | Week 7 |
| Full comparison table (all 5 conditions) | All | Week 9 |
| Error taxonomy | C | Week 9 |
| Final report (8–10 pages, NeurIPS format) | C (lead) | Week 12 |
| Presentation slides | C | Week 12 |

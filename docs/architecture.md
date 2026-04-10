# Architecture: Text-to-SQL with RL

## The Big Picture

This project answers one question: **can reinforcement learning teach an LLM to write better SQL than supervised fine-tuning alone, and how does that compare to just optimizing the prompt?**

```
                         ┌─────────────────────────┐
                         │   Natural Language       │
                         │   "How many employees    │
                         │    are in Engineering?"  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │         Llama 3.1 8B Instruct    │
                    │    (with schema context in prompt)│
                    └────────────┬────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────┐
                    │  SELECT COUNT(*) FROM employees  │
                    │  WHERE department = 'Engineering' │
                    └────────────┬────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────────┐
                    │      Execute against SQLite       │
                    │      Compare with gold result     │
                    │      → Reward signal (0 or 1)     │
                    └─────────────────────────────────┘
```

We compare three strategies for improving the model's SQL accuracy:

| Strategy | What changes | What stays fixed |
|----------|-------------|-----------------|
| **SFT** (Supervised Fine-Tuning) | Model weights (LoRA) | Prompt template |
| **GRPO** (RL) | Model weights (LoRA) | Prompt template |
| **DSPy** (Prompt Optimization) | Prompt instructions + examples | Model weights |

---

## Data Flow

```
Spider Dataset (JSON + SQLite databases)
        │
        ▼
┌──────────────────┐     ┌───────────────────┐
│  spider_loader.py │────→│   preprocess.py    │
│                  │     │                   │
│ • Load JSONs     │     │ • Execute gold SQL │
│ • Extract DDL    │     │ • Remove empty     │
│   from SQLite    │     │   results (1,616)  │
│ • Format chat    │     │ • Remove timeouts  │
│   template       │     │ • Save filtered    │
│ • Build HF       │     │   dataset (7,040)  │
│   Dataset        │     └───────┬───────────┘
└──────────────────┘             │
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
   ┌─────────────┐      ┌──────────────┐      ┌──────────────┐
   │   sft.py     │      │   grpo.py     │      │  optimize.py  │
   │   SFT        │─────→│   GRPO        │      │  DSPy         │
   │   warm-up    │      │   training    │      │  MIPROv2      │
   └──────┬──────┘      └──────┬───────┘      └──────┬───────┘
          │                     │                      │
          └─────────────────────┼──────────────────────┘
                                │
                                ▼
                       ┌──────────────┐
                       │  run_eval.py  │
                       │  Evaluation   │
                       └──────┬───────┘
                              │
                              ▼
                     ┌────────────────┐
                     │error_analysis.py│
                     │ Failure modes   │
                     └────────────────┘
```

---

## Component Deep-Dive

### 1. Spider Loader (`src/data/spider_loader.py`)

**What it does:** Loads the Spider benchmark dataset and formats it for Llama 3.1 Instruct.

**Why it's essential:** The model needs to see the database schema to write correct SQL. How you present the schema directly affects accuracy. We extract CREATE TABLE DDL from the actual SQLite files (not from `tables.json`) because:
- DDL includes exact column names, types, PRIMARY KEY, and FOREIGN KEY constraints
- It matches what the model would see at inference time on a new database
- No information is lost in translation

**Key design decisions:**
- **System prompt is minimal:** `"You are a SQL expert..."` — deliberately short so DSPy can replace/augment it later
- **User message format:** `### Database Schema: ... ### Question: ... ### SQL:` — the `### SQL:` suffix primes the model to emit SQL immediately
- **Schema truncation:** Capped at 3,500 chars to avoid exceeding context window on large databases (some Spider DBs have 15+ tables)

**Output format (HuggingFace Dataset columns):**
```
prompt    → list[dict]  (chat messages for tokenizer.apply_chat_template)
gold_sql  → str         (ground truth SQL)
db_id     → str         (database identifier)
db_path   → str         (absolute path to .sqlite file)
question  → str         (natural language question)
```

---

### 2. Preprocessing (`src/data/preprocess.py`)

**What it does:** Filters training examples where the gold SQL returns empty results or times out.

**Why it's essential:** Empty-result queries are poison for RL training. If the gold SQL returns no rows, then *any* model output that also returns no rows gets reward 1.0 — including completely wrong queries like `SELECT 1 WHERE 0`. This creates false positive rewards that corrupt the training signal.

**What gets filtered:**
| Reason | Count | Why it's removed |
|--------|-------|-----------------|
| Empty result | 1,616 | Any wrong query that returns empty would get full reward |
| Timeout (>5s) | 0 | Would block training; Spider queries are fast |
| SQL error | 3 | Gold SQL itself is broken in these examples |

**Before:** 8,659 examples → **After:** 7,040 examples

---

### 3. Reward Functions (`src/rewards/`)

The reward functions are the **most critical component** — they define what "good SQL" means to the RL agent. Without correct rewards, GRPO learns nothing or learns the wrong thing.

#### 3a. Execution Reward (`execution.py`)

**What it does:** Runs both the model's SQL and the gold SQL against the same SQLite database, compares result sets.

**Why it's essential:** This is the *only* reward that measures actual correctness. String-matching SQL would fail on semantically equivalent queries:
```sql
-- Gold
SELECT name FROM employees WHERE dept = 'Eng'
-- Model (also correct, but string-different)
SELECT e.name FROM employees AS e WHERE e.dept = 'Eng'
```
Both return identical rows. Execution accuracy catches this.

**Key implementation details:**
- **In-memory DB copy:** `disk_conn.backup(mem_conn)` — prevents the model from corrupting the database with `DROP TABLE` or `DELETE`
- **Timeout (5 seconds):** Prevents infinite loops from `SELECT * FROM t1, t2, t3, t4` cross-joins
- **Order-insensitive comparison:** SQL results are unordered unless `ORDER BY` is present. We sort both result sets before comparing, *unless* the gold SQL contains `ORDER BY`
- **Value normalization:** Convert all values to lowercase strings — handles `1` vs `1.0` vs `True` mismatches
- **Platform-aware timeout:** `signal.SIGALRM` on Linux (reliable), `threading.Thread` with join timeout on Windows (for local testing)

#### 3b. Syntax Reward (`syntax.py`)

**What it does:** Returns 1.0 if `sqlparse` can parse the output and it contains a SELECT keyword.

**Why it's essential:** Gives the model a small reward for producing syntactically valid SQL, even if the results are wrong. This is the cheapest reward to compute and provides gradient signal for outputs that are "close but not right."

**Why SELECT is required:** `sqlparse` is extremely permissive — it "parses" almost any string, including `"hello world"`. Requiring SELECT prevents false positives.

#### 3c. Schema Coverage (`schema_coverage.py`)

**What it does:** Computes F1 score over table and column names between the model's SQL and the gold SQL.

**Why it's essential:** Bridges the gap between "syntactically valid but totally wrong" (syntax reward = 1, execution reward = 0) and "exactly correct" (execution reward = 1). A query that references the right tables and columns but gets the join condition wrong will score high on schema coverage, giving the model a gradient direction toward the correct answer.

**Example:**
```sql
-- Gold:  SELECT name FROM head WHERE age > 56
-- Model: SELECT name FROM head WHERE age > 30  (wrong filter value)
-- Schema F1: 1.0 (all identifiers match: name, head, age)
-- Execution: 0.0 (different result set)
```
Without schema coverage, this output gets 0.0 total reward — identical to `SELECT garbage FROM nowhere`. With schema coverage, the model knows it's on the right track.

#### 3d. Composite Reward (`composite.py`)

**What it does:** Combines individual rewards into Phase 1 (simple) and Phase 2 (rich) signals.

**Why two phases exist:**

**Phase 1** — for early training when the model is still learning basic SQL:
```
R = 1.0 × execution_accuracy + 0.2 × syntax_valid
```
Simple signal. The model either gets it right or doesn't.

**Phase 2** — for later training when the model needs fine-grained feedback:
```
R = 1.0 × execution_accuracy
  + 0.2 × execution_success    (query runs without error — partial credit)
  + 0.2 × syntax_valid
  + 0.2 × schema_coverage_f1
  + 0.1 × format_compliance    (output is pure SQL, no markdown/explanation)
```
The intermediate rewards (execution success, schema coverage) give the model something to learn from on examples where it can't yet get the full answer right.

**Why weights are pre-multiplied:** TRL's GRPOTrainer **sums** all reward functions. There's no separate weighting mechanism. Each function must return its already-weighted value. `_weighted(syntax_reward, 0.2)` wraps the function to multiply by 0.2.

**Correct SQL total reward:** Phase 1 = 1.2, Phase 2 = 1.7

#### 3e. Conversational Format Handling

TRL 0.14.0 passes completions to reward functions as `list[list[dict]]` (conversation format), not `list[str]`, because our dataset uses chat messages. Every reward function uses `_get_completion_text()` to extract the assistant's text from either format:

```python
# String input (direct testing)
_get_completion_text("SELECT 1")  → "SELECT 1"

# Conversation input (from TRL during training)
_get_completion_text([{"role": "assistant", "content": "SELECT 1"}])  → "SELECT 1"
```

---

### 4. SFT Training (`src/training/sft.py`)

**What it does:** Supervised fine-tuning — teaches the model to mimic gold SQL outputs.

**Why it's essential:** GRPO needs a policy that occasionally produces correct SQL to learn from. Raw Llama 3.1 8B at ~68% zero-shot means ~32% of attempts fail completely. GRPO's group-relative advantage estimation needs *variance* in reward within each group — if all 6 generations score 0, the gradient is zero and training stalls. SFT to ≥70% first ensures enough successful rollouts.

**How it works:**
1. Load Llama 3.1 8B Instruct
2. Attach LoRA adapters (rank 32) to all attention + MLP layers
3. Train on (prompt, gold_sql) pairs — the model learns to output gold SQL given schema + question
4. Save the LoRA weights as a checkpoint

**LoRA (Low-Rank Adaptation):**
Instead of updating all 8 billion parameters, LoRA freezes the base model and trains two small matrices (rank 32) per layer. This reduces trainable parameters from 8B to ~50M (~0.6%), making training feasible on a single GPU.

```
Original weight W (4096 × 4096) — frozen
LoRA: W + A × B where A (4096 × 32), B (32 × 4096) — trained
Total LoRA params: 2 × 4096 × 32 = 262,144 per layer
```

**4-bit Quantization (QLoRA):**
On V100 32GB, the full model doesn't fit. BitsAndBytes quantizes the frozen base weights to 4-bit NF4 format (~4GB instead of ~16GB), while LoRA adapters remain in bf16/fp16. Quality loss is <1%.

---

### 5. GRPO Training (`src/training/grpo.py`)

**What it does:** Reinforcement learning via Group Relative Policy Optimization — the core of the project.

**Why it's essential:** SFT teaches the model to *copy* gold SQL. GRPO teaches it to *explore* and find SQL that actually executes correctly. This is the difference between memorization and generalization.

**How GRPO works (conceptual):**

For each training prompt:
1. **Generate G completions** (G=6) from the current policy using temperature sampling
2. **Score each completion** using the reward function (execute against SQLite)
3. **Compute group-relative advantage:** for each completion, subtract the group mean reward and divide by group std. Completions better than the group average get positive advantage, worse ones get negative.
4. **Update the policy:** increase the probability of high-advantage completions, decrease low-advantage ones

```
Prompt: "How many employees in Engineering?"

Generation 1: SELECT COUNT(*) FROM emp WHERE dept='Eng'  → reward 1.0 → advantage +1.2
Generation 2: SELECT * FROM emp WHERE dept='Eng'          → reward 0.0 → advantage -0.8
Generation 3: SELECT COUNT(*) FROM emp                    → reward 0.0 → advantage -0.8
Generation 4: SELECT COUNT(*) FROM emp WHERE dept='Eng'  → reward 1.0 → advantage +1.2
Generation 5: SELCT CUNT FROM emp                         → reward 0.0 → advantage -0.8
Generation 6: SELECT COUNT(emp_id) FROM emp WHERE dept='Eng' → reward 1.0 → advantage +1.2

→ Policy update: make Generations 1, 4, 6 more likely; make 2, 3, 5 less likely
```

**Why GRPO over PPO:** GRPO doesn't need a separate critic/value network. The "baseline" is the group mean, which is free to compute. This saves ~50% memory and simplifies training.

**Dr. GRPO loss (`loss_type: dr_grpo`):**
Standard GRPO has a length bias — shorter SQL gets unfairly higher reward because the KL penalty scales with token count. Dr. GRPO normalizes by `max_completion_length` (a constant) instead of actual length, removing this bias. Critical for SQL which varies from `SELECT 1` (2 tokens) to complex subqueries (100+ tokens).

**Key hyperparameters:**
| Parameter | Value | Why |
|-----------|-------|-----|
| `beta: 0.001` | KL penalty coefficient | 0.1 over-constrains the policy; 0.001 allows exploration |
| `num_generations: 6` | Group size G | 4-8 is the sweet spot; fewer = noisy advantage, more = OOM |
| `temperature: 0.8` | Sampling temp | <0.5 = entropy collapse (all same output); >1.0 = too random |
| `max_grad_norm: 0.1` | Gradient clipping | Aggressive clipping for RL stability |
| `max_completion_length: 512` | Max SQL tokens | SQL is short; saves memory |

---

### 6. DSPy Prompt Optimization (`src/prompts/`)

**What it does:** Optimizes the prompt instructions and few-shot examples *without* changing model weights.

**Why it's essential:** This is the control condition that answers RQ2: "Can prompt optimization approach GRPO's gains without weight updates?" If DSPy gets close to GRPO, it means the model already *knows* how to write SQL — it just needs better instructions. If GRPO significantly beats DSPy, the model genuinely learned new capabilities through RL.

#### Signatures (`signatures.py`)

Defines the input/output contract for the Text-to-SQL task:
```python
class Text2SQL(dspy.Signature):
    question: str = dspy.InputField(...)    # Natural language question
    db_schema: str = dspy.InputField(...)   # CREATE TABLE DDL
    sql_query: str = dspy.OutputField(...)  # Generated SQL
```

DSPy uses this signature to automatically construct prompts, parse outputs, and optimize both the instructions and the few-shot demonstrations.

#### MIPROv2 Optimization (`optimize.py`)

**MIPROv2** (Multi-prompt Instruction Proposal Optimizer v2) optimizes in three phases:
1. **Bootstrap:** Generate candidate few-shot demonstrations from training data
2. **Propose:** Use the optimizer model to propose multiple instruction candidates
3. **Bayesian search:** Find the best combination of instructions + demonstrations on a validation set

The optimizer model (same Llama 8B, or an API model like Gemini) reasons about what makes a good prompt. It sees the task examples and proposes instructions like "When joining tables, always check for matching foreign key columns" or "Use COUNT(*) for counting rows, not COUNT(column)."

**Local model support:**
The `LocalTransformersLM` class wraps HuggingFace transformers to be compatible with DSPy's LM interface. It:
- Loads the model on GPU with automatic dtype selection (V100 → float16, A100 → bfloat16)
- Implements `forward()` (not `__call__`) returning objects matching the OpenAI response format that DSPy's internals expect
- Handles stop tokens (EOS + Llama's `<|eot_id|>`) to prevent generation runaway
- Cleans GPU memory between calls

---

### 7. Evaluation (`src/eval/`)

#### Run Eval (`run_eval.py`)

**What it does:** Runs any model (base, SFT checkpoint, GRPO checkpoint) on Spider dev/test and computes execution accuracy (EX).

**Three modes:**
- **Zero-shot:** Schema + question, no examples
- **Few-shot:** Add k demonstration examples (same-DB preferred) before the question
- **Model:** Load a trained checkpoint and generate

**Difficulty tiers:** Results are broken down by query complexity (easy/medium/hard/extra) based on the number of SQL components (JOINs, WHERE conditions, subqueries, set operations).

#### Error Analysis (`error_analysis.py`)

**What it does:** Categorizes failed predictions into actionable failure modes.

**Why it's essential:** A single EX number doesn't tell you *why* the model fails. Error analysis reveals:
- Is the model picking the wrong tables? → Need better schema understanding
- Is it getting joins wrong? → Need more complex training examples  
- Is it failing on aggregation? → Need reward signal for partial aggregation correctness
- Is it producing syntax errors? → SFT needs more epochs

**Categories:**
```
wrong_table       — References tables not in gold SQL
wrong_column      — Correct tables, wrong columns
wrong_join        — Missing or incorrect JOIN condition
wrong_aggregation — Wrong GROUP BY, HAVING, or aggregate function
wrong_filter      — Wrong WHERE conditions
wrong_order       — Wrong ORDER BY or LIMIT
syntax_error      — SQL doesn't parse
execution_error   — SQL parses but fails to execute
empty_result      — SQL executes but returns empty when gold doesn't
other             — Uncategorized
```

---

## Why This Order Matters

### SFT before GRPO (non-negotiable)

GRPO learns by comparing completions within a group. If the base model is too weak:
- All 6 generations score 0 → group mean = 0, std = 0 → no gradient
- Training stalls completely

SFT raises the baseline so that at least 2-3 out of 6 generations are correct, giving GRPO meaningful variance to learn from.

### Filtering before training (non-negotiable)

Empty-result gold queries produce reward 1.0 for any model output that also returns empty — including `SELECT 1 WHERE FALSE`. The model learns to produce empty-result queries to get free reward. This is a form of reward hacking.

### Phase 1 rewards before Phase 2 (recommended)

Start with the simple execution + syntax reward. Once the model is reliably generating valid SQL (~80% EX), switch to Phase 2 composite rewards for fine-grained optimization. Starting with Phase 2 too early can confuse the model — it might optimize for schema coverage instead of actual correctness.

---

## Model Architecture

```
Llama 3.1 8B Instruct (frozen base weights)
│
├── Attention layers (32 layers)
│   ├── q_proj ──→ LoRA A (8B×32) + B (32×8B)  ← trained
│   ├── k_proj ──→ LoRA A + B                    ← trained
│   ├── v_proj ──→ LoRA A + B                    ← trained
│   └── o_proj ──→ LoRA A + B                    ← trained
│
├── MLP layers (32 layers)
│   ├── gate_proj ──→ LoRA A + B                 ← trained
│   ├── up_proj   ──→ LoRA A + B                 ← trained
│   └── down_proj ──→ LoRA A + B                 ← trained
│
└── Embedding + LM head (frozen)

Total parameters: ~8B (frozen) + ~50M (LoRA, trained)
LoRA rank: 32
```

---

## Research Questions ↔ Components

| RQ | What it asks | Components involved | Comparison |
|----|-------------|--------------------|----|
| RQ1 | Does GRPO beat zero-shot and SFT? | grpo.py vs sft.py vs run_eval.py (zero-shot) | GRPO EX vs SFT EX vs Zero-shot EX |
| RQ2 | Can DSPy match GRPO without weight updates? | optimize.py vs grpo.py | DSPy EX vs GRPO EX |
| RQ3 | Do DSPy + GRPO stack? | Run optimized DSPy prompt with GRPO checkpoint | Combined EX vs individual EX |

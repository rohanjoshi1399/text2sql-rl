# WHY.md - Design Decision Log

Every non-obvious choice made during the project, why it was made, and what alternatives were rejected. Organized chronologically by project phase.

---

## 1. Model Choice

### Why Llama 3.1 8B Instruct?
- **Instruct, not base**: GRPO needs the model to produce coherent SQL from the start. Base models generate random continuations, meaning all 6 group completions score 0 and training stalls (zero gradient).
- **8B, not larger**: NEU Explorer cluster has A100 40GB GPUs. Llama 70B doesn't fit even with 4-bit quantization. 8B in bf16 uses ~16GB, leaving headroom for GRPO's 6 concurrent generations.
- **Not a smaller model**: 3B models (Llama 3.2) lack the capacity for complex SQL (JOINs, subqueries, set operations). Our zero-shot baseline with 8B was 67.8% -- 3B would likely be well under 50%.
- **Why not GPT-4 / Claude**: This is an RL fine-tuning project. We need to update model weights with GRPO, which requires local model access. API models are black-box.

---

## 2. Metric Choice

### Why Execution Accuracy (EX) instead of Exact Match (EM)?
- **EM is broken**: The same query can be written many valid ways. `SELECT name FROM emp WHERE dept='Eng'` and `SELECT e.name FROM emp AS e WHERE e.dept='Eng'` return identical results but EM scores the second as wrong.
- **Quantified impact**: Same model scored 29.2% EM vs 67.8% EX. ~40% of outputs were semantically correct but syntactically different from gold.
- **EX is the Spider standard**: The original Spider benchmark paper and all subsequent work (SkyRL-SQL, DAIL-SQL, etc.) report EX as the primary metric.
- **We learned this the hard way**: Update 1 used EM on 48 samples and reported 29.2%. Switching to EX on 1,034 samples was the single biggest correction in the project.

---

## 3. Data Decisions

### Why include train_others.json (Scholar, Yelp, IMDB, etc.)?
- Spider's `train_others.json` adds 1,659 examples from 6 databases (Scholar 569, GeoQuery 564, Academic 181, Restaurants 125, Yelp 111, IMDB 109). All in Spider JSON format with gold SQL.
- Adds schema diversity -- the model sees 6 new database schemas during training, improving generalization.
- Verified zero overlap between train databases (146) and dev databases (20). No data leakage.
- Total: 7,000 + 1,659 = 8,659 training examples.

### Why filter out 1,619 examples?
- **1,616 had empty gold results**: If the gold SQL returns empty, then *any* model output that also returns empty gets reward 1.0 -- including `SELECT 1 WHERE FALSE`. The model quickly learns to produce empty results for free reward. This is textbook reward hacking.
- **3 had execution errors**: Gold SQL itself failed to execute (malformed queries in the dataset).
- **Final count**: 8,659 - 1,619 = 7,040 clean training examples.

### Why cache gold results as JSON during preprocessing?
- During GRPO training, every reward computation requires executing gold SQL. With 7,040 examples x 6 generations x multiple epochs, that's hundreds of thousands of gold SQL executions.
- Pre-computing gold results once and storing as JSON avoids redundant execution and provides a major training speedup.
- Fallback: if cached result is missing or corrupted, the reward function re-executes gold SQL on the fly.

### Why truncate schema DDL at 3,500 characters?
- Some Spider databases have 15+ tables. The `academic` database DDL alone exceeds 5,000 characters. Combined with the question and system prompt, this risks exceeding Llama's context window.
- 3,500 chars covers all but ~3 databases. For those, we include all CREATE TABLE statements but truncate from the end.
- Alternative rejected: filtering specific columns or tables, which loses information and could make queries unanswerable.

### Why load DDL from `sqlite_master` instead of `tables.json`?
- `sqlite_master` gives exact CREATE TABLE statements including column types, PRIMARY KEY, and FOREIGN KEY constraints. This is what the model needs to generate valid JOINs.
- `tables.json` is a secondary representation that loses some information (no constraints, no exact types).
- The DDL format is also what the model would see at inference on new databases.

---

## 4. Reward Function Design

### Why two phases (simple then composite)?
- **Phase 1** (execution + syntax): Clean, dominant signal. The model learns "produce SQL that executes correctly." Adding too many weak signals early can confuse optimization.
- **Phase 2** (adds exec_success, schema F1, format compliance): Once the model reliably produces valid SQL, intermediate rewards provide gradient on partial successes. A query using the right tables but wrong filter gets schema F1 credit instead of flat 0.
- **Hypothesis**: Phase 2 helps the model learn from near-misses. If it doesn't beat Phase 1, that's itself a finding: GRPO's group-relative advantage is sufficient without reward shaping.

### Why these specific reward weights?

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Execution accuracy | 1.0 | Dominant signal. Correctness is what matters. |
| Syntax valid | 0.2 | Cheap validation. Rewards parseable SQL even if wrong. |
| Execution success | 0.2 | Partial credit for SQL that runs but returns wrong result. |
| Schema coverage F1 | 0.2 | Rewards referencing correct tables/columns even if logic is wrong. |
| Format compliance | 0.1 | Lowest weight. Clean SQL output is nice but not critical. |

- Weights are **pre-multiplied** into each reward function because TRL's GRPOTrainer sums all reward function outputs. There's no separate weighting mechanism.
- A perfect Phase 2 query scores 1.7 total. Execution accuracy contributes 59% of the maximum reward.

### Why compare result sets, not SQL strings?
- Same reason as the metric choice: semantically equivalent SQL should get equal reward. Comparing result sets is the only way to judge functional correctness.
- **Order sensitivity**: Results are sorted before comparison UNLESS the gold SQL contains `ORDER BY`, in which case order matters.
- **Value normalization**: All values converted to lowercase strings, floats rounded to 6 decimals. This handles `1` vs `1.0` vs `True` mismatches across SQLite's loose typing.

### Why in-memory database copies?
- The model can generate `DROP TABLE`, `DELETE`, or `UPDATE` statements. Executing these against the real database would corrupt it for all subsequent queries.
- Solution: `disk_conn.backup(mem_conn)` creates an in-memory copy. Each query gets its own isolated connection.
- Spider has ~140 databases, so caching all in-memory copies costs ~50MB RAM -- negligible.

### Why 5-second timeout?
- Spider queries are simple and execute in milliseconds. A query taking >5 seconds is almost certainly a runaway cross-join (`SELECT * FROM A, B, C, D` with no WHERE clause).
- Without timeout, one bad query can hang the entire training loop.
- 5 seconds is generous enough to never timeout a legitimate query, strict enough to catch infinite joins.

### Why platform-aware timeout (signal vs threading)?
- `signal.SIGALRM` is the gold standard on Linux -- it interrupts CPU-bound operations. But it only works in the main thread and doesn't exist on Windows.
- DSPy's MIPROv2 optimizer runs evaluations in parallel worker threads, where `signal.alarm` crashes.
- Solution: auto-detect `platform.system()` and `threading.current_thread()`. Use signal in main thread on Linux, threading-based timeout everywhere else.
- This was discovered the hard way when DSPy evaluation crashed with "signal only works in main thread".

### Why `_get_completion_text()` in every reward function?
- TRL 0.14.0's GRPOTrainer passes completions as conversation dicts (`[{"role": "assistant", "content": "..."}]`), not plain strings. Earlier TRL versions passed strings.
- Every reward function needs to extract the actual SQL text from whatever format it receives.
- `_get_completion_text()` handles both: if it's a string, return it; if it's a list of dicts, find the assistant message.
- This was a multi-hour debugging session. Reward functions were silently receiving dicts, failing to parse them, and returning 0.0 for everything.

### Why schema coverage uses F1 (not precision or recall alone)?
- **Precision alone**: A query using only 1 correct table (of 3 needed) gets precision 1.0 but misses 2 tables. Not useful.
- **Recall alone**: A query referencing every table in the database gets recall 1.0 but most are irrelevant. Not useful.
- **F1**: Balances "did you use the right tables?" with "did you avoid unnecessary tables?" Provides meaningful gradient for partial matches.

### Why format compliance checks for markdown / natural language?
- Llama 3.1 Instruct loves to wrap SQL in \`\`\`sql blocks and add explanations ("Here is the SQL query..."). This extra text wastes tokens and can confuse downstream parsing.
- Format compliance reward (weight 0.1) gently pushes the model toward outputting pure SQL. Lowest weight because it's cosmetic, not functional.

---

## 5. Training Decisions

### Why SFT before GRPO (mandatory warm-up)?
- GRPO computes advantage by comparing G completions per prompt. If the model is too weak, all G completions score 0, mean reward = 0, advantage = 0, gradient = 0. Training does nothing.
- SFT to ~70% EX ensures that roughly half of GRPO's completions score >0 per group, providing meaningful variance for the advantage calculation.
- Empirically, SFT reached 69.2% after 1 epoch. That's close enough to enable GRPO.
- This is documented in the PRD as a "high risk if skipped" item.

### Why 70% EX as the SFT target for GRPO?
- Below ~60%, too many groups have all-zero rewards (no learning signal).
- Above ~80%, there's not enough room for GRPO to improve (diminishing returns on the RL investment).
- 70% is the sweet spot where enough completions succeed (for gradient signal) but enough fail (for room to improve).

### Why LoRA instead of full fine-tuning?
- Llama 8B has 8 billion parameters. Full fine-tuning requires storing optimizer states for all parameters: ~64GB for AdamW on fp32 gradients. Doesn't fit on A100 40GB.
- LoRA adds ~50M trainable parameters (0.6% of the model). Fits comfortably with 4-bit base model on V100 32GB.
- Quality loss from LoRA is <1% on SQL tasks at rank 32 (per empirical literature).

### Why rank 32 LoRA, not 16 or 64?
- Rank 16: validated in literature, works fine, slightly lower capacity for complex SQL patterns.
- Rank 32: marginally better than 16 at 8B scale for SQL generation, with negligible VRAM overhead (~25MB more).
- Rank 64+: diminishing returns. Extra parameters don't improve SQL quality but increase OOM risk during GRPO.
- lora_alpha = rank (standard practice, effective learning rate scales as alpha/rank = 1.0).
- dropout = 0.0: avoids introducing variance in RL training where the reward signal is already noisy.

### Why target all 7 linear modules (q/k/v/o_proj + gate/up/down_proj)?
- Attention modules (q, k, v, o) handle cross-column and cross-table understanding -- essential for JOINs.
- MLP modules (gate, up, down) handle SQL syntax patterns and logical reasoning.
- Targeting fewer modules (e.g., only q, v) was tested briefly and underperformed.

### Why QLoRA (4-bit quantization) on V100?
- V100 has 32GB VRAM. Llama 8B in bf16 is ~16GB. Add LoRA adapters, optimizer states, KV cache, and gradient checkpointing activations: exceeds 32GB.
- NF4 quantization reduces the base model to ~4GB. Everything else fits.
- A100 40GB runs in full bf16 without quantization -- only V100 needs 4-bit.
- bf16 vs fp16: V100 doesn't support native bf16 compute. We use fp16 LoRA adapters on V100, bf16 on A100.

### Why `gradient_checkpointing_kwargs={"use_reentrant": False}`?
- PyTorch's default gradient checkpointing uses `use_reentrant=True`, which has a known bug with 4-bit quantized models: "None of the inputs have requires_grad".
- Setting `use_reentrant=False` uses the newer implementation that correctly handles quantized inputs.
- This was discovered during smoke testing when 4-bit + gradient checkpointing crashed.
- For the smoke test specifically, gradient checkpointing is disabled entirely (faster, fewer variables to debug).

### Why paged_adamw_8bit optimizer?
- Standard AdamW stores 2 states per parameter (momentum + variance) in fp32. 8-bit Adam compresses these to 8-bit, halving optimizer VRAM.
- "Paged" variant additionally offloads optimizer states to CPU when GPU memory is tight.
- Essential for fitting GRPO training on A100 40GB with 6 generations per prompt.

### Why cosine learning rate schedule?
- Smooth decay without sharp drops. Prevents training instability from sudden LR changes.
- Warmup ratio 0.05 for SFT (standard), 0.1 for GRPO (RL needs gentler ramp-up to avoid early policy divergence).

### Why SFT learning rate 2e-4 but GRPO learning rate 5e-6?
- SFT is supervised -- the gradient signal is clean and consistent. Higher LR fine.
- GRPO is RL -- the reward signal is noisy (stochastic sampling, binary rewards). High LR causes catastrophic forgetting or policy collapse.
- 5e-6 is typical for RL fine-tuning on language models.

---

## 6. GRPO-Specific Decisions

### Why GRPO instead of PPO or DPO?
- **PPO**: Requires a separate value network (critic), which doubles VRAM. Doesn't fit on A100 40GB with Llama 8B + 6 generations.
- **DPO**: Requires pre-collected preference pairs. We don't have human SQL preferences. Could construct them synthetically, but that's an extra pipeline.
- **GRPO**: No critic needed (uses group-relative advantage). Generates G completions per prompt, computes mean reward as the baseline. Simpler, lower VRAM, and TRL has native support.

### Why num_generations=6?
- <4: noisy advantage estimate (too few samples to compute meaningful mean).
- 6: sweet spot for A100 40GB. Each generation consumes KV cache memory (~2GB per generation at 512 tokens).
- >8: OOM on A100 40GB. Would need to reduce batch size to 0, which isn't valid.

### Why beta=0.001 (KL penalty)?
- beta controls how much the policy can diverge from the reference (SFT) model.
- 0.1: too conservative. Model barely changes, GRPO training is slow.
- 0.001: allows meaningful exploration while preventing catastrophic divergence. Standard in SQL RL literature.
- 0.0001: too permissive. Model can wander far from SFT, producing incoherent SQL.

### Why temperature=0.8?
- <0.5: entropy collapse. All 6 completions per group become nearly identical, advantage is ~0, training stalls.
- 0.8: diverse enough to explore different SQL formulations, focused enough to stay on-task.
- >1.0: too random. Completions become nonsensical, all score 0, training stalls.

### Why max_completion_length=512?
- Spider SQL queries rarely exceed 200 tokens. 512 provides generous headroom for complex nested queries.
- Longer = more VRAM per generation (KV cache scales linearly).
- 512 is 4x the typical query length, enough for UNION/INTERSECT patterns.

### Why max_grad_norm=0.1 (aggressive gradient clipping)?
- RL training has high gradient variance. Occasional outlier gradients can destabilize the policy.
- 0.1 is aggressive but standard for RL fine-tuning. SFT uses the default 1.0 (clean gradients).

### Why `use_vllm: false`?
- TRL + vLLM + LoRA had known compatibility issues at the time (TRL issue #2698).
- Installing vLLM on the cluster broke PyTorch (pulled incompatible CUDA wheels). Had to reinstall torch==2.5.1+cu121 to recover.
- Standard HF `generate()` is slower but stable with LoRA adapters.
- vLLM is only beneficial for inference speed during generation; training speed is dominated by backward pass anyway.

---

## 7. DSPy Decisions

### Why DSPy MIPROv2, not APO?
- APO (Automatic Prompt Optimization / ProTeGi) was validated only on classification tasks. No published SQL results.
- DSPy MIPROv2 has demonstrated Text-to-SQL capability at the 7B model scale.
- MIPROv2's 3-phase approach (bootstrap demos, propose instructions, Bayesian search) systematically explores the prompt space.

### Why separate task model and optimizer model?
- The optimizer model needs to reason about *what makes a good SQL prompt* -- a meta-cognitive task harder than generating SQL.
- Using Llama 8B as both task and optimizer (self-optimization) scored 58.6%, below zero-shot (67.8%). The model can't teach itself what it doesn't know.
- Plan: use Gemini Flash or GPT-4o-mini as optimizer (much stronger reasoning) with Llama 8B as the task model.
- API quota limits forced the self-optimization fallback for Update 2.

### Why build a custom LocalTransformersLM wrapper?
- DSPy expects OpenAI-format API responses (`response.choices[i].message.content`).
- Tried `dspy.LM("meta-llama/...")`: routes through litellm, which doesn't recognize local model paths.
- Tried `dspy.LM("huggingface/meta-llama/...")`: sends requests to HuggingFace Inference API (cloud), not local GPU.
- Solution: custom class that loads HF model locally, runs `model.generate()`, and wraps results in OpenAI-compatible `SimpleNamespace` objects.

### Why ChatAdapter instead of JSONAdapter?
- JSONAdapter expects the model to output structured JSON fields. Llama 8B frequently produces malformed JSON, causing optimization to fail silently.
- ChatAdapter uses `[[ ## field ## ]]` markdown-style headers. Llama's instruction tuning handles these reliably.
- Switched after multiple failed DSPy runs where JSON parsing errors killed optimization.

### Why rename `schema` to `db_schema` in DSPy signatures?
- `schema` shadows `dspy.Signature.schema`, a built-in class method. This caused cryptic errors during MIPROv2 optimization.
- Renamed to `db_schema` -- functionally identical, avoids the name collision.

---

## 8. Evaluation Decisions

### Why evaluate on all 1,034 dev examples (not a subset)?
- Update 1 used 48 samples, which had high variance and wasn't representative.
- 1,034 gives stable estimates. Confidence interval is ~+/-2.8% at 95% confidence.
- Test set (2,147 examples) is locked until final evaluation to prevent overfitting.

### Why report by difficulty tier (Easy/Medium/Hard/Extra Hard)?
- Overall accuracy hides where methods differ. SFT improved Hard queries by +7.2% but *destroyed* Extra Hard (-16.7%, catastrophic forgetting).
- Difficulty breakdown reveals which SQL patterns each method handles. This directly informs GRPO reward design -- we know which queries need the most improvement.

### Why greedy decoding (temperature=0) for evaluation?
- Reproducible results. Same input always produces same output.
- Temperature sampling adds noise that makes comparisons unreliable.
- Standard practice for evaluation benchmarks.

---

## 9. Infrastructure Decisions

### Why TRL 0.14.0 (pinned, not latest)?
- TRL 0.14.0 was the version that passed our GRPO smoke test on the cluster.
- Newer TRL versions (0.23+) changed the GRPOTrainer API: added `loss_type`, `epsilon`, `log_completions` parameters that don't exist in 0.14.0, and changed how completions are formatted.
- Upgrading TRL would require rewriting reward functions, training scripts, and re-running smoke tests.
- We removed `loss_type`, `epsilon`, `log_completions` from our GRPO config since they don't exist in 0.14.0.

### Why transformers==4.47.1?
- Pinned to match TRL 0.14.0's tested dependency. Newer transformers versions can break TRL's internal imports (e.g., `FSDPModule`, `AutoProcessor`).
- Discovered through import errors during cluster setup.

### Why conda, not venv?
- NEU Explorer cluster manages environments through `module load anaconda3/2024.06`.
- CUDA toolkit, cuDNN, and NCCL are pre-configured for conda. venv would require manual library path setup.

### Why checkpoint-based resume across Slurm jobs?
- Cluster has an 8-hour wall time limit per job. SFT training for 3 epochs takes ~24 hours.
- Trainer saves checkpoints every 200 steps. On resubmit, the script finds the latest checkpoint and resumes.
- SIGUSR1 signal handler saves a checkpoint before Slurm kills the job, minimizing lost work.

### Why `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`?
- GRPO generates 6 completions per prompt, causing VRAM usage to spike. Default CUDA allocator pre-allocates large contiguous blocks that fragment memory.
- `expandable_segments` allows the allocator to grow blocks incrementally, reducing fragmentation and preventing OOM at peak usage.

### Why `TOKENIZERS_PARALLELISM=false`?
- HuggingFace tokenizers use Rust-based parallelism. When combined with PyTorch DataLoader's multiprocessing, this causes deadlocks.
- Disabling tokenizer parallelism adds negligible overhead (tokenization is fast) and eliminates random training hangs.

### Why `HF_HOME=/scratch/$USER/hf_cache`?
- Home directory quota on the cluster is small. Llama 8B's model files are ~16GB.
- `/scratch` has large per-user storage. Pointing HF cache there avoids quota errors during model downloads.

---

## 10. Key Findings (Empirical)

### Why did 5-shot perform worse than zero-shot? (66.3% vs 67.8%)
- Few-shot examples came from different databases with different schemas.
- A demo showing a JOIN on `employee_id` primes the model to look for similar columns in unrelated schemas.
- Particularly harmful on Hard queries: -6.6 percentage points.
- Implication: naive example selection is counterproductive. DSPy's value is in optimizing *which* examples to show.

### Why did SFT drop to 0% on Extra Hard after just 1 epoch?
- Extra Hard queries use UNION, INTERSECT, EXCEPT -- rare patterns in training data (<2% of examples).
- After 1 epoch of SFT, the model overfits to common patterns (simple SELECT, basic JOINs) and loses its zero-shot ability on rare constructs.
- This is catastrophic forgetting -- exactly what GRPO is supposed to help with, since it explores diverse solutions.

### Why is the training loss (~0.055) much lower than eval loss (~0.87)?
- The model has memorized most training examples after 1+ epochs (low training loss).
- But it doesn't generalize well to dev set schemas it hasn't seen (high eval loss).
- This gap suggests continued SFT training risks overfitting. The best model may be at epoch ~2, not epoch 3.

---

## 11. Rejected Alternatives

| Considered | Rejected Because |
|------------|-----------------|
| Agent Lightning framework | Adds orchestration complexity for multi-turn tool calling. Our task is single-turn SQL generation. TRL handles everything needed. |
| vLLM for generation | Breaks LoRA compatibility in TRL 0.14.0. Installing it corrupted PyTorch on the cluster. |
| Exact match metric | Penalizes correct queries that use different syntax. 40% false negative rate. |
| GPT-4 / Claude as base model | Can't do weight-level RL training on API models. Need local access for GRPO. |
| Rank 64+ LoRA | Diminishing returns on quality, increased OOM risk with 6 GRPO generations. |
| APO instead of DSPy | Never validated on SQL tasks. MIPROv2 has proven Text-to-SQL results. |
| Full fine-tuning (no LoRA) | 64GB optimizer states for 8B params. Doesn't fit on A100 40GB. |
| PPO instead of GRPO | Requires separate critic network, doubling VRAM. GRPO achieves similar results without one. |
| DPO instead of GRPO | Needs pre-collected preference pairs. We don't have human SQL preferences. |
| Ignoring train_others.json | Loses 1,659 examples and 6 database schemas. Less diversity hurts generalization. |
| Keeping empty-result queries | Enables reward hacking. Model learns to produce empty results for free reward. |

---

## 12. Lessons Learned (Mistakes)

### save_total_limit: 3 was too aggressive
- **What happened**: SFT training required 3 Slurm jobs across ~24 hours (8-hour wall time limit). With `save_steps: 200` and `save_total_limit: 3`, only the last 3 checkpoints (600 steps) were kept. By the time we noticed overfitting at epoch 2.27 (eval_loss 0.983 vs 0.871 at epoch 1.36), checkpoint-800 (the likely best model at epoch ~1.8) had already been deleted.
- **Root cause**: Didn't account for the interaction between multi-job training, overfitting risk, and checkpoint retention. Each factor was considered individually but not together.
- **Fix**: Changed `save_total_limit` to 10. With 1320 total steps and `save_steps: 200`, that's 7 checkpoints total -- all of them fit. Disk cost is ~200MB per checkpoint (LoRA adapters only), so 10 checkpoints = ~2GB. Negligible on `/scratch`.
- **Rule**: When training spans multiple Slurm jobs and overfitting is possible, keep ALL checkpoints. Delete manually after evaluation, not automatically during training.

---

*Last updated: 2026-04-08 (SFT at epoch 2.73, best checkpoint likely lost to save_total_limit)*

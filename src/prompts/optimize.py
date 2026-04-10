"""DSPy MIPROv2 prompt optimization for Text-to-SQL."""

import argparse
import os
import time
from types import SimpleNamespace

import dspy

from src.prompts.signatures import Text2SQL, Text2SQLWithReasoning, build_dspy_examples
from src.rewards.execution import execute_sql_safe, compare_results, extract_sql


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def execution_accuracy_metric(
    example: dspy.Example, prediction: dspy.Prediction, trace=None
) -> float:
    """
    DSPy metric: execute predicted SQL against the database, compare with gold.
    Returns 1.0 if result sets match, 0.0 otherwise.
    """
    pred_sql = extract_sql(prediction.sql_query)
    gold_sql = example.sql_query
    db_path = example.db_path

    gold_result, gold_err = execute_sql_safe(gold_sql, db_path)
    if gold_err:
        return 0.0

    pred_result, pred_err = execute_sql_safe(pred_sql, db_path)
    if pred_err:
        return 0.0

    return 1.0 if compare_results(gold_result, pred_result, gold_sql) else 0.0


# ---------------------------------------------------------------------------
# Local Transformers LM — compatible with DSPy 3.1.3
# ---------------------------------------------------------------------------

class _DictableNamespace(SimpleNamespace):
    """SimpleNamespace that supports dict() conversion for DSPy's dict(response.usage)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data = kwargs

    def __iter__(self):
        return iter(self._data.items())


class LocalTransformersLM(dspy.LM):
    """
    DSPy-compatible LM that runs HuggingFace transformers locally on GPU.

    Contract with DSPy 3.1.3:
    - super().__init__() called with model name, sets self.model, self.kwargs, etc.
    - forward() overridden (NOT __call__) — __call__ is in the base class
    - forward() returns an object with attribute access matching OpenAI response:
        response.choices[i].message.content  (str)
        response.usage                       (dict()-able)
        response.model                       (str)
        response._hidden_params              (dict)
    - _process_lm_response in base class handles history, cost tracking
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(
            model=model_name,
            model_type="chat",
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 256),
        )
        self._model_name = model_name
        self._hf_model = None
        self._hf_tokenizer = None
        self._stop_token_ids = None
        self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading local model: {self._model_name}")

        self._hf_tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

        # V100 (compute 7.0) → float16, A100+ (8.0+) → bfloat16
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU: {gpu_name} (compute {cap[0]}.{cap[1]}), dtype={dtype}")
        else:
            dtype = torch.float32
            print("  No GPU — using CPU (will be slow)")

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self._hf_model.eval()

        # Build stop token IDs: EOS + Llama's <|eot_id|>
        self._stop_token_ids = [self._hf_tokenizer.eos_token_id]
        eot_id = self._hf_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id != self._hf_tokenizer.unk_token_id:
            self._stop_token_ids.append(eot_id)

        print(f"  Model loaded. Stop tokens: {self._stop_token_ids}")

    def forward(self, prompt=None, messages=None, **kwargs):
        import torch

        # Build input text
        if messages:
            text = self._hf_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif prompt:
            text = prompt
        else:
            return self._empty_response()

        # Tokenize
        inputs = self._hf_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # Temperature: 0 or None → greedy, else sample
        temp = kwargs.get("temperature", self.kwargs.get("temperature", 0.7))
        if temp is None or temp <= 0:
            gen_kwargs = {"do_sample": False}
        else:
            gen_kwargs = {"do_sample": True, "temperature": float(temp)}

        # Generate
        max_new = kwargs.get("max_tokens", self.kwargs.get("max_tokens", 256))
        with torch.no_grad():
            output_ids = self._hf_model.generate(
                **inputs,
                max_new_tokens=max_new,
                pad_token_id=self._hf_tokenizer.pad_token_id,
                eos_token_id=self._stop_token_ids,
                **gen_kwargs,
            )

        generated_ids = output_ids[0][input_len:]
        response_text = self._hf_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completion_tokens = len(generated_ids)

        # Free memory
        del inputs, output_ids, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self._build_response(response_text, input_len, completion_tokens)

    def _build_response(self, text: str, prompt_tokens: int, completion_tokens: int):
        """Build object matching what DSPy's _process_lm_response expects."""
        message = SimpleNamespace(content=text, role="assistant")
        choice = SimpleNamespace(message=message)
        usage = _DictableNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=self._model_name,
            created=int(time.time()),
            _hidden_params={},
        )

    def _empty_response(self):
        return self._build_response("", 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_api_model(name: str) -> bool:
    return any(name.startswith(p) for p in
               ["groq/", "gemini/", "openai/", "anthropic/", "together/"])


def _make_lm(model_name: str):
    """Create a DSPy LM — API-based or local via transformers."""
    if _is_api_model(model_name):
        return dspy.LM(model_name)
    return LocalTransformersLM(model_name)


# ---------------------------------------------------------------------------
# Main optimization
# ---------------------------------------------------------------------------

def optimize(
    task_model: str = "groq/llama-3.1-8b-instant",
    opt_model: str = "gemini/gemini-2.0-flash",
    optimizer_type: str = "miprov2",
    use_reasoning: bool = False,
    trainset_size: int = 200,
    output_path: str = "checkpoints/dspy_optimized.json",
):
    """
    Run DSPy prompt optimization.

    Args:
        task_model: API model (groq/..., gemini/...) or local HF model ID
        opt_model: model that proposes prompt improvements
        optimizer_type: "miprov2" or "bootstrap"
        use_reasoning: use ChainOfThought (adds reasoning step)
        trainset_size: number of training examples to use
        output_path: where to save the optimized program (.json)
    """
    # Guard: can't load two different local models (would OOM)
    if (not _is_api_model(task_model) and not _is_api_model(opt_model)
            and task_model != opt_model):
        raise ValueError(
            "Cannot load two different local models simultaneously (OOM). "
            "Use the same model for both, or use an API model for one."
        )

    # Configure LMs
    task_lm = _make_lm(task_model)
    opt_lm = task_lm if task_model == opt_model else _make_lm(opt_model)

    # Use ChatAdapter — more reliable with local models than JSONAdapter.
    # ChatAdapter uses [[ ## field ## ]] headers which Llama follows well.
    # JSONAdapter expects raw JSON output which local models often botch.
    dspy.configure(lm=task_lm, adapter=dspy.ChatAdapter(), provide_traceback=True)

    # Build examples
    print(f"Loading {trainset_size} training examples...")
    examples = build_dspy_examples(max_examples=trainset_size)

    split_idx = int(0.8 * len(examples))
    trainset = examples[:split_idx]
    devset = examples[split_idx:]
    print(f"Train: {len(trainset)}, Dev: {len(devset)}")

    # Build program
    # Predict = direct answer, ChainOfThought = adds reasoning step
    if use_reasoning:
        program = dspy.ChainOfThought(Text2SQLWithReasoning)
    else:
        program = dspy.Predict(Text2SQL)

    # Optimize
    if optimizer_type == "miprov2":
        print("Running MIPROv2 optimization...")
        optimizer = dspy.MIPROv2(
            metric=execution_accuracy_metric,
            auto="medium",
            prompt_model=opt_lm,
        )
        optimized = optimizer.compile(program, trainset=trainset)

    elif optimizer_type == "bootstrap":
        print("Running BootstrapFewShot optimization...")
        optimizer = dspy.BootstrapFewShot(
            metric=execution_accuracy_metric,
            max_bootstrapped_demos=5,
            max_labeled_demos=5,
        )
        optimized = optimizer.compile(program, trainset=trainset)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Save
    if not output_path.endswith(".json"):
        output_path += ".json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    optimized.save(output_path)
    print(f"Saved optimized program to {output_path}")

    # Evaluate on devset
    print("Evaluating on dev set...")
    correct = 0
    total_evaluated = 0
    for ex in devset:
        try:
            pred = optimized(question=ex.question, db_schema=ex.db_schema)
            score = execution_accuracy_metric(ex, pred)
            correct += score
            total_evaluated += 1
        except Exception as e:
            total_evaluated += 1
            print(f"  Error: {e}")

    dev_acc = correct / total_evaluated if total_evaluated else 0
    print(f"Optimized program dev accuracy: {dev_acc:.4f} ({int(correct)}/{total_evaluated})")

    return optimized


def main():
    parser = argparse.ArgumentParser(description="DSPy prompt optimization for Text-to-SQL")
    parser.add_argument("--optimizer", default="miprov2", choices=["miprov2", "bootstrap"])
    parser.add_argument("--task-model", default="groq/llama-3.1-8b-instant")
    parser.add_argument("--opt-model", default="gemini/gemini-2.0-flash")
    parser.add_argument("--use-reasoning", action="store_true")
    parser.add_argument("--trainset-size", type=int, default=200)
    parser.add_argument("--output", default="checkpoints/dspy_optimized.json")
    args = parser.parse_args()

    optimize(
        task_model=args.task_model,
        opt_model=args.opt_model,
        optimizer_type=args.optimizer,
        use_reasoning=args.use_reasoning,
        trainset_size=args.trainset_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

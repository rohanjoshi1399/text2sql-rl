"""Evaluation pipeline — run any model on Spider dev/test and compute execution accuracy."""

import argparse
import json
import os
import re
import time

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.spider_loader import SPIDER_DATA_DIR, load_spider_split, format_prompt, get_schema_ddl
from src.rewards.execution import execute_sql_safe, compare_results, extract_sql


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def classify_difficulty(sql_struct: dict) -> str:
    """
    Approximate Spider difficulty from the structured SQL representation.
    Based on number of components (JOINs, WHERE conditions, subqueries, set ops).
    """
    if sql_struct is None:
        return "unknown"

    score = 0
    if sql_struct.get("where"):
        score += len(sql_struct["where"])
    if sql_struct.get("groupBy"):
        score += 1
    if sql_struct.get("having"):
        score += 1
    if sql_struct.get("orderBy"):
        score += 1
    if sql_struct.get("limit") is not None:
        score += 1
    if sql_struct.get("intersect"):
        score += 3
    if sql_struct.get("union"):
        score += 3
    if sql_struct.get("except"):
        score += 3

    # Count nested tables (JOINs)
    from_clause = sql_struct.get("from", {})
    if isinstance(from_clause, dict):
        table_units = from_clause.get("table_units", [])
        if len(table_units) > 1:
            score += len(table_units) - 1

    if score <= 1:
        return "easy"
    elif score <= 3:
        return "medium"
    elif score <= 6:
        return "hard"
    else:
        return "extra"


def build_few_shot_prompt(
    question: str,
    schema_ddl: str,
    k: int,
    train_data: list[dict],
    db_id: str | None = None,
) -> list[dict]:
    """
    Build a few-shot prompt with k demonstration examples.
    Prefers examples from the same database. Falls back to random.
    """
    # Find same-db examples
    same_db = [ex for ex in train_data if ex.get("db_id") == db_id] if db_id else []
    if len(same_db) >= k:
        demos = same_db[:k]
    else:
        demos = same_db + [ex for ex in train_data if ex.get("db_id") != db_id][: k - len(same_db)]

    # Build demo text
    demo_text = ""
    for demo in demos[:k]:
        demo_text += f"Question: {demo['question']}\nSQL: {demo['gold_sql']}\n\n"

    from src.data.spider_loader import SYSTEM_PROMPT

    user_content = (
        f"### Database Schema:\n{schema_ddl}\n\n"
        f"### Examples:\n{demo_text}"
        f"### Question:\n{question}\n\n### SQL:"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def evaluate(
    model_path: str,
    split: str = "dev",
    mode: str = "zero-shot",
    data_dir: str = SPIDER_DATA_DIR,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    few_shot_k: int = 5,
    output_dir: str = "results",
    raw_data: list[dict] | None = None,
) -> dict:
    """
    Run evaluation on a Spider split.
    Returns dict with overall_ex, by_difficulty, and predictions.
    """
    print(f"Evaluating {model_path} on {split} split (mode={mode})...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left padding for batch generation

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    model.eval()

    # Load dataset
    dataset = load_spider_split(split, data_dir=data_dir)

    # For few-shot, load training data as demos
    train_data = None
    if mode == "few-shot":
        train_ds = load_spider_split("train", data_dir=data_dir)
        train_data = [train_ds[i] for i in range(len(train_ds))]

    predictions = []
    correct = 0
    difficulty_counts = {}
    difficulty_correct = {}

    start_time = time.time()

    for i in range(0, len(dataset), batch_size):
        batch_end = min(i + batch_size, len(dataset))
        batch = [dataset[j] for j in range(i, batch_end)]

        # Build prompts
        prompts = []
        for ex in batch:
            if mode == "few-shot" and train_data:
                prompt = build_few_shot_prompt(
                    ex["question"],
                    get_schema_ddl(ex["db_id"], data_dir, split),
                    few_shot_k,
                    train_data,
                    ex["db_id"],
                )
            else:
                prompt = ex["prompt"]
            prompts.append(prompt)

        # Tokenize
        texts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=None,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated portion
        for j, (output, ex) in enumerate(zip(outputs, batch)):
            input_len = inputs["input_ids"][j].shape[0]
            generated_ids = output[input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            pred_sql = extract_sql(generated_text)

            # Execute and compare
            gold_result, gold_err = execute_sql_safe(ex["gold_sql"], ex["db_path"])
            pred_result, pred_err = execute_sql_safe(pred_sql, ex["db_path"])

            is_correct = False
            if gold_err is None and pred_err is None:
                is_correct = compare_results(gold_result, pred_result, ex["gold_sql"])

            if is_correct:
                correct += 1

            # Difficulty tracking (from raw data if available)
            difficulty = "unknown"
            if raw_data and (i + j) < len(raw_data):
                difficulty = classify_difficulty(raw_data[i + j].get("sql"))
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            if is_correct:
                difficulty_correct[difficulty] = difficulty_correct.get(difficulty, 0) + 1

            predictions.append({
                "index": i + j,
                "db_id": ex["db_id"],
                "question": ex["question"],
                "gold_sql": ex["gold_sql"],
                "pred_sql": pred_sql,
                "correct": is_correct,
                "difficulty": difficulty,
                "pred_error": pred_err,
            })

        # Progress
        elapsed = time.time() - start_time
        processed = min(batch_end, len(dataset))
        current_ex = correct / processed if processed > 0 else 0
        print(f"  [{processed}/{len(dataset)}] EX={current_ex:.4f} ({elapsed:.1f}s)")

    # Compute metrics
    overall_ex = correct / len(dataset) if len(dataset) > 0 else 0
    by_difficulty = {}
    for diff in difficulty_counts:
        by_difficulty[diff] = (
            difficulty_correct.get(diff, 0) / difficulty_counts[diff]
            if difficulty_counts[diff] > 0
            else 0
        )

    results = {
        "model": model_path,
        "split": split,
        "mode": mode,
        "overall_ex": overall_ex,
        "total": len(dataset),
        "correct": correct,
        "by_difficulty": by_difficulty,
        "predictions": predictions,
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"eval_{split}_{mode}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: EX={overall_ex:.4f} ({correct}/{len(dataset)})")
    print(f"By difficulty: {by_difficulty}")
    print(f"Saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Spider")
    parser.add_argument("--model", required=True, help="Model name or checkpoint path")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--mode", default="zero-shot", choices=["zero-shot", "few-shot", "model"])
    parser.add_argument("--data-dir", default=SPIDER_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("-k", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--output", default="results")
    parser.add_argument("--config", default=None, help="Load settings from eval config YAML")
    args = parser.parse_args()

    # Override from config if provided
    if args.config:
        config = load_config(args.config)
        eval_cfg = config.get("eval", {})
        args.split = eval_cfg.get("split", args.split)
        args.mode = eval_cfg.get("mode", args.mode)
        args.batch_size = eval_cfg.get("batch_size", args.batch_size)
        args.max_new_tokens = eval_cfg.get("max_new_tokens", args.max_new_tokens)
        args.k = eval_cfg.get("few_shot_k", args.k)
        if config.get("checkpoint_path"):
            args.model = config["checkpoint_path"]

    # Load raw data for difficulty classification
    raw_data = None
    raw_path = os.path.join(args.data_dir, f"{args.split}.json")
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            raw_data = json.load(f)

    evaluate(
        model_path=args.model,
        split=args.split,
        mode=args.mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        few_shot_k=args.k,
        output_dir=args.output,
        raw_data=raw_data,
    )


if __name__ == "__main__":
    main()

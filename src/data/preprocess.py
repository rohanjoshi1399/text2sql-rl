"""Preprocessing — filter Spider training data to remove empty-result and slow gold queries."""

import argparse
import json
import os
import sys
import time

from datasets import Dataset

from src.data.spider_loader import SPIDER_DATA_DIR, load_spider_split
from src.rewards.execution import execute_sql_safe


def preprocess_split(
    dataset: Dataset,
    timeout: int = 5,
    remove_empty: bool = True,
    output_path: str | None = None,
    verbose: bool = True,
) -> Dataset:
    """
    Filter a HuggingFace Dataset (from spider_loader):
      1. Execute each gold_sql against its db_path
      2. Remove examples where gold SQL returns empty results
      3. Remove examples where gold SQL exceeds timeout or errors

    Returns filtered Dataset.
    """
    reasons = {"empty_result": 0, "timeout": 0, "error": 0}
    keep_indices = []
    gold_results = []  # Pre-computed gold SQL results (JSON strings)
    total = len(dataset)

    if verbose:
        print(f"Preprocessing {total} examples (timeout={timeout}s)...")

    start_time = time.time()

    for i in range(total):
        gold_sql = dataset[i]["gold_sql"]
        db_path = dataset[i]["db_path"]

        result, err = execute_sql_safe(gold_sql, db_path, timeout)

        if err:
            if "Timeout" in str(err):
                reasons["timeout"] += 1
            else:
                reasons["error"] += 1
            continue

        if remove_empty and (result is None or len(result) == 0):
            reasons["empty_result"] += 1
            continue

        keep_indices.append(i)
        # Cache the gold result as JSON — avoids re-executing during training
        gold_results.append(json.dumps(result))

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {i + 1}/{total} ({elapsed:.1f}s) — kept {len(keep_indices)}")

    filtered = dataset.select(keep_indices)
    # Add pre-computed gold results as a dataset column
    filtered = filtered.add_column("gold_result", gold_results)

    if verbose:
        elapsed = time.time() - start_time
        removed = total - len(filtered)
        print(f"\nDone in {elapsed:.1f}s")
        print(f"Original:  {total}")
        print(f"Kept:      {len(filtered)}")
        print(f"Removed:   {removed}")
        print(f"  empty_result: {reasons['empty_result']}")
        print(f"  timeout:      {reasons['timeout']}")
        print(f"  error:        {reasons['error']}")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        filtered.save_to_disk(output_path)
        if verbose:
            print(f"Saved filtered dataset to {output_path}")

    return filtered


def main():
    parser = argparse.ArgumentParser(description="Preprocess Spider training data")
    parser.add_argument("--data-dir", default=SPIDER_DATA_DIR, help="Path to Spider data directory")
    parser.add_argument("--timeout", type=int, default=5, help="SQL execution timeout in seconds")
    parser.add_argument("--output", default="data/spider_train_filtered", help="Output directory for filtered dataset")
    parser.add_argument("--no-others", action="store_true", help="Exclude train_others.json")
    args = parser.parse_args()

    dataset = load_spider_split("train", data_dir=args.data_dir, include_others=not args.no_others)
    preprocess_split(dataset, timeout=args.timeout, output_path=args.output)


if __name__ == "__main__":
    main()

"""Error analysis — categorize failure modes from evaluation predictions."""

import argparse
import json
from collections import Counter

from src.rewards.execution import execute_sql_safe, extract_sql
from src.rewards.schema_coverage import extract_tables_and_columns
from src.rewards.syntax import is_valid_sql

FAILURE_CATEGORIES = [
    "wrong_table",
    "wrong_column",
    "wrong_join",
    "wrong_aggregation",
    "wrong_filter",
    "wrong_order",
    "syntax_error",
    "execution_error",
    "empty_result",
    "other",
]


def _classify_single_failure(pred: dict) -> str:
    """Classify a single failed prediction into a failure category."""
    pred_sql = pred.get("pred_sql", "")
    gold_sql = pred.get("gold_sql", "")

    # Syntax error
    if not is_valid_sql(pred_sql):
        return "syntax_error"

    # Execution error
    if pred.get("pred_error"):
        return "execution_error"

    # Extract identifiers from both
    pred_ids = extract_tables_and_columns(pred_sql)
    gold_ids = extract_tables_and_columns(gold_sql)

    gold_upper = gold_sql.upper()
    pred_upper = pred_sql.upper()

    # Wrong table: predicted SQL references tables not in gold (or misses tables)
    # Heuristic: check for FROM/JOIN table identifiers
    gold_tables = set()
    pred_tables = set()
    for token in gold_ids:
        # Rough heuristic: tokens after FROM or JOIN are likely table names
        if token.lower() not in ("select", "from", "where", "and", "or", "on", "as", "join",
                                  "inner", "left", "right", "outer", "group", "by", "order",
                                  "having", "limit", "distinct", "count", "sum", "avg", "max",
                                  "min", "asc", "desc", "not", "in", "like", "between", "null",
                                  "is", "exists", "case", "when", "then", "else", "end", "union",
                                  "except", "intersect", "values", "into", "insert", "update",
                                  "delete", "create", "drop", "alter", "table", "index"):
            gold_tables.add(token)
    for token in pred_ids:
        if token.lower() not in ("select", "from", "where", "and", "or", "on", "as", "join",
                                  "inner", "left", "right", "outer", "group", "by", "order",
                                  "having", "limit", "distinct", "count", "sum", "avg", "max",
                                  "min", "asc", "desc", "not", "in", "like", "between", "null",
                                  "is", "exists", "case", "when", "then", "else", "end", "union",
                                  "except", "intersect", "values", "into", "insert", "update",
                                  "delete", "create", "drop", "alter", "table", "index"):
            pred_tables.add(token)

    if gold_tables != pred_tables:
        missing = gold_tables - pred_tables
        extra = pred_tables - gold_tables
        # If table-level mismatch, classify as wrong_table
        if missing or extra:
            return "wrong_table"

    # Check aggregation issues
    agg_keywords = ["COUNT", "SUM", "AVG", "MAX", "MIN", "GROUP BY", "HAVING"]
    gold_has_agg = any(kw in gold_upper for kw in agg_keywords)
    pred_has_agg = any(kw in pred_upper for kw in agg_keywords)
    if gold_has_agg != pred_has_agg:
        return "wrong_aggregation"
    if gold_has_agg and pred_has_agg:
        # Both have aggregation but results differ — likely wrong aggregation
        return "wrong_aggregation"

    # Check ORDER BY
    if ("ORDER BY" in gold_upper) != ("ORDER BY" in pred_upper):
        return "wrong_order"

    # Check WHERE conditions
    if ("WHERE" in gold_upper) != ("WHERE" in pred_upper):
        return "wrong_filter"

    # Check JOIN
    if ("JOIN" in gold_upper) != ("JOIN" in pred_upper):
        return "wrong_join"

    # Check if predicted result is empty when gold isn't
    db_path = pred.get("db_path")
    if db_path:
        pred_result, _ = execute_sql_safe(pred_sql, db_path)
        gold_result, _ = execute_sql_safe(gold_sql, db_path)
        if pred_result is not None and len(pred_result) == 0 and gold_result and len(gold_result) > 0:
            return "empty_result"

    return "other"


def categorize_failures(predictions_path: str) -> dict[str, list]:
    """
    Load evaluation predictions JSON, categorize each failure.
    Returns dict mapping category -> list of failed prediction dicts.
    """
    with open(predictions_path, "r") as f:
        data = json.load(f)

    preds = data.get("predictions", data) if isinstance(data, dict) else data
    failures = [p for p in preds if not p.get("correct", False)]

    categories = {cat: [] for cat in FAILURE_CATEGORIES}
    for failure in failures:
        cat = _classify_single_failure(failure)
        categories[cat].append(failure)

    return categories


def print_summary(categories: dict[str, list], total_predictions: int = 0):
    """Print failure category distribution."""
    total_failures = sum(len(v) for v in categories.values())
    print(f"\n{'='*60}")
    print(f"Error Analysis Summary")
    print(f"{'='*60}")
    if total_predictions:
        print(f"Total predictions: {total_predictions}")
    print(f"Total failures:    {total_failures}")
    print(f"{'─'*60}")
    print(f"{'Category':<25} {'Count':>8} {'% of failures':>15}")
    print(f"{'─'*60}")

    for cat in FAILURE_CATEGORIES:
        count = len(categories.get(cat, []))
        pct = (count / total_failures * 100) if total_failures > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{cat:<25} {count:>8} {pct:>14.1f}% {bar}")

    print(f"{'─'*60}")

    # Show example failures for top categories
    sorted_cats = sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)
    for cat, failures in sorted_cats[:3]:
        if not failures:
            continue
        print(f"\nTop failures in '{cat}':")
        for f in failures[:2]:
            print(f"  Q: {f.get('question', 'N/A')[:80]}")
            print(f"  Gold: {f.get('gold_sql', 'N/A')[:80]}")
            print(f"  Pred: {f.get('pred_sql', 'N/A')[:80]}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Error analysis on evaluation predictions")
    parser.add_argument("--predictions", required=True, help="Path to evaluation results JSON")
    args = parser.parse_args()

    categories = categorize_failures(args.predictions)

    # Get total count
    with open(args.predictions) as f:
        data = json.load(f)
    total = data.get("total", len(data.get("predictions", [])))

    print_summary(categories, total)


if __name__ == "__main__":
    main()

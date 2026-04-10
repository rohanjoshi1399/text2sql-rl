"""Schema coverage reward — F1 over table and column names between predicted and gold SQL."""

import sqlparse
import sqlparse.tokens as T

from src.rewards.execution import extract_sql, _get_completion_text


def extract_tables_and_columns(sql: str) -> set[str]:
    """
    Extract referenced table and column identifier names from SQL using sqlparse.
    Returns a set of lowercase identifiers.
    """
    identifiers = set()
    sql = sql.strip()
    if not sql:
        return identifiers
    try:
        parsed = sqlparse.parse(sql)
        for stmt in parsed:
            for token in stmt.flatten():
                if token.ttype is T.Name:
                    identifiers.add(str(token).lower())
    except Exception:
        pass
    return identifiers


def compute_f1(predicted: set, gold: set) -> float:
    """Standard F1 over two sets. Returns 1.0 if both are empty (vacuously correct)."""
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0

    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def schema_coverage_reward(
    prompts: list[str],
    completions: list[str],
    gold_sql: list[str],
    **kwargs,
) -> list[float]:
    """
    TRL reward function: schema coverage F1.
    Computes F1 over table+column identifier names between generated and gold SQL.
    """
    rewards = []
    for comp, gold in zip(completions, gold_sql):
        pred_ids = extract_tables_and_columns(extract_sql(_get_completion_text(comp)))
        gold_ids = extract_tables_and_columns(gold)
        rewards.append(compute_f1(pred_ids, gold_ids))
    return rewards

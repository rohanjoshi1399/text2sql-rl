"""Syntax validation reward — checks if generated SQL is parseable via sqlparse."""

import sqlparse


def is_valid_sql(sql: str) -> bool:
    """
    Check if a string is syntactically valid SQL using sqlparse.
    Returns True if it contains a SELECT keyword and parses successfully.
    sqlparse is very permissive, so we require SELECT to be present.
    """
    sql = sql.strip()
    if not sql:
        return False
    # Must contain SELECT keyword (sqlparse alone parses almost anything)
    if "SELECT" not in sql.upper():
        return False
    try:
        parsed = sqlparse.parse(sql)
        if not parsed or not parsed[0].tokens:
            return False
        return True
    except Exception:
        return False


def _get_completion_text(completion) -> str:
    """Extract text from a completion — handles both str and conversation dict formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # List of message dicts — take the last assistant message
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # Fallback: join all content
        return " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def syntax_reward(
    prompts, completions, **kwargs
) -> list[float]:
    """
    TRL reward function signature.
    Returns 1.0 if sqlparse parses the completion as valid SQL, 0.0 otherwise.
    """
    return [1.0 if is_valid_sql(_get_completion_text(c)) else 0.0 for c in completions]

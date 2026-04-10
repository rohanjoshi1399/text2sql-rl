"""Composite reward functions — combine individual rewards into Phase 1 and Phase 2 signals."""

import re

from src.rewards.execution import execution_reward, execution_success_reward, extract_sql, _get_completion_text
from src.rewards.schema_coverage import schema_coverage_reward
from src.rewards.syntax import syntax_reward


def _weighted(fn, weight: float):
    """Wrap a reward function to pre-multiply its output by a weight.
    TRL GRPOTrainer sums all reward functions — weights must be baked in."""

    def wrapped(prompts, completions, **kwargs):
        raw = fn(prompts, completions, **kwargs)
        return [r * weight for r in raw]

    wrapped.__name__ = f"weighted_{fn.__name__}_{weight}"
    return wrapped


def format_compliance_reward(
    prompts, completions, **kwargs
) -> list[float]:
    """
    Returns 1.0 if completion looks like pure SQL output:
    - Starts with SELECT (after stripping whitespace)
    - No markdown code blocks
    - No natural language explanation before the SQL
    """
    rewards = []
    for comp in completions:
        text = _get_completion_text(comp)
        sql = extract_sql(text)
        clean = sql.strip().upper()
        is_compliant = (
            clean.startswith("SELECT")
            and "```" not in text
            and not re.search(r"^(here|the|this|i |let)", text.strip().lower())
        )
        rewards.append(1.0 if is_compliant else 0.0)
    return rewards


def make_phase1_rewards() -> list[callable]:
    """
    Phase 1 (simple) reward functions.
    R = 1.0 * execution_accuracy + 0.2 * syntax_valid
    """
    return [
        execution_reward,
        _weighted(syntax_reward, 0.2),
    ]


def make_phase2_rewards() -> list[callable]:
    """
    Phase 2 (composite) reward functions.
    R = 1.0 * execution_accuracy
      + 0.2 * execution_success
      + 0.2 * syntax_valid
      + 0.2 * schema_coverage_f1
      + 0.1 * format_compliance
    """
    return [
        execution_reward,
        _weighted(execution_success_reward, 0.2),
        _weighted(syntax_reward, 0.2),
        _weighted(schema_coverage_reward, 0.2),
        _weighted(format_compliance_reward, 0.1),
    ]

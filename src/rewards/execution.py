"""Execution-based rewards — run SQL against SQLite, compare result sets with gold."""

import platform
import re
import sqlite3
import threading

TIMEOUT_SECONDS = 5

# ---------------------------------------------------------------------------
# In-memory DB cache — avoids repeated disk reads for the same database.
# Spider has ~140 unique databases; caching them all costs ~50 MB RAM.
# ---------------------------------------------------------------------------
_db_cache: dict[str, sqlite3.Connection] = {}
_db_cache_lock = threading.Lock()


def _get_cached_mem_db(db_path: str) -> sqlite3.Connection:
    """Get or create a cached in-memory copy of a SQLite database."""
    if db_path not in _db_cache:
        with _db_cache_lock:
            if db_path not in _db_cache:  # double-check after acquiring lock
                disk_conn = sqlite3.connect(db_path)
                # check_same_thread=False: the cached conn is read-only (used
                # only as a backup source) and access is serialized by the lock,
                # so cross-thread usage is safe here.
                mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
                disk_conn.backup(mem_conn)
                disk_conn.close()
                _db_cache[db_path] = mem_conn
    return _db_cache[db_path]


def _get_query_connection(db_path: str) -> sqlite3.Connection:
    """
    Get a fresh query connection via fast memory-to-memory backup.
    Each caller gets its own connection for thread safety.
    """
    source = _get_cached_mem_db(db_path)
    query_conn = sqlite3.connect(":memory:")
    source.backup(query_conn)
    return query_conn


def clear_db_cache():
    """Clear all cached database connections. Call at end of training if needed."""
    with _db_cache_lock:
        for conn in _db_cache.values():
            try:
                conn.close()
            except Exception:
                pass
        _db_cache.clear()


def _get_completion_text(completion) -> str:
    """Extract text from a completion — handles both str and conversation dict formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def extract_sql(text: str) -> str:
    """
    Extract SQL from model completion text.
    Handles markdown code blocks, explanations, and trailing semicolons.
    """
    text = text.strip()
    if not text:
        return ""

    # Remove markdown code blocks
    if "```sql" in text.lower():
        match = re.search(r"```sql\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    elif "```" in text:
        match = re.search(r"```\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Take up to first semicolon (remove trailing explanation)
    if ";" in text:
        text = text[: text.index(";")]

    return text.strip()


def _execute_with_timeout_thread(sql: str, db_path: str, timeout: int):
    """Thread-based timeout for Windows compatibility."""
    result = [None]
    error = [None]

    def _run():
        try:
            mem_conn = _get_query_connection(db_path)
            cursor = mem_conn.cursor()
            cursor.execute(sql)
            result[0] = cursor.fetchall()
            mem_conn.close()
        except Exception as e:
            error[0] = str(e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None, "Timeout"
    if error[0]:
        return None, error[0]
    return result[0], None


def _execute_with_timeout_signal(sql: str, db_path: str, timeout: int):
    """Signal-based timeout for Linux (more reliable for CPU-bound queries)."""
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError("Query timed out")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        mem_conn = _get_query_connection(db_path)
        cursor = mem_conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        mem_conn.close()
        return result, None
    except TimeoutError:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def execute_sql_safe(
    sql: str, db_path: str, timeout: int = TIMEOUT_SECONDS
) -> tuple[list | None, str | None]:
    """
    Execute SQL against an in-memory copy of the SQLite database.
    Returns (result_rows, None) on success or (None, error_message) on failure.
    Uses signal-based timeout on Linux, threading-based on Windows.
    """
    sql = sql.strip()
    if not sql:
        return None, "Empty SQL"

    # signal.alarm only works in the main thread. DSPy's MIPROv2 runs
    # evaluation in parallel threads, so fall back to threading-based timeout.
    import threading
    use_signal = (
        platform.system() != "Windows"
        and threading.current_thread() is threading.main_thread()
    )

    if use_signal:
        return _execute_with_timeout_signal(sql, db_path, timeout)
    else:
        return _execute_with_timeout_thread(sql, db_path, timeout)


def _normalize_value(v) -> str:
    """Normalize a single value for comparison."""
    if v is None:
        return "none"
    if isinstance(v, float):
        return str(round(v, 6)).lower()
    return str(v).lower().strip()


def _normalize_rows(rows: list) -> list[tuple]:
    """Normalize all values in result rows."""
    return [tuple(_normalize_value(v) for v in row) for row in rows]


def compare_results(
    gold_results: list, pred_results: list, gold_sql: str
) -> bool:
    """
    Compare two SQL result sets.
    - Normalize all values to lowercase strings (handles type mismatches)
    - Sort both (order-insensitive) UNLESS gold SQL contains ORDER BY
    """
    if gold_results is None or pred_results is None:
        return False

    gold_norm = _normalize_rows(gold_results)
    pred_norm = _normalize_rows(pred_results)

    # If gold SQL has ORDER BY, order matters
    if re.search(r"\bORDER\s+BY\b", gold_sql, re.IGNORECASE):
        return gold_norm == pred_norm
    else:
        return sorted(gold_norm) == sorted(pred_norm)


def execution_reward(
    prompts: list[str],
    completions: list[str],
    gold_sql: list[str],
    db_path: list[str],
    gold_result: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """
    TRL reward function: execution accuracy.
    Executes both generated and gold SQL, compares result sets.
    Returns 1.0 if match, 0.0 otherwise.

    If gold_result is provided (pre-computed JSON from preprocessing),
    skips re-executing gold SQL — a major speedup during GRPO training.
    """
    import json

    rewards = []
    for i, (comp, gold, path) in enumerate(zip(completions, gold_sql, db_path)):
        pred_sql = extract_sql(_get_completion_text(comp))

        # Use pre-computed gold result if available
        if gold_result is not None and gold_result[i] is not None:
            try:
                gold_res = [tuple(row) for row in json.loads(gold_result[i])]
                gold_err = None
            except (json.JSONDecodeError, TypeError):
                gold_res, gold_err = execute_sql_safe(gold, path)
        else:
            gold_res, gold_err = execute_sql_safe(gold, path)

        if gold_err:
            rewards.append(0.0)
            continue

        pred_result, pred_err = execute_sql_safe(pred_sql, path)
        if pred_err:
            rewards.append(0.0)
            continue

        rewards.append(1.0 if compare_results(gold_res, pred_result, gold) else 0.0)
    return rewards


def execution_success_reward(
    prompts: list[str],
    completions: list[str],
    db_path: list[str],
    **kwargs,
) -> list[float]:
    """
    TRL reward function: execution success (partial credit).
    Returns 1.0 if the generated SQL executes without error, 0.0 otherwise.
    Does NOT compare to gold — just checks executability.
    """
    rewards = []
    for comp, path in zip(completions, db_path):
        pred_sql = extract_sql(_get_completion_text(comp))
        _, err = execute_sql_safe(pred_sql, path)
        rewards.append(1.0 if err is None else 0.0)
    return rewards

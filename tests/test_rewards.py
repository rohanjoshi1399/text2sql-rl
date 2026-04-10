"""Unit tests for all reward functions. Runs on Windows without GPU using real Spider SQLite DBs."""

import json
import os
import tempfile
import pytest

from src.rewards.syntax import syntax_reward, is_valid_sql, _get_completion_text
from src.rewards.execution import (
    execution_reward,
    execution_success_reward,
    execute_sql_safe,
    extract_sql,
    compare_results,
    _db_cache,
    _get_cached_mem_db,
    clear_db_cache,
)
from src.training.utils import find_latest_checkpoint
from src.rewards.schema_coverage import (
    schema_coverage_reward,
    extract_tables_and_columns,
    compute_f1,
)
from src.rewards.composite import (
    make_phase1_rewards,
    make_phase2_rewards,
    format_compliance_reward,
)

SPIDER_DATA_DIR = os.path.join("data", "spider_data", "spider_data")
TEST_DB = "department_management"
TEST_DB_PATH = os.path.join(SPIDER_DATA_DIR, "database", TEST_DB, f"{TEST_DB}.sqlite")

# Skip all tests if Spider data is not present
pytestmark = pytest.mark.skipif(
    not os.path.exists(TEST_DB_PATH),
    reason=f"Spider database not found at {TEST_DB_PATH}",
)


# ── SQL Extraction ──────────────────────────────────────────────────────────


class TestSQLExtraction:
    def test_plain_sql(self):
        assert extract_sql("SELECT 1") == "SELECT 1"

    def test_markdown_block(self):
        assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_generic_markdown_block(self):
        assert extract_sql("```\nSELECT 1\n```") == "SELECT 1"

    def test_with_semicolon(self):
        assert extract_sql("SELECT 1; some explanation after") == "SELECT 1"

    def test_with_explanation(self):
        result = extract_sql("Here is the query:\nSELECT name FROM head")
        assert "SELECT" in result

    def test_empty(self):
        assert extract_sql("") == ""

    def test_whitespace_only(self):
        assert extract_sql("   \n  ") == ""


# ── Syntax Reward ───────────────────────────────────────────────────────────


class TestSyntaxReward:
    def test_valid_select(self):
        assert syntax_reward([""], ["SELECT count(*) FROM head WHERE age > 56"]) == [1.0]

    def test_valid_complex(self):
        sql = "SELECT T1.name FROM head AS T1 JOIN management AS T2 ON T1.head_ID = T2.head_ID"
        assert syntax_reward([""], [sql]) == [1.0]

    def test_invalid_garbage(self):
        assert syntax_reward([""], ["hello world not sql"]) == [0.0]

    def test_empty_string(self):
        assert syntax_reward([""], [""]) == [0.0]

    def test_batch(self):
        result = syntax_reward(["", ""], ["SELECT 1", "NOT SQL AT ALL"])
        assert result == [1.0, 0.0]

    def test_is_valid_sql_directly(self):
        assert is_valid_sql("SELECT 1") is True
        assert is_valid_sql("") is False
        assert is_valid_sql("DROP TABLE x") is False


# ── Execution Reward ────────────────────────────────────────────────────────


class TestExecuteSQLSafe:
    def test_simple_query(self):
        result, err = execute_sql_safe("SELECT count(*) FROM head", TEST_DB_PATH)
        assert err is None
        assert result is not None
        assert len(result) == 1

    def test_nonexistent_table(self):
        result, err = execute_sql_safe("SELECT * FROM nonexistent", TEST_DB_PATH)
        assert result is None
        assert err is not None

    def test_empty_sql(self):
        result, err = execute_sql_safe("", TEST_DB_PATH)
        assert result is None
        assert "Empty SQL" in err

    def test_syntax_error(self):
        result, err = execute_sql_safe("SELEKT * FROM head", TEST_DB_PATH)
        assert result is None
        assert err is not None


class TestCompareResults:
    def test_identical(self):
        rows = [(1, "Alice"), (2, "Bob")]
        assert compare_results(rows, rows, "SELECT id, name FROM t") is True

    def test_different_order_no_order_by(self):
        gold = [(1, "Alice"), (2, "Bob")]
        pred = [(2, "Bob"), (1, "Alice")]
        assert compare_results(gold, pred, "SELECT id, name FROM t") is True

    def test_different_order_with_order_by(self):
        gold = [(1, "Alice"), (2, "Bob")]
        pred = [(2, "Bob"), (1, "Alice")]
        assert compare_results(gold, pred, "SELECT id, name FROM t ORDER BY id") is False

    def test_different_values(self):
        gold = [(1,)]
        pred = [(2,)]
        assert compare_results(gold, pred, "SELECT count(*) FROM t") is False

    def test_empty_both(self):
        assert compare_results([], [], "SELECT * FROM t") is True

    def test_none_input(self):
        assert compare_results(None, [(1,)], "SELECT 1") is False
        assert compare_results([(1,)], None, "SELECT 1") is False


class TestExecutionReward:
    def test_correct_sql_matches_itself(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        r = execution_reward([""], [gold], gold_sql=[gold], db_path=[TEST_DB_PATH])
        assert r == [1.0]

    def test_wrong_sql(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        pred = "SELECT count(*) FROM head"  # counts all, not just age > 56
        r = execution_reward([""], [pred], gold_sql=[gold], db_path=[TEST_DB_PATH])
        assert r == [0.0]

    def test_invalid_sql_returns_zero(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        pred = "SELECT * FROM nonexistent_table"
        r = execution_reward([""], [pred], gold_sql=[gold], db_path=[TEST_DB_PATH])
        assert r == [0.0]

    def test_equivalent_sql(self):
        gold = "SELECT name FROM department"
        pred = "SELECT Name FROM department"  # same query, different case
        r = execution_reward([""], [pred], gold_sql=[gold], db_path=[TEST_DB_PATH])
        assert r == [1.0]

    def test_batch(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        r = execution_reward(
            ["", ""],
            [gold, "SELECT 999"],
            gold_sql=[gold, gold],
            db_path=[TEST_DB_PATH, TEST_DB_PATH],
        )
        assert r[0] == 1.0
        assert r[1] == 0.0


class TestExecutionSuccessReward:
    def test_executable(self):
        r = execution_success_reward(
            [""], ["SELECT 1"], db_path=[TEST_DB_PATH]
        )
        assert r == [1.0]

    def test_not_executable(self):
        r = execution_success_reward(
            [""], ["SELECT * FROM fake_table"], db_path=[TEST_DB_PATH]
        )
        assert r == [0.0]


# ── Schema Coverage Reward ──────────────────────────────────────────────────


class TestExtractTablesAndColumns:
    def test_simple(self):
        ids = extract_tables_and_columns("SELECT count(*) FROM head WHERE age > 56")
        assert "head" in ids
        assert "age" in ids

    def test_join(self):
        sql = "SELECT T1.name FROM head AS T1 JOIN management AS T2 ON T1.head_ID = T2.head_ID"
        ids = extract_tables_and_columns(sql)
        assert "head" in ids
        assert "management" in ids
        assert "name" in ids

    def test_empty(self):
        assert extract_tables_and_columns("") == set()


class TestComputeF1:
    def test_identical(self):
        s = {"head", "age", "name"}
        assert compute_f1(s, s) == 1.0

    def test_no_overlap(self):
        assert compute_f1({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial(self):
        pred = {"head", "name"}
        gold = {"head", "name", "age"}
        f1 = compute_f1(pred, gold)
        assert 0.0 < f1 < 1.0

    def test_both_empty(self):
        assert compute_f1(set(), set()) == 1.0


class TestSchemaCoverageReward:
    def test_exact_match(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        r = schema_coverage_reward([""], [gold], gold_sql=[gold])
        assert r == [1.0]

    def test_partial_match(self):
        gold = "SELECT name FROM head WHERE age > 56"
        pred = "SELECT name FROM head"
        r = schema_coverage_reward([""], [pred], gold_sql=[gold])
        assert 0.0 < r[0] < 1.0

    def test_no_match(self):
        gold = "SELECT name FROM head"
        pred = "SELECT Budget_in_Billions FROM department"
        r = schema_coverage_reward([""], [pred], gold_sql=[gold])
        assert r[0] < 0.5


# ── Format Compliance ───────────────────────────────────────────────────────


class TestFormatCompliance:
    def test_clean_sql(self):
        r = format_compliance_reward([""], ["SELECT count(*) FROM head"])
        assert r == [1.0]

    def test_with_explanation(self):
        r = format_compliance_reward([""], ["Here is the SQL: SELECT count(*) FROM head"])
        assert r == [0.0]

    def test_with_markdown(self):
        r = format_compliance_reward([""], ["```sql\nSELECT 1\n```"])
        assert r == [0.0]


# ── Composite Reward ────────────────────────────────────────────────────────


class TestCompositeReward:
    def test_phase1_returns_correct_count(self):
        rewards = make_phase1_rewards()
        assert len(rewards) == 2

    def test_phase2_returns_correct_count(self):
        rewards = make_phase2_rewards()
        assert len(rewards) == 5

    def test_phase1_correct_sql(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        rewards = make_phase1_rewards()
        total = 0.0
        for fn in rewards:
            r = fn([""], [gold], gold_sql=[gold], db_path=[TEST_DB_PATH])
            assert len(r) == 1
            assert isinstance(r[0], float)
            total += r[0]
        # Correct SQL: 1.0 (exec) + 0.2 (syntax) = 1.2
        assert abs(total - 1.2) < 0.01

    def test_phase1_wrong_sql(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        pred = "SELECT count(*) FROM head"
        rewards = make_phase1_rewards()
        total = 0.0
        for fn in rewards:
            r = fn([""], [pred], gold_sql=[gold], db_path=[TEST_DB_PATH])
            total += r[0]
        # Wrong SQL: 0.0 (exec) + 0.2 (syntax) = 0.2
        assert abs(total - 0.2) < 0.01

    def test_phase2_correct_sql(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        rewards = make_phase2_rewards()
        total = 0.0
        for fn in rewards:
            r = fn([""], [gold], gold_sql=[gold], db_path=[TEST_DB_PATH])
            total += r[0]
        # Correct SQL: 1.0 + 0.2 + 0.2 + 0.2 + 0.1 = 1.7
        assert abs(total - 1.7) < 0.01


# ── Conversational Format (TRL 0.14.0) ─────────────────────────────────────
# TRL 0.14.0 passes completions as list[list[dict]] for conversational datasets


def _wrap_completion(sql: str) -> list[dict]:
    """Wrap a SQL string as a TRL 0.14.0 conversational completion."""
    return [{"role": "assistant", "content": sql}]


def _wrap_prompt() -> list[dict]:
    """Dummy conversational prompt."""
    return [{"role": "system", "content": "You are a SQL expert."},
            {"role": "user", "content": "Write SQL"}]


class TestGetCompletionText:
    def test_string_passthrough(self):
        assert _get_completion_text("SELECT 1") == "SELECT 1"

    def test_conversation_dict(self):
        comp = [{"role": "assistant", "content": "SELECT 1"}]
        assert _get_completion_text(comp) == "SELECT 1"

    def test_empty_conversation(self):
        assert _get_completion_text([]) == ""

    def test_multi_message_takes_last_assistant(self):
        comp = [
            {"role": "user", "content": "ignored"},
            {"role": "assistant", "content": "SELECT 1"},
        ]
        assert _get_completion_text(comp) == "SELECT 1"


class TestConversationalSyntaxReward:
    def test_valid_sql_as_dict(self):
        comp = _wrap_completion("SELECT count(*) FROM head")
        r = syntax_reward([_wrap_prompt()], [comp])
        assert r == [1.0]

    def test_invalid_sql_as_dict(self):
        comp = _wrap_completion("not sql at all")
        r = syntax_reward([_wrap_prompt()], [comp])
        assert r == [0.0]


class TestConversationalExecutionReward:
    def test_correct_sql_as_dict(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        comp = _wrap_completion(gold)
        r = execution_reward(
            [_wrap_prompt()], [comp],
            gold_sql=[gold], db_path=[TEST_DB_PATH],
        )
        assert r == [1.0]

    def test_wrong_sql_as_dict(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        comp = _wrap_completion("SELECT count(*) FROM head")
        r = execution_reward(
            [_wrap_prompt()], [comp],
            gold_sql=[gold], db_path=[TEST_DB_PATH],
        )
        assert r == [0.0]


class TestConversationalSchemaCoverage:
    def test_exact_match_as_dict(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        comp = _wrap_completion(gold)
        r = schema_coverage_reward([_wrap_prompt()], [comp], gold_sql=[gold])
        assert r == [1.0]


class TestConversationalFormatCompliance:
    def test_clean_sql_as_dict(self):
        comp = _wrap_completion("SELECT count(*) FROM head")
        r = format_compliance_reward([_wrap_prompt()], [comp])
        assert r == [1.0]

    def test_explanation_as_dict(self):
        comp = _wrap_completion("Here is the SQL: SELECT count(*) FROM head")
        r = format_compliance_reward([_wrap_prompt()], [comp])
        assert r == [0.0]


class TestConversationalComposite:
    def test_phase1_correct_as_dict(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        comp = _wrap_completion(gold)
        rewards = make_phase1_rewards()
        total = 0.0
        for fn in rewards:
            r = fn([_wrap_prompt()], [comp], gold_sql=[gold], db_path=[TEST_DB_PATH])
            assert len(r) == 1
            total += r[0]
        assert abs(total - 1.2) < 0.01

    def test_phase2_correct_as_dict(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        comp = _wrap_completion(gold)
        rewards = make_phase2_rewards()
        total = 0.0
        for fn in rewards:
            r = fn([_wrap_prompt()], [comp], gold_sql=[gold], db_path=[TEST_DB_PATH])
            total += r[0]
        assert abs(total - 1.7) < 0.01


# ── DB Cache ───────────────────────────────────────────────────────────────


class TestDBCache:
    def setup_method(self):
        clear_db_cache()

    def teardown_method(self):
        clear_db_cache()

    def test_cache_hit(self):
        """Second call should return the same cached connection object."""
        conn1 = _get_cached_mem_db(TEST_DB_PATH)
        conn2 = _get_cached_mem_db(TEST_DB_PATH)
        assert conn1 is conn2

    def test_cache_populated(self):
        _get_cached_mem_db(TEST_DB_PATH)
        assert TEST_DB_PATH in _db_cache

    def test_clear_cache(self):
        _get_cached_mem_db(TEST_DB_PATH)
        clear_db_cache()
        assert len(_db_cache) == 0

    def test_execute_still_works_with_cache(self):
        """execute_sql_safe should work correctly using cached DB."""
        result, err = execute_sql_safe("SELECT count(*) FROM head", TEST_DB_PATH)
        assert err is None
        assert result is not None
        # Second call uses cache
        result2, err2 = execute_sql_safe("SELECT count(*) FROM head", TEST_DB_PATH)
        assert err2 is None
        assert result == result2


# ── Gold Result Caching ────────────────────────────────────────────────────


class TestGoldResultCaching:
    def test_with_precomputed_gold_result(self):
        """execution_reward with pre-computed gold_result should match live execution."""
        gold = "SELECT count(*) FROM head WHERE age > 56"
        # Get the actual gold result
        gold_res, _ = execute_sql_safe(gold, TEST_DB_PATH)
        gold_result_json = json.dumps(gold_res)

        # Test with pre-computed result
        r = execution_reward(
            [""], [gold],
            gold_sql=[gold],
            db_path=[TEST_DB_PATH],
            gold_result=[gold_result_json],
        )
        assert r == [1.0]

    def test_with_wrong_pred_and_precomputed_gold(self):
        gold = "SELECT count(*) FROM head WHERE age > 56"
        gold_res, _ = execute_sql_safe(gold, TEST_DB_PATH)
        gold_result_json = json.dumps(gold_res)

        pred = "SELECT count(*) FROM head"
        r = execution_reward(
            [""], [pred],
            gold_sql=[gold],
            db_path=[TEST_DB_PATH],
            gold_result=[gold_result_json],
        )
        assert r == [0.0]

    def test_falls_back_when_gold_result_is_none(self):
        """When gold_result is None, should fall back to live execution."""
        gold = "SELECT count(*) FROM head WHERE age > 56"
        r = execution_reward(
            [""], [gold],
            gold_sql=[gold],
            db_path=[TEST_DB_PATH],
            gold_result=None,
        )
        assert r == [1.0]

    def test_falls_back_on_invalid_json(self):
        """Malformed JSON should fall back to live execution, not crash."""
        gold = "SELECT count(*) FROM head WHERE age > 56"
        r = execution_reward(
            [""], [gold],
            gold_sql=[gold],
            db_path=[TEST_DB_PATH],
            gold_result=["not valid json {{{"],
        )
        assert r == [1.0]


# ── find_latest_checkpoint ─────────────────────────────────────────────────


class TestFindLatestCheckpoint:
    def test_no_directory(self):
        assert find_latest_checkpoint("/nonexistent/path/xyz") is None

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert find_latest_checkpoint(tmpdir) is None

    def test_finds_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "checkpoint-100"))
            os.makedirs(os.path.join(tmpdir, "checkpoint-300"))
            os.makedirs(os.path.join(tmpdir, "checkpoint-200"))
            result = find_latest_checkpoint(tmpdir)
            assert result.endswith("checkpoint-300")

    def test_ignores_non_checkpoint_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "checkpoint-100"))
            os.makedirs(os.path.join(tmpdir, "best"))
            os.makedirs(os.path.join(tmpdir, "logs"))
            result = find_latest_checkpoint(tmpdir)
            assert result.endswith("checkpoint-100")

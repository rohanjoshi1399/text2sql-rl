"""Spider dataset loader — loads JSONs, extracts schema DDL, formats for Llama chat template."""

import json
import os
import sqlite3
from pathlib import Path

from datasets import Dataset

SPIDER_DATA_DIR = os.path.join("data", "spider_data", "spider_data")

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "generate a valid SQLite query that answers the question.\n"
    "Return only the SQL query, with no explanation or markdown formatting."
)

# Max characters for schema DDL to avoid exceeding model context
MAX_SCHEMA_CHARS = 3500


def get_db_path(db_id: str, data_dir: str = SPIDER_DATA_DIR, split: str = "train") -> str:
    """Return absolute path to the .sqlite file for a given db_id."""
    if split == "test":
        db_dir = os.path.join(data_dir, "test_database", db_id)
    else:
        db_dir = os.path.join(data_dir, "database", db_id)
    return os.path.abspath(os.path.join(db_dir, f"{db_id}.sqlite"))


def get_schema_ddl(db_id: str, data_dir: str = SPIDER_DATA_DIR, split: str = "train") -> str:
    """
    Extract CREATE TABLE statements from the SQLite database.
    Returns concatenated DDL with column types, PRIMARY KEY, FOREIGN KEY constraints.
    """
    db_path = get_db_path(db_id, data_dir, split)
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        # Fallback: try reading schema.sql
        return _read_schema_sql(db_id, data_dir, split)

    ddl = "\n\n".join(row[0].strip() for row in rows if row[0])

    if len(ddl) > MAX_SCHEMA_CHARS:
        ddl = ddl[:MAX_SCHEMA_CHARS] + "\n-- (schema truncated)"

    return ddl


def _read_schema_sql(db_id: str, data_dir: str, split: str) -> str:
    """Fallback: read schema.sql file if sqlite_master extraction fails."""
    if split == "test":
        schema_path = os.path.join(data_dir, "test_database", db_id, "schema.sql")
    else:
        schema_path = os.path.join(data_dir, "database", db_id, "schema.sql")

    if not os.path.exists(schema_path):
        return ""

    with open(schema_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract only CREATE TABLE statements
    lines = []
    in_create = False
    for line in content.split("\n"):
        if line.strip().upper().startswith("CREATE TABLE"):
            in_create = True
        if in_create:
            lines.append(line)
            if ";" in line:
                in_create = False
                lines.append("")

    ddl = "\n".join(lines).strip()
    if len(ddl) > MAX_SCHEMA_CHARS:
        ddl = ddl[:MAX_SCHEMA_CHARS] + "\n-- (schema truncated)"
    return ddl


def format_prompt(question: str, schema_ddl: str) -> list[dict]:
    """
    Format a single example into Llama 3.1 Instruct chat messages.
    Returns list of message dicts for tokenizer.apply_chat_template().
    """
    user_content = f"### Database Schema:\n{schema_ddl}\n\n### Question:\n{question}\n\n### SQL:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _load_json(filepath: str) -> list[dict]:
    """Load a Spider JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_spider_split(
    split: str = "train",
    data_dir: str = SPIDER_DATA_DIR,
    include_others: bool = True,
) -> Dataset:
    """
    Load a Spider split and return a HuggingFace Dataset.

    Columns:
      - prompt: list[dict]  (chat messages for tokenizer.apply_chat_template)
      - gold_sql: str       (ground truth SQL)
      - db_id: str          (database identifier)
      - db_path: str        (absolute path to .sqlite file)
      - question: str       (natural language question)

    For train split: combines train_spider.json (7000) + train_others.json (1659).
    For dev: loads dev.json (1034).
    For test: loads test.json (2147), uses test_database/ paths.
    """
    data_dir = str(Path(data_dir).resolve()) if not os.path.isabs(data_dir) else data_dir

    if split == "train":
        examples = _load_json(os.path.join(data_dir, "train_spider.json"))
        if include_others:
            others_path = os.path.join(data_dir, "train_others.json")
            if os.path.exists(others_path):
                examples.extend(_load_json(others_path))
    elif split == "dev":
        examples = _load_json(os.path.join(data_dir, "dev.json"))
    elif split == "test":
        examples = _load_json(os.path.join(data_dir, "test.json"))
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'dev', or 'test'.")

    # Cache DDL per db_id to avoid redundant sqlite reads
    ddl_cache: dict[str, str] = {}

    records = []
    for ex in examples:
        db_id = ex["db_id"]
        if db_id not in ddl_cache:
            ddl_cache[db_id] = get_schema_ddl(db_id, data_dir, split)

        db_path = get_db_path(db_id, data_dir, split)
        prompt = format_prompt(ex["question"], ddl_cache[db_id])

        records.append({
            "prompt": prompt,
            "gold_sql": ex["query"],
            "db_id": db_id,
            "db_path": db_path,
            "question": ex["question"],
        })

    return Dataset.from_list(records)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and inspect Spider data")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--data-dir", default=SPIDER_DATA_DIR)
    parser.add_argument("--show", type=int, default=3, help="Number of examples to print")
    args = parser.parse_args()

    ds = load_spider_split(args.split, args.data_dir)
    print(f"Loaded {len(ds)} examples from {args.split} split")
    print(f"Columns: {ds.column_names}")
    print()

    for i in range(min(args.show, len(ds))):
        ex = ds[i]
        print(f"--- Example {i} ---")
        print(f"DB: {ex['db_id']}")
        print(f"Question: {ex['question']}")
        print(f"Gold SQL: {ex['gold_sql']}")
        print(f"DB Path: {ex['db_path']}")
        print(f"Prompt messages: {len(ex['prompt'])} messages")
        print()

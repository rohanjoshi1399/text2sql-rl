"""DSPy signature definitions for Text-to-SQL."""

import dspy

from src.data.spider_loader import SPIDER_DATA_DIR, get_schema_ddl, load_spider_split


class Text2SQL(dspy.Signature):
    """Given a database schema and a natural language question, generate a SQLite query."""

    question: str = dspy.InputField(desc="Natural language question about the database")
    db_schema: str = dspy.InputField(desc="Database schema as CREATE TABLE statements")
    sql_query: str = dspy.OutputField(desc="SQLite SELECT query that answers the question")


class Text2SQLWithReasoning(dspy.Signature):
    """Given a database schema and a natural language question, reason step by step about which tables, columns, and joins to use, then generate a SQLite query."""

    question: str = dspy.InputField(desc="Natural language question about the database")
    db_schema: str = dspy.InputField(desc="Database schema as CREATE TABLE statements")
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning about tables, joins, and filters needed"
    )
    sql_query: str = dspy.OutputField(desc="SQLite SELECT query that answers the question")


def build_dspy_examples(
    data_dir: str = SPIDER_DATA_DIR,
    split: str = "train",
    max_examples: int | None = None,
) -> list[dspy.Example]:
    """
    Build DSPy Example objects from Spider data.
    Each Example has: question, db_schema, sql_query (label), plus db_id and db_path for eval.
    """
    dataset = load_spider_split(split, data_dir=data_dir)

    examples = []
    for i in range(len(dataset)):
        row = dataset[i]
        ex = dspy.Example(
            question=row["question"],
            db_schema=get_schema_ddl(row["db_id"], data_dir, split),
            sql_query=row["gold_sql"],
            db_id=row["db_id"],
            db_path=row["db_path"],
        ).with_inputs("question", "db_schema")
        examples.append(ex)

        if max_examples and len(examples) >= max_examples:
            break

    return examples

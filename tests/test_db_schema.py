from app.db.schema import build_schema_sql


def test_build_schema_sql_contains_required_tables_and_vector_columns() -> None:
    sql = build_schema_sql(embedding_dim=1024)

    for table in [
        "docs",
        "doc_chunks",
        "conversations",
        "messages",
        "task_runs",
        "tool_calls",
        "conversation_summaries",
    ]:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql

    assert "content_embedding vector(1024)" in sql
    assert "metadata_embedding vector(1024)" in sql
    assert "node_type text NOT NULL DEFAULT 'section'" in sql
    assert "structural_only boolean NOT NULL DEFAULT false" in sql
    assert "content_tsv tsvector" in sql
    assert "metadata_tsv tsvector" in sql

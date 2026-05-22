from app.memory.store import build_insert_conversation_summary_sql


def test_build_insert_conversation_summary_sql_targets_summary_table() -> None:
    sql = build_insert_conversation_summary_sql()

    assert "INSERT INTO conversation_summaries" in sql
    assert "removed_message_count" in sql

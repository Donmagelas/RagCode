from __future__ import annotations

from uuid import uuid4

import psycopg


class PostgresConversationSummaryStore:
    def __init__(self, *, database_url: str, connect_timeout_seconds: int = 5) -> None:
        self._database_url = database_url
        self._connect_timeout_seconds = connect_timeout_seconds

    def __call__(
        self,
        *,
        conversation_id: str,
        summary: str,
        removed_message_count: int,
    ) -> None:
        """把 compact_context 产生的摘要写入业务表，供恢复和审计使用。"""
        with psycopg.connect(
            _psycopg_url(self._database_url),
            autocommit=True,
            connect_timeout=self._connect_timeout_seconds,
        ) as conn:
            conn.execute(
                build_insert_conversation_summary_sql(),
                {
                    "id": f"summary_{uuid4().hex}",
                    "conversation_id": conversation_id,
                    "summary": summary,
                    "removed_message_count": removed_message_count,
                },
            )


def build_insert_conversation_summary_sql() -> str:
    """集中维护 summary 入库 SQL，便于单测和后续迁移检查。"""
    return """
        INSERT INTO conversation_summaries (
            id, conversation_id, summary, removed_message_count
        )
        VALUES (
            %(id)s, %(conversation_id)s, %(summary)s, %(removed_message_count)s
        )
    """


def _psycopg_url(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)

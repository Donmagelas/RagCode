from __future__ import annotations

from alembic import op

from app.db.schema import build_schema_sql

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(build_schema_sql(embedding_dim=1024))


def downgrade() -> None:
    for table in [
        "conversation_summaries",
        "tool_calls",
        "task_runs",
        "messages",
        "conversations",
        "doc_chunks",
        "docs",
    ]:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

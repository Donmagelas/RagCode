from __future__ import annotations

from alembic import op

revision = "0003_chunk_token_count"
down_revision = "0002_chunk_structural_fields"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 保存 ingest 阶段的 chunk token 统计，便于报告和后续调参。
    op.execute("ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS token_count integer NOT NULL DEFAULT 0")


def downgrade() -> None:
    op.execute("ALTER TABLE doc_chunks DROP COLUMN IF EXISTS token_count")

from __future__ import annotations

from alembic import op

revision = "0002_chunk_structural_fields"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 为长节点切分后的结构透传节点补充显式标记。
    op.execute("ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS node_type text NOT NULL DEFAULT 'section'")
    op.execute(
        "ALTER TABLE doc_chunks ADD COLUMN IF NOT EXISTS structural_only boolean NOT NULL DEFAULT false"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE doc_chunks DROP COLUMN IF EXISTS structural_only")
    op.execute("ALTER TABLE doc_chunks DROP COLUMN IF EXISTS node_type")

from __future__ import annotations


def build_schema_sql(*, embedding_dim: int) -> str:
    """生成第一阶段所需 schema；向量维度由配置控制。"""
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS docs (
    id text PRIMARY KEY,
    skill_name text NOT NULL UNIQUE,
    description text NOT NULL DEFAULT '',
    framework_name text NOT NULL DEFAULT '',
    framework_version text NOT NULL DEFAULT '',
    file_path text NOT NULL,
    title text NOT NULL DEFAULT '',
    content_hash text NOT NULL,
    indexed_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS doc_chunks (
    id text PRIMARY KEY,
    doc_id text NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    framework_name text NOT NULL DEFAULT '',
    framework_version text NOT NULL DEFAULT '',
    file_path text NOT NULL,
    heading text NOT NULL,
    heading_level integer NOT NULL,
    heading_path jsonb NOT NULL DEFAULT '[]'::jsonb,
    sort_order integer NOT NULL,
    node_type text NOT NULL DEFAULT 'section',
    structural_only boolean NOT NULL DEFAULT false,
    parent_id text NULL,
    prev_sibling_id text NULL,
    next_sibling_id text NULL,
    own_content text NOT NULL DEFAULT '',
    raw_markdown text NOT NULL DEFAULT '',
    metadata_json jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    metadata_text text NOT NULL DEFAULT '',
    content_embedding vector({embedding_dim}),
    metadata_embedding vector({embedding_dim}),
    content_tsv tsvector,
    metadata_tsv tsvector,
    content_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_sort
    ON doc_chunks(doc_id, sort_order);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_parent
    ON doc_chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_content_tsv
    ON doc_chunks USING gin(content_tsv);
CREATE INDEX IF NOT EXISTS idx_doc_chunks_metadata_tsv
    ON doc_chunks USING gin(metadata_tsv);

CREATE TABLE IF NOT EXISTS conversations (
    id text PRIMARY KEY,
    workspace_path text NOT NULL,
    status text NOT NULL DEFAULT 'active',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id text PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role text NOT NULL,
    content text NOT NULL,
    token_count integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS task_runs (
    id text PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_goal text NOT NULL,
    selected_skills jsonb NOT NULL DEFAULT '[]'::jsonb,
    approved_chunks jsonb NOT NULL DEFAULT '[]'::jsonb,
    files_read jsonb NOT NULL DEFAULT '[]'::jsonb,
    files_changed jsonb NOT NULL DEFAULT '[]'::jsonb,
    commands_run jsonb NOT NULL DEFAULT '[]'::jsonb,
    test_results jsonb NOT NULL DEFAULT '[]'::jsonb,
    status text NOT NULL DEFAULT 'running',
    final_summary text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id text PRIMARY KEY,
    task_run_id text NOT NULL REFERENCES task_runs(id) ON DELETE CASCADE,
    tool_name text NOT NULL,
    tool_input jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    tool_output text NOT NULL DEFAULT '',
    is_error boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id text PRIMARY KEY,
    conversation_id text NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    summary text NOT NULL,
    removed_message_count integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);
""".strip()

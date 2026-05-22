from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.models.embeddings import format_pgvector
from app.models.metadata_extractor import MetadataExtractor
from app.rag.ingest import _metadata_json_to_text


@dataclass(frozen=True)
class MetadataTarget:
    id: str
    metadata_json: dict[str, Any]
    raw_markdown: str
    heading_path: list[str]


@dataclass(frozen=True)
class MetadataEnrichResult:
    selected: int
    enriched: int
    failed: int
    embedded: int


def enrich_missing_metadata(
    database_url: str,
    *,
    api_key: str,
    base_url: str,
    chat_model: str,
    embedding_client: Any,
    skill_name: str | None = None,
    limit: int | None = None,
    workers: int = 4,
    connect_timeout_seconds: int = 5,
    on_progress: Callable[[str], None] | None = None,
) -> MetadataEnrichResult:
    """为数据库中缺失语义 metadata 的正文 chunk 做可恢复增强。"""
    targets = load_metadata_targets(
        database_url,
        skill_name=skill_name,
        limit=limit,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    if on_progress is not None:
        on_progress(f"metadata targets={len(targets)}")
    if not targets:
        embedded = embed_missing_metadata_vectors(
            database_url,
            embedding_client=embedding_client,
            connect_timeout_seconds=connect_timeout_seconds,
        )
        return MetadataEnrichResult(selected=0, enriched=0, failed=0, embedded=embedded)

    enriched_ids: list[str] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = {
            executor.submit(
                _extract_one,
                target,
                api_key=api_key,
                base_url=base_url,
                chat_model=chat_model,
            ): target
            for target in targets
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            target = futures[future]
            try:
                metadata = future.result()
                update_chunk_metadata(
                    database_url,
                    target=target,
                    extracted_metadata=metadata,
                    connect_timeout_seconds=connect_timeout_seconds,
                )
                enriched_ids.append(target.id)
                if on_progress is not None:
                    on_progress(
                        f"metadata {completed}/{len(targets)} ok: {' > '.join(target.heading_path)}"
                    )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if on_progress is not None:
                    on_progress(
                        f"metadata {completed}/{len(targets)} failed: "
                        f"{' > '.join(target.heading_path)} ({exc})"
                    )

    embedded = embed_missing_metadata_vectors(
        database_url,
        embedding_client=embedding_client,
        chunk_ids=enriched_ids,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    return MetadataEnrichResult(
        selected=len(targets),
        enriched=len(enriched_ids),
        failed=failed,
        embedded=embedded,
    )


def load_metadata_targets(
    database_url: str,
    *,
    skill_name: str | None = None,
    limit: int | None = None,
    connect_timeout_seconds: int = 5,
) -> list[MetadataTarget]:
    """读取需要 LLM 语义 metadata 的正文 chunk；结构节点不参与。"""
    where = [
        "NOT c.structural_only",
        "NOT " + semantic_metadata_sql("c.metadata_json"),
    ]
    params: dict[str, object] = {}
    if skill_name:
        where.append("d.skill_name = %(skill_name)s")
        params["skill_name"] = skill_name
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT %(limit)s"
        params["limit"] = limit

    with psycopg.connect(
        _psycopg_url(database_url),
        row_factory=dict_row,
        connect_timeout=connect_timeout_seconds,
    ) as conn:
        rows = conn.execute(
            f"""
            SELECT c.id, c.metadata_json, c.raw_markdown, c.heading_path
            FROM doc_chunks c
            JOIN docs d ON d.id = c.doc_id
            WHERE {" AND ".join(where)}
            ORDER BY d.skill_name, c.sort_order
            {limit_sql}
            """,
            params,
        ).fetchall()
    return [
        MetadataTarget(
            id=str(row["id"]),
            metadata_json=dict(row["metadata_json"]),
            raw_markdown=str(row["raw_markdown"]),
            heading_path=list(row["heading_path"]),
        )
        for row in rows
    ]


def update_chunk_metadata(
    database_url: str,
    *,
    target: MetadataTarget,
    extracted_metadata: dict[str, Any],
    connect_timeout_seconds: int = 5,
) -> None:
    metadata_json = {**target.metadata_json, **extracted_metadata}
    metadata_text = _metadata_json_to_text(metadata_json)
    with psycopg.connect(
        _psycopg_url(database_url),
        autocommit=True,
        connect_timeout=connect_timeout_seconds,
    ) as conn:
        conn.execute(
            """
            UPDATE doc_chunks
            SET metadata_json = %(metadata_json)s::jsonb,
                metadata_text = %(metadata_text)s,
                metadata_embedding = NULL,
                metadata_tsv = to_tsvector('simple', %(metadata_text)s),
                updated_at = now()
            WHERE id = %(id)s
            """,
            {
                "id": target.id,
                "metadata_json": json.dumps(metadata_json, ensure_ascii=False),
                "metadata_text": metadata_text,
            },
        )


def embed_missing_metadata_vectors(
    database_url: str,
    *,
    embedding_client: Any,
    chunk_ids: list[str] | None = None,
    connect_timeout_seconds: int = 5,
) -> int:
    """为 metadata_text 已更新但 metadata_embedding 为空的 chunk 补向量。"""
    params: dict[str, object] = {}
    where = ["metadata_embedding IS NULL"]
    if chunk_ids is not None:
        if not chunk_ids:
            return 0
        where.append("id = ANY(%(chunk_ids)s)")
        params["chunk_ids"] = chunk_ids
    with psycopg.connect(
        _psycopg_url(database_url),
        row_factory=dict_row,
        connect_timeout=connect_timeout_seconds,
    ) as conn:
        rows = conn.execute(
            f"""
            SELECT id, metadata_text
            FROM doc_chunks
            WHERE {" AND ".join(where)}
            ORDER BY id
            """,
            params,
        ).fetchall()
        if not rows:
            return 0
        embeddings = embedding_client.embed_texts([str(row["metadata_text"]) for row in rows])
        with conn.transaction():
            for row, embedding in zip(rows, embeddings, strict=True):
                conn.execute(
                    """
                    UPDATE doc_chunks
                    SET metadata_embedding = %(embedding)s::vector,
                        updated_at = now()
                    WHERE id = %(id)s
                    """,
                    {"id": row["id"], "embedding": format_pgvector(embedding)},
                )
    return len(rows)


def semantic_metadata_sql(column: str) -> str:
    """生成判断语义 metadata 是否存在的 SQL 片段。"""
    return f"""
    (
        coalesce({column}->>'summary', '') <> ''
        OR coalesce({column}->>'api_name', '') <> ''
        OR coalesce({column}->>'module_type', '') <> ''
        OR jsonb_array_length(coalesce({column}->'searchable_keywords', '[]'::jsonb)) > 0
    )
    """


def _extract_one(
    target: MetadataTarget,
    *,
    api_key: str,
    base_url: str,
    chat_model: str,
) -> dict[str, Any]:
    extractor = MetadataExtractor(
        api_key=api_key,
        base_url=base_url,
        model=chat_model,
        batch_size=1,
    )
    return extractor.extract(target.raw_markdown, target.heading_path)


def _psycopg_url(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)

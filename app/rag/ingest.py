from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.rag.markdown_parser import parse_markdown_document_with_report
from app.rag.token_counter import TokenCounter
from app.routing.skill_router import SkillManifest, discover_skill_manifests


@dataclass(frozen=True)
class IngestRecords:
    docs: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    warnings: list[dict[str, Any]]


@dataclass(frozen=True)
class SkillIngestReport:
    skill_name: str
    file_path: str
    chunk_count: int
    section_count: int
    part_count: int
    structural_only_count: int
    max_chunk_tokens: int
    avg_chunk_tokens: float
    warning_count: int


@dataclass(frozen=True)
class IngestReport:
    doc_count: int
    chunk_count: int
    section_count: int
    part_count: int
    structural_only_count: int
    max_chunk_tokens: int
    avg_chunk_tokens: float
    warning_count: int
    skills: list[SkillIngestReport]
    warnings: list[dict[str, Any]]


def build_ingest_records(
    skills_dir: str | Path,
    *,
    max_chunk_tokens: int | None = None,
    chunk_overlap_tokens: int = 0,
    min_chunk_tokens: int = 80,
    token_counter: TokenCounter | None = None,
) -> IngestRecords:
    """把 skill Markdown 文档转换成可入库记录。"""
    manifests = discover_skill_manifests(skills_dir)
    docs: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for manifest in manifests:
        text = manifest.path.read_text(encoding="utf-8")
        doc_id = _stable_id("doc", str(manifest.path.resolve()))
        content_hash = _sha256(text)
        docs.append(_doc_record(doc_id, manifest, text, content_hash))

        parse_result = parse_markdown_document_with_report(
            doc_id,
            str(manifest.path),
            text,
            max_chunk_tokens=max_chunk_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            min_chunk_tokens=min_chunk_tokens,
            token_counter=token_counter,
        )
        for warning in parse_result.warnings:
            warnings.append(
                {
                    "skill_name": manifest.skill_name,
                    "file_path": str(manifest.path),
                    "code": warning.code,
                    "message": warning.message,
                    "line_number": warning.line_number,
                    "heading_path": warning.heading_path,
                }
            )

        for chunk in parse_result.chunks:
            metadata = {
                "skill_name": manifest.skill_name,
                "heading_path": chunk.heading_path,
                "tags": manifest.tags,
            }
            metadata_text = _metadata_text(manifest, chunk.heading_path, chunk.heading)
            chunks.append(
                {
                    "id": chunk.id,
                    "doc_id": doc_id,
                    "framework_name": manifest.framework_name,
                    "framework_version": manifest.framework_version,
                    "file_path": str(manifest.path),
                    "heading": chunk.heading,
                    "heading_level": chunk.heading_level,
                    "heading_path": chunk.heading_path,
                    "sort_order": chunk.sort_order,
                    "node_type": chunk.node_type,
                    "structural_only": chunk.structural_only,
                    "token_count": chunk.token_count,
                    "parent_id": chunk.parent_id,
                    "prev_sibling_id": chunk.prev_sibling_id,
                    "next_sibling_id": chunk.next_sibling_id,
                    "own_content": chunk.own_content,
                    "raw_markdown": chunk.raw_markdown,
                    "metadata_json": metadata,
                    "metadata_text": metadata_text,
                    "content_embedding": None,
                    "metadata_embedding": None,
                    "content_hash": _sha256(chunk.raw_markdown),
                }
            )

    return IngestRecords(docs=docs, chunks=chunks, warnings=warnings)


def build_ingest_report(records: IngestRecords) -> IngestReport:
    """汇总 ingest 解析结果，供 dry-run/report 和人工检查使用。"""
    skill_reports: list[SkillIngestReport] = []
    chunks_by_skill: dict[str, list[dict[str, Any]]] = {}
    warnings_by_skill: dict[str, list[dict[str, Any]]] = {}
    for chunk in records.chunks:
        chunks_by_skill.setdefault(str(chunk["metadata_json"]["skill_name"]), []).append(chunk)
    for warning in records.warnings:
        warnings_by_skill.setdefault(str(warning["skill_name"]), []).append(warning)

    for doc in records.docs:
        skill_name = str(doc["skill_name"])
        skill_chunks = chunks_by_skill.get(skill_name, [])
        skill_reports.append(
            SkillIngestReport(
                skill_name=skill_name,
                file_path=str(doc["file_path"]),
                chunk_count=len(skill_chunks),
                section_count=_count_chunks(skill_chunks, node_type="section"),
                part_count=_count_chunks(skill_chunks, node_type="part"),
                structural_only_count=sum(
                    1 for chunk in skill_chunks if bool(chunk["structural_only"])
                ),
                max_chunk_tokens=max((int(chunk["token_count"]) for chunk in skill_chunks), default=0),
                avg_chunk_tokens=_avg_token_count(skill_chunks),
                warning_count=len(warnings_by_skill.get(skill_name, [])),
            )
        )

    return IngestReport(
        doc_count=len(records.docs),
        chunk_count=len(records.chunks),
        section_count=_count_chunks(records.chunks, node_type="section"),
        part_count=_count_chunks(records.chunks, node_type="part"),
        structural_only_count=sum(1 for chunk in records.chunks if bool(chunk["structural_only"])),
        max_chunk_tokens=max((int(chunk["token_count"]) for chunk in records.chunks), default=0),
        avg_chunk_tokens=_avg_token_count(records.chunks),
        warning_count=len(records.warnings),
        skills=skill_reports,
        warnings=records.warnings,
    )


def render_ingest_report(report: IngestReport) -> str:
    """把 ingest 报告渲染成 CLI 友好的文本。"""
    lines = [
        "Ingest report",
        (
            f"docs={report.doc_count} chunks={report.chunk_count} "
            f"sections={report.section_count} parts={report.part_count} "
            f"structural_only={report.structural_only_count}"
        ),
        (
            f"max_chunk_tokens={report.max_chunk_tokens} "
            f"avg_chunk_tokens={report.avg_chunk_tokens:.1f} warnings={report.warning_count}"
        ),
    ]
    for skill in report.skills:
        lines.append(
            (
                f"- {skill.skill_name}: chunks={skill.chunk_count} "
                f"sections={skill.section_count} parts={skill.part_count} "
                f"structural_only={skill.structural_only_count} "
                f"max_tokens={skill.max_chunk_tokens} "
                f"avg_tokens={skill.avg_chunk_tokens:.1f} warnings={skill.warning_count}"
            )
        )
    if report.warnings:
        lines.append("warnings:")
        for warning in report.warnings:
            heading_path = " > ".join(warning["heading_path"])
            lines.append(
                (
                    f"- {warning['skill_name']}:{warning['line_number']} "
                    f"{warning['code']} {heading_path} - {warning['message']}"
                )
            )
    return "\n".join(lines)


def _count_chunks(chunks: list[dict[str, Any]], *, node_type: str) -> int:
    return sum(1 for chunk in chunks if chunk["node_type"] == node_type)


def _avg_token_count(chunks: list[dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    return sum(int(chunk["token_count"]) for chunk in chunks) / len(chunks)


def apply_record_cache(records: IngestRecords, cache: dict[str, dict[str, Any]]) -> None:
    """按 content_hash 复用已经入库的元数据和向量，避免重复调用模型。"""
    for chunk in records.chunks:
        cached = cache.get(chunk["content_hash"])
        if not cached:
            continue
        chunk["metadata_json"] = cached["metadata_json"]
        chunk["metadata_text"] = cached["metadata_text"]
        chunk["content_embedding"] = cached["content_embedding"]
        chunk["metadata_embedding"] = cached["metadata_embedding"]


def load_record_cache(
    database_url: str,
    *,
    connect_timeout_seconds: int = 5,
) -> dict[str, dict[str, Any]]:
    """读取已入库 chunk 的元数据和向量缓存。"""
    with psycopg.connect(
        _psycopg_url(database_url),
        row_factory=dict_row,
        connect_timeout=connect_timeout_seconds,
    ) as conn:
        rows = conn.execute(
            """
            SELECT content_hash, metadata_json, metadata_text,
                   content_embedding::text AS content_embedding,
                   metadata_embedding::text AS metadata_embedding
            FROM doc_chunks
            WHERE content_embedding IS NOT NULL
              AND metadata_embedding IS NOT NULL
            """
        ).fetchall()
    return {
        str(row["content_hash"]): {
            "metadata_json": row["metadata_json"],
            "metadata_text": row["metadata_text"],
            "content_embedding": _parse_pgvector_text(row["content_embedding"]),
            "metadata_embedding": _parse_pgvector_text(row["metadata_embedding"]),
        }
        for row in rows
    }


def apply_embeddings(records: IngestRecords, embedding_client: Any) -> None:
    """为 chunk 生成正文向量和元数据向量；调用方决定是否启用。"""
    content_chunks = [chunk for chunk in records.chunks if chunk["content_embedding"] is None]
    if content_chunks:
        content_embeddings = embedding_client.embed_texts(
            [chunk["raw_markdown"] for chunk in content_chunks]
        )
        for chunk, content_embedding in zip(content_chunks, content_embeddings, strict=True):
            chunk["content_embedding"] = content_embedding

    metadata_chunks = [chunk for chunk in records.chunks if chunk["metadata_embedding"] is None]
    if metadata_chunks:
        metadata_embeddings = embedding_client.embed_texts(
            [chunk["metadata_text"] for chunk in metadata_chunks]
        )
        for chunk, metadata_embedding in zip(metadata_chunks, metadata_embeddings, strict=True):
            chunk["metadata_embedding"] = metadata_embedding


SEMANTIC_METADATA_KEYS = {
    "module_type",
    "component_name",
    "api_name",
    "usage_type",
    "searchable_keywords",
    "summary",
}


def _has_semantic_metadata(metadata_json: dict[str, Any]) -> bool:
    """判断是否已有 LLM 语义元数据，避免 content_hash 未变化时重复调用模型。"""
    for key in SEMANTIC_METADATA_KEYS:
        value = metadata_json.get(key)
        if isinstance(value, list) and any(value):
            return True
        if isinstance(value, str) and value.strip():
            return True
        if value and not isinstance(value, list | str):
            return True
    return False


def apply_metadata_extraction(records: IngestRecords, metadata_extractor: Any, on_progress: Any = None) -> None:
    """在 ingest 阶段用 LLM 元数据增强检索字段。"""
    chunks = [
        chunk
        for chunk in records.chunks
        if not chunk["structural_only"] and not _has_semantic_metadata(chunk["metadata_json"])
    ]
    total = len(chunks)
    if not chunks:
        return

    items = [
        {"raw_markdown": chunk["raw_markdown"], "heading_path": chunk["heading_path"]}
        for chunk in chunks
    ]
    if hasattr(metadata_extractor, "iter_extract_many"):
        extracted_items = metadata_extractor.iter_extract_many(items)
    elif hasattr(metadata_extractor, "extract_many"):
        extracted_items = iter(metadata_extractor.extract_many(items))
    else:
        extracted_items = (
            metadata_extractor.extract(item["raw_markdown"], item["heading_path"]) for item in items
        )

    for index, (chunk, extracted) in enumerate(zip(chunks, extracted_items, strict=True), start=1):
        chunk["metadata_json"] = {**chunk["metadata_json"], **extracted}
        chunk["metadata_text"] = _metadata_json_to_text(chunk["metadata_json"])
        # metadata_text 已变化，旧的元数据向量不再可信，交给 apply_embeddings 重新生成。
        chunk["metadata_embedding"] = None
        if on_progress is not None:
            on_progress(index, total, " > ".join(chunk["heading_path"]))


def ingest_records(
    database_url: str,
    records: IngestRecords,
    *,
    prune: bool = False,
    connect_timeout_seconds: int = 5,
) -> None:
    """写入 docs/doc_chunks；默认不删除未扫描到的旧文档。"""
    with psycopg.connect(
        _psycopg_url(database_url),
        autocommit=True,
        connect_timeout=connect_timeout_seconds,
    ) as conn:
        if prune:
            _prune_missing_docs(conn, [doc["id"] for doc in records.docs])
        chunk_ids_by_doc = _chunk_ids_by_doc(records.chunks)
        for doc in records.docs:
            _upsert_doc(conn, doc)
            _delete_stale_chunks(conn, doc["id"], chunk_ids_by_doc.get(doc["id"], []))
        for chunk in records.chunks:
            _upsert_chunk(conn, chunk)


def _doc_record(
    doc_id: str, manifest: SkillManifest, text: str, content_hash: str
) -> dict[str, Any]:
    title = _first_markdown_heading(text) or manifest.skill_name
    return {
        "id": doc_id,
        "skill_name": manifest.skill_name,
        "description": manifest.description,
        "framework_name": manifest.framework_name,
        "framework_version": manifest.framework_version,
        "file_path": str(manifest.path),
        "title": title,
        "content_hash": content_hash,
    }


def _upsert_doc(conn: psycopg.Connection, doc: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO docs (
            id, skill_name, description, framework_name, framework_version,
            file_path, title, content_hash, indexed_at
        )
        VALUES (
            %(id)s, %(skill_name)s, %(description)s, %(framework_name)s, %(framework_version)s,
            %(file_path)s, %(title)s, %(content_hash)s, now()
        )
        ON CONFLICT (id) DO UPDATE SET
            skill_name = EXCLUDED.skill_name,
            description = EXCLUDED.description,
            framework_name = EXCLUDED.framework_name,
            framework_version = EXCLUDED.framework_version,
            file_path = EXCLUDED.file_path,
            title = EXCLUDED.title,
            content_hash = EXCLUDED.content_hash,
            indexed_at = now()
        """,
        doc,
    )


def _upsert_chunk(conn: psycopg.Connection, chunk: dict[str, Any]) -> None:
    params = dict(chunk)
    params["heading_path"] = json.dumps(chunk["heading_path"], ensure_ascii=False)
    params["metadata_json"] = json.dumps(chunk["metadata_json"], ensure_ascii=False)
    params["content_embedding"] = _format_optional_vector(chunk["content_embedding"])
    params["metadata_embedding"] = _format_optional_vector(chunk["metadata_embedding"])
    conn.execute(
        """
        INSERT INTO doc_chunks (
            id, doc_id, framework_name, framework_version, file_path,
            heading, heading_level, heading_path, sort_order,
            node_type, structural_only, token_count,
            parent_id, prev_sibling_id, next_sibling_id,
            own_content, raw_markdown, metadata_json, metadata_text,
            content_embedding, metadata_embedding,
            content_tsv, metadata_tsv, content_hash, updated_at
        )
        VALUES (
            %(id)s, %(doc_id)s, %(framework_name)s, %(framework_version)s, %(file_path)s,
            %(heading)s, %(heading_level)s, %(heading_path)s::jsonb, %(sort_order)s,
            %(node_type)s, %(structural_only)s, %(token_count)s,
            %(parent_id)s, %(prev_sibling_id)s, %(next_sibling_id)s,
            %(own_content)s, %(raw_markdown)s, %(metadata_json)s::jsonb, %(metadata_text)s,
            %(content_embedding)s::vector, %(metadata_embedding)s::vector,
            to_tsvector('simple', %(raw_markdown)s),
            to_tsvector('simple', %(metadata_text)s),
            %(content_hash)s, now()
        )
        ON CONFLICT (id) DO UPDATE SET
            heading = EXCLUDED.heading,
            heading_level = EXCLUDED.heading_level,
            heading_path = EXCLUDED.heading_path,
            sort_order = EXCLUDED.sort_order,
            node_type = EXCLUDED.node_type,
            structural_only = EXCLUDED.structural_only,
            token_count = EXCLUDED.token_count,
            parent_id = EXCLUDED.parent_id,
            prev_sibling_id = EXCLUDED.prev_sibling_id,
            next_sibling_id = EXCLUDED.next_sibling_id,
            own_content = EXCLUDED.own_content,
            raw_markdown = EXCLUDED.raw_markdown,
            metadata_json = EXCLUDED.metadata_json,
            metadata_text = EXCLUDED.metadata_text,
            content_embedding = EXCLUDED.content_embedding,
            metadata_embedding = EXCLUDED.metadata_embedding,
            content_tsv = EXCLUDED.content_tsv,
            metadata_tsv = EXCLUDED.metadata_tsv,
            content_hash = EXCLUDED.content_hash,
            updated_at = now()
        """,
        params,
    )


def _prune_missing_docs(conn: psycopg.Connection, doc_ids: list[str]) -> None:
    if not doc_ids:
        conn.execute("DELETE FROM docs")
        return
    conn.execute("DELETE FROM docs WHERE NOT (id = ANY(%s))", (doc_ids,))


def _chunk_ids_by_doc(chunks: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk["doc_id"], []).append(chunk["id"])
    return grouped


def _delete_stale_chunks(conn: psycopg.Connection, doc_id: str, chunk_ids: list[str]) -> None:
    """删除同一文档上一次入库残留、但本次解析已经不存在的 chunk。"""
    if not chunk_ids:
        conn.execute("DELETE FROM doc_chunks WHERE doc_id = %s", (doc_id,))
        return
    conn.execute(
        "DELETE FROM doc_chunks WHERE doc_id = %s AND NOT (id = ANY(%s))",
        (doc_id, chunk_ids),
    )


def _metadata_text(manifest: SkillManifest, heading_path: list[str], heading: str) -> str:
    # 中文 FTS 先依赖标题、标签、描述等关键词补强。
    return " ".join(
        part
        for part in [
            manifest.skill_name,
            manifest.description,
            manifest.framework_name,
            manifest.framework_version,
            heading,
            " ".join(heading_path),
            " ".join(manifest.tags),
        ]
        if part
    )


def _metadata_json_to_text(metadata: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in metadata.values():
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def _first_markdown_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{_sha256(value)[:16]}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _psycopg_url(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def _format_optional_vector(values: Any) -> str | None:
    if values is None:
        return None
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _parse_pgvector_text(value: Any) -> list[float] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return [float(part) for part in text.strip("[]").split(",") if part]

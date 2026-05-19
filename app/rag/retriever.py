from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.models.embeddings import format_pgvector


@dataclass(frozen=True)
class FusedScore:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    doc_id: str
    skill_name: str
    file_path: str
    heading: str
    heading_path: list[str]
    sort_order: int
    raw_markdown: str
    score: float


def fuse_rrf(rank_lists: list[list[str]], *, k: int = 60) -> list[FusedScore]:
    """使用 RRF 融合多个召回排序列表。"""
    scores: dict[str, float] = {}
    for rank_list in rank_lists:
        for rank, chunk_id in enumerate(rank_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return [
        FusedScore(chunk_id=chunk_id, score=score)
        for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def extract_query_terms(query: str) -> list[str]:
    """提取 API 名、类名等适合 ILIKE 兜底的检索词。"""
    terms = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", query)
    seen: set[str] = set()
    unique_terms: list[str] = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    return unique_terms


def expand_queries(query: str, *, max_terms: int = 4) -> list[str]:
    """生成检索扩展 query；扩展项只用于召回，不作为框架事实。"""
    queries = [query]
    for term in extract_query_terms(query)[:max_terms]:
        if term != query and term not in queries:
            queries.append(term)
    return queries


def expand_scores_once(
    seed_scores: dict[str, float], related_ids: dict[str, list[str]], *, ratio: float
) -> dict[str, float]:
    """把结构邻居加入候选集合，邻居分数按当前节点折扣继承。"""
    expanded = dict(seed_scores)
    for chunk_id, score in seed_scores.items():
        for related_id in related_ids.get(chunk_id, []):
            inherited_score = score * ratio
            expanded[related_id] = max(expanded.get(related_id, 0.0), inherited_score)
    return expanded


def limit_scores(score_by_id: dict[str, float], *, top_k: int) -> dict[str, float]:
    """结构扩展后按得分做最终截断，避免返回块数超过上下文预算。"""
    return dict(
        sorted(score_by_id.items(), key=lambda item: item[1], reverse=True)[:top_k]
    )


def retrieve_chunks(
    database_url: str,
    *,
    skill_name: str,
    query: str,
    top_k: int = 12,
    rrf_k: int = 60,
    query_expansion_max_terms: int = 4,
    query_embedding: list[float] | None = None,
    metadata_query_embedding: list[float] | None = None,
) -> list[RetrievedChunk]:
    """在指定 skill 内执行四路召回并做 RRF 融合。"""
    with psycopg.connect(_psycopg_url(database_url), row_factory=dict_row) as conn:
        rank_lists = []
        for expanded_query in expand_queries(query, max_terms=query_expansion_max_terms):
            rank_lists.extend(
                [
                    _fts_rank_list(conn, skill_name, expanded_query, "content_tsv", "raw_markdown"),
                    _fts_rank_list(conn, skill_name, expanded_query, "metadata_tsv", "metadata_text"),
                ]
            )
        if query_embedding is not None:
            rank_lists.append(_vector_rank_list(conn, skill_name, query_embedding, "content_embedding"))
        if metadata_query_embedding is not None:
            rank_lists.append(
                _vector_rank_list(conn, skill_name, metadata_query_embedding, "metadata_embedding")
            )
        fused = fuse_rrf([items for items in rank_lists if items], k=rrf_k)
        selected = fused[:top_k]
        if not selected:
            return []

        score_by_id = {item.chunk_id: item.score for item in selected}
        related_ids = _load_related_ids(conn, list(score_by_id))
        score_by_id = expand_scores_once(score_by_id, related_ids, ratio=0.55)
        score_by_id = limit_scores(score_by_id, top_k=top_k)
        rows = _load_chunks(conn, list(score_by_id))

    chunks = [
        RetrievedChunk(
            id=str(row["id"]),
            doc_id=str(row["doc_id"]),
            skill_name=str(row["skill_name"]),
            file_path=str(row["file_path"]),
            heading=str(row["heading"]),
            heading_path=list(row["heading_path"]),
            sort_order=int(row["sort_order"]),
            raw_markdown=str(row["raw_markdown"]),
            score=score_by_id[str(row["id"])],
        )
        for row in rows
    ]
    return sorted(chunks, key=lambda chunk: (chunk.doc_id, chunk.sort_order))


def build_vector_rank_sql(vector_column: str) -> str:
    """生成 pgvector 召回 SQL；列名只允许内部白名单传入。"""
    if vector_column not in {"content_embedding", "metadata_embedding"}:
        raise ValueError(f"Unsupported vector column: {vector_column}")
    return f"""
        SELECT c.id
        FROM doc_chunks c
        JOIN docs d ON d.id = c.doc_id
        WHERE d.skill_name = %(skill_name)s
          AND c.{vector_column} IS NOT NULL
        ORDER BY c.{vector_column} <=> %(query_embedding)s::vector
        LIMIT 30
    """


def _fts_rank_list(
    conn: psycopg.Connection,
    skill_name: str,
    query: str,
    tsv_column: str,
    text_column: str,
) -> list[str]:
    like_terms = [f"%{term}%" for term in extract_query_terms(query)]
    if not like_terms:
        like_terms = [f"%{query}%"]

    sql = f"""
        WITH q AS (SELECT plainto_tsquery('simple', %(query)s) AS tsq)
        SELECT c.id
        FROM doc_chunks c
        JOIN docs d ON d.id = c.doc_id
        CROSS JOIN q
        WHERE d.skill_name = %(skill_name)s
          AND (
            c.{tsv_column} @@ q.tsq
            OR c.{text_column} ILIKE ANY(%(like_terms)s)
          )
        ORDER BY
          ts_rank(c.{tsv_column}, q.tsq) DESC,
          c.sort_order ASC
        LIMIT 30
    """
    rows = conn.execute(
        sql,
        {
            "query": query,
            "skill_name": skill_name,
            "like_terms": like_terms,
        },
    ).fetchall()
    return [str(row["id"]) for row in rows]


def _vector_rank_list(
    conn: psycopg.Connection,
    skill_name: str,
    embedding: list[float],
    vector_column: str,
) -> list[str]:
    rows = conn.execute(
        build_vector_rank_sql(vector_column),
        {
            "skill_name": skill_name,
            "query_embedding": format_pgvector(embedding),
        },
    ).fetchall()
    return [str(row["id"]) for row in rows]


def _load_chunks(conn: psycopg.Connection, chunk_ids: list[str]) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT
            c.id, c.doc_id, d.skill_name, c.file_path, c.heading,
            c.heading_path, c.sort_order, c.raw_markdown
        FROM doc_chunks c
        JOIN docs d ON d.id = c.doc_id
        WHERE c.id = ANY(%(chunk_ids)s)
        """,
        {"chunk_ids": chunk_ids},
    ).fetchall()


def _load_related_ids(conn: psycopg.Connection, chunk_ids: list[str]) -> dict[str, list[str]]:
    rows = conn.execute(
        """
        SELECT id, parent_id, prev_sibling_id, next_sibling_id
        FROM doc_chunks
        WHERE id = ANY(%(chunk_ids)s)
        """,
        {"chunk_ids": chunk_ids},
    ).fetchall()
    child_rows = conn.execute(
        """
        SELECT parent_id, id
        FROM doc_chunks
        WHERE parent_id = ANY(%(chunk_ids)s)
        ORDER BY sort_order
        """,
        {"chunk_ids": chunk_ids},
    ).fetchall()

    related: dict[str, list[str]] = {chunk_id: [] for chunk_id in chunk_ids}
    for row in rows:
        source_id = str(row["id"])
        for column in ["parent_id", "prev_sibling_id", "next_sibling_id"]:
            if row[column]:
                related[source_id].append(str(row[column]))
    for row in child_rows:
        related.setdefault(str(row["parent_id"]), []).append(str(row["id"]))
    return related


def _psycopg_url(database_url: str) -> str:
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)

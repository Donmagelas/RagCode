from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.models.embeddings import format_pgvector
from app.rag.token_counter import HeuristicTokenCounter, TokenCounter


@dataclass(frozen=True)
class FusedScore:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class RouteHit:
    route: str
    query: str
    chunk_id: str
    rank: int
    score: float | None = None
    heading_path: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExpandedChunk:
    chunk_id: str
    source_chunk_id: str
    relation: str
    score: float
    heading_path: list[str] = field(default_factory=list)


@dataclass
class SkillRetrieveTrace:
    skill_name: str
    query: str
    expanded_queries: list[str]
    route_results: dict[str, list[RouteHit]]
    rrf_results: list[FusedScore]
    seed_chunks: list[FusedScore]
    expanded_chunks: list[ExpandedChunk]
    final_chunk_ids: list[str]
    human_review: dict[str, Any] | None = None


@dataclass(frozen=True)
class RetrieveChunksResult:
    chunks: list["RetrievedChunk"]
    trace: SkillRetrieveTrace


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


def build_route_hits(
    *, route: str, query: str, chunk_ids: list[str], scores: dict[str, float] | None = None
) -> list[RouteHit]:
    """把一路召回结果转成可调试的 rank trace。"""
    score_by_id = scores or {}
    return [
        RouteHit(
            route=route,
            query=query,
            chunk_id=chunk_id,
            rank=rank,
            score=score_by_id.get(chunk_id),
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


def record_human_review(
    trace: SkillRetrieveTrace, *, selected_indexes: list[int], approved_chunk_ids: list[str]
) -> None:
    """把 debug-rag 的人工确认结果写入 trace。"""
    trace.human_review = {
        "enabled": True,
        "selected_indexes": selected_indexes,
        "approved_chunk_ids": approved_chunk_ids,
    }


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


def build_expansion_events(
    seed_scores: dict[str, float], related_ids: dict[str, list[Any]], *, ratio: float
) -> list[ExpandedChunk]:
    """记录结构扩展来源，便于 debug-rag 解释为什么额外加入某些块。"""
    events: list[ExpandedChunk] = []
    for source_id, score in seed_scores.items():
        for related_item in related_ids.get(source_id, []):
            if isinstance(related_item, tuple):
                related_id, relation = related_item
            else:
                related_id, relation = related_item, "related"
            events.append(
                ExpandedChunk(
                    chunk_id=str(related_id),
                    source_chunk_id=source_id,
                    relation=str(relation),
                    score=score * ratio,
                )
            )
    return events


def limit_scores(score_by_id: dict[str, float], *, top_k: int) -> dict[str, float]:
    """结构扩展后按得分做最终截断，避免返回块数超过上下文预算。"""
    return dict(
        sorted(score_by_id.items(), key=lambda item: item[1], reverse=True)[:top_k]
    )


def select_seed_chunks(
    fused_scores: list[FusedScore] | list[tuple[str, float]],
    *,
    seed_top_n: int,
    threshold_ratio: float,
) -> dict[str, float]:
    """按 top_n 和动态阈值选择 seed chunk，避免低分结果进入结构扩展。"""
    if not fused_scores or seed_top_n <= 0:
        return {}

    def chunk_id(item: FusedScore | tuple[str, float]) -> str:
        return item.chunk_id if isinstance(item, FusedScore) else item[0]

    def score(item: FusedScore | tuple[str, float]) -> float:
        return item.score if isinstance(item, FusedScore) else item[1]

    top_score = score(fused_scores[0])
    threshold = top_score * threshold_ratio
    return {
        chunk_id(item): score(item)
        for item in fused_scores[:seed_top_n]
        if score(item) >= threshold
    }


def expand_scores_recursive(
    seed_scores: dict[str, float],
    *,
    load_related_edges,
    ratio: float,
    max_depth: int,
) -> tuple[dict[str, float], list[ExpandedChunk]]:
    """按标题树关系递归扩展，深度和折扣全部由配置控制。"""
    score_by_id = dict(seed_scores)
    frontier = dict(seed_scores)
    events: list[ExpandedChunk] = []

    for _depth in range(max_depth):
        if not frontier:
            break
        related_edges = load_related_edges(list(frontier))
        next_frontier: dict[str, float] = {}
        for source_id, edges in related_edges.items():
            source_score = frontier.get(source_id)
            if source_score is None:
                continue
            for related_id, relation in edges:
                inherited_score = source_score * ratio
                if inherited_score <= score_by_id.get(str(related_id), 0.0):
                    continue
                related_chunk_id = str(related_id)
                score_by_id[related_chunk_id] = inherited_score
                next_frontier[related_chunk_id] = inherited_score
                events.append(
                    ExpandedChunk(
                        chunk_id=related_chunk_id,
                        source_chunk_id=source_id,
                        relation=str(relation),
                        score=inherited_score,
                    )
                )
        frontier = next_frontier
    return score_by_id, events


def limit_chunk_ids_by_context_tokens(
    chunk_ids: list[str],
    *,
    raw_markdown_by_id: dict[str, str],
    score_by_id: dict[str, float],
    max_context_tokens: int | None,
    token_counter: TokenCounter | None,
) -> list[str]:
    """按得分顺序装入 token 预算；最终显示顺序仍由调用方按文档顺序恢复。"""
    if max_context_tokens is None:
        return chunk_ids
    counter = token_counter or HeuristicTokenCounter()
    selected: list[str] = []
    used_tokens = 0
    for chunk_id in sorted(chunk_ids, key=lambda item: score_by_id[item], reverse=True):
        token_count = len(counter.encode(raw_markdown_by_id.get(chunk_id, "")))
        if used_tokens + token_count > max_context_tokens:
            continue
        selected.append(chunk_id)
        used_tokens += token_count
    return selected


def retrieve_chunks(
    database_url: str,
    *,
    skill_name: str,
    query: str,
    top_k: int = 12,
    rrf_k: int = 60,
    retriever_top_k: int = 30,
    seed_top_n: int = 8,
    seed_threshold_ratio: float = 0.75,
    expand_threshold_ratio: float = 0.55,
    query_expansion_max_terms: int = 4,
    max_depth: int = 3,
    max_context_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
    query_embedding: list[float] | None = None,
    metadata_query_embedding: list[float] | None = None,
    return_trace: bool = False,
) -> list[RetrievedChunk] | RetrieveChunksResult:
    """在指定 skill 内执行四路召回并做 RRF 融合。"""
    with psycopg.connect(_psycopg_url(database_url), row_factory=dict_row) as conn:
        rank_lists = []
        route_results: dict[str, list[RouteHit]] = {
            "content_fts": [],
            "metadata_fts": [],
            "content_vector": [],
            "metadata_vector": [],
        }
        expanded_queries = expand_queries(query, max_terms=query_expansion_max_terms)
        for expanded_query in expanded_queries:
            content_fts = _fts_rank_list(
                conn,
                skill_name,
                expanded_query,
                "content_tsv",
                "raw_markdown",
                limit=retriever_top_k,
            )
            metadata_fts = _fts_rank_list(
                conn,
                skill_name,
                expanded_query,
                "metadata_tsv",
                "metadata_text",
                limit=retriever_top_k,
            )
            rank_lists.extend(
                [
                    content_fts,
                    metadata_fts,
                ]
            )
            route_results["content_fts"].extend(
                build_route_hits(route="content_fts", query=expanded_query, chunk_ids=content_fts)
            )
            route_results["metadata_fts"].extend(
                build_route_hits(route="metadata_fts", query=expanded_query, chunk_ids=metadata_fts)
            )
        if query_embedding is not None:
            content_vector = _vector_rank_list(
                conn,
                skill_name,
                query_embedding,
                "content_embedding",
                limit=retriever_top_k,
            )
            rank_lists.append(content_vector)
            route_results["content_vector"].extend(
                build_route_hits(route="content_vector", query=query, chunk_ids=content_vector)
            )
        if metadata_query_embedding is not None:
            metadata_vector = _vector_rank_list(
                conn,
                skill_name,
                metadata_query_embedding,
                "metadata_embedding",
                limit=retriever_top_k,
            )
            rank_lists.append(metadata_vector)
            route_results["metadata_vector"].extend(
                build_route_hits(route="metadata_vector", query=query, chunk_ids=metadata_vector)
            )
        fused = fuse_rrf([items for items in rank_lists if items], k=rrf_k)
        if not fused:
            if return_trace:
                return RetrieveChunksResult(
                    chunks=[],
                    trace=SkillRetrieveTrace(
                        skill_name=skill_name,
                        query=query,
                        expanded_queries=expanded_queries,
                        route_results=route_results,
                        rrf_results=[],
                        seed_chunks=[],
                        expanded_chunks=[],
                        final_chunk_ids=[],
                    ),
                )
            return []

        seed_score_by_id = select_seed_chunks(
            fused,
            seed_top_n=seed_top_n,
            threshold_ratio=seed_threshold_ratio,
        )
        selected = [
            FusedScore(chunk_id=chunk_id, score=score)
            for chunk_id, score in seed_score_by_id.items()
        ]
        if not selected:
            if return_trace:
                return RetrieveChunksResult(
                    chunks=[],
                    trace=SkillRetrieveTrace(
                        skill_name=skill_name,
                        query=query,
                        expanded_queries=expanded_queries,
                        route_results=route_results,
                        rrf_results=fused,
                        seed_chunks=[],
                        expanded_chunks=[],
                        final_chunk_ids=[],
                    ),
                )
            return []

        score_by_id, expanded_chunks = expand_scores_recursive(
            seed_score_by_id,
            load_related_edges=lambda chunk_ids: _load_related_edges(conn, chunk_ids),
            ratio=expand_threshold_ratio,
            max_depth=max_depth,
        )
        score_by_id = limit_scores(score_by_id, top_k=top_k)
        rows = _load_chunks(conn, list(score_by_id))
        raw_markdown_by_id = {str(row["id"]): str(row["raw_markdown"]) for row in rows}
        selected_chunk_ids = limit_chunk_ids_by_context_tokens(
            list(score_by_id),
            raw_markdown_by_id=raw_markdown_by_id,
            score_by_id=score_by_id,
            max_context_tokens=max_context_tokens,
            token_counter=token_counter,
        )
        score_by_id = {chunk_id: score_by_id[chunk_id] for chunk_id in selected_chunk_ids}
        rows = [row for row in rows if str(row["id"]) in score_by_id]
        if return_trace:
            heading_by_id = _load_heading_paths(conn, _trace_chunk_ids(route_results, expanded_chunks, rows))
            _attach_heading_paths(route_results, heading_by_id)
            _attach_expansion_heading_paths(expanded_chunks, heading_by_id)

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
    chunks = sorted(chunks, key=lambda chunk: (chunk.doc_id, chunk.sort_order))
    if not return_trace:
        return chunks
    return RetrieveChunksResult(
        chunks=chunks,
        trace=SkillRetrieveTrace(
            skill_name=skill_name,
            query=query,
            expanded_queries=expanded_queries,
            route_results=route_results,
            rrf_results=fused,
            seed_chunks=selected,
            expanded_chunks=expanded_chunks,
            final_chunk_ids=[chunk.id for chunk in chunks],
        ),
    )


def _trace_chunk_ids(
    route_results: dict[str, list[RouteHit]],
    expanded_chunks: list[ExpandedChunk],
    rows: list[dict[str, Any]],
) -> list[str]:
    chunk_ids = {str(row["id"]) for row in rows}
    for hits in route_results.values():
        chunk_ids.update(hit.chunk_id for hit in hits)
    chunk_ids.update(event.chunk_id for event in expanded_chunks)
    return sorted(chunk_ids)


def _attach_heading_paths(
    route_results: dict[str, list[RouteHit]], heading_by_id: dict[str, list[str]]
) -> None:
    for route, hits in route_results.items():
        route_results[route] = [
            RouteHit(
                route=hit.route,
                query=hit.query,
                chunk_id=hit.chunk_id,
                rank=hit.rank,
                score=hit.score,
                heading_path=heading_by_id.get(hit.chunk_id, []),
            )
            for hit in hits
        ]


def _attach_expansion_heading_paths(
    expanded_chunks: list[ExpandedChunk], heading_by_id: dict[str, list[str]]
) -> None:
    for index, event in enumerate(expanded_chunks):
        expanded_chunks[index] = ExpandedChunk(
            chunk_id=event.chunk_id,
            source_chunk_id=event.source_chunk_id,
            relation=event.relation,
            score=event.score,
            heading_path=heading_by_id.get(event.chunk_id, []),
        )


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
        LIMIT %(limit)s
    """


def _fts_rank_list(
    conn: psycopg.Connection,
    skill_name: str,
    query: str,
    tsv_column: str,
    text_column: str,
    *,
    limit: int,
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
        LIMIT %(limit)s
    """
    rows = conn.execute(
        sql,
        {
            "query": query,
            "skill_name": skill_name,
            "like_terms": like_terms,
            "limit": limit,
        },
    ).fetchall()
    return [str(row["id"]) for row in rows]


def _vector_rank_list(
    conn: psycopg.Connection,
    skill_name: str,
    embedding: list[float],
    vector_column: str,
    *,
    limit: int,
) -> list[str]:
    rows = conn.execute(
        build_vector_rank_sql(vector_column),
        {
            "skill_name": skill_name,
            "query_embedding": format_pgvector(embedding),
            "limit": limit,
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


def _load_heading_paths(conn: psycopg.Connection, chunk_ids: list[str]) -> dict[str, list[str]]:
    if not chunk_ids:
        return {}
    rows = conn.execute(
        """
        SELECT id, heading_path
        FROM doc_chunks
        WHERE id = ANY(%(chunk_ids)s)
        """,
        {"chunk_ids": chunk_ids},
    ).fetchall()
    return {str(row["id"]): list(row["heading_path"]) for row in rows}


def _load_related_edges(conn: psycopg.Connection, chunk_ids: list[str]) -> dict[str, list[tuple[str, str]]]:
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

    related: dict[str, list[tuple[str, str]]] = {chunk_id: [] for chunk_id in chunk_ids}
    for row in rows:
        source_id = str(row["id"])
        if row["parent_id"]:
            related[source_id].append((str(row["parent_id"]), "parent"))
        if row["prev_sibling_id"]:
            related[source_id].append((str(row["prev_sibling_id"]), "prev_sibling"))
        if row["next_sibling_id"]:
            related[source_id].append((str(row["next_sibling_id"]), "next_sibling"))
    for row in child_rows:
        related.setdefault(str(row["parent_id"]), []).append((str(row["id"]), "child"))
    return related


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

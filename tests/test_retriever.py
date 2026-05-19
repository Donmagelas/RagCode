from app.rag.retriever import (
    build_vector_rank_sql,
    expand_queries,
    expand_scores_once,
    extract_query_terms,
    fuse_rrf,
    limit_scores,
)


def test_fuse_rrf_combines_rank_lists() -> None:
    fused = fuse_rrf(
        [
            ["a", "b", "c"],
            ["b", "d", "a"],
        ],
        k=60,
    )

    assert fused[0].chunk_id == "b"
    assert {item.chunk_id for item in fused} == {"a", "b", "c", "d"}
    assert fused[0].score > fused[-1].score


def test_extract_query_terms_keeps_api_like_identifiers() -> None:
    assert extract_query_terms("OnBoot 里可以加载资源吗？") == ["OnBoot"]


def test_expand_queries_keeps_original_and_api_terms() -> None:
    assert expand_queries("OnBoot 里调用 LoadAsync 可以吗", max_terms=2) == [
        "OnBoot 里调用 LoadAsync 可以吗",
        "OnBoot",
        "LoadAsync",
    ]


def test_expand_scores_once_adds_related_chunks_with_discount() -> None:
    expanded = expand_scores_once(
        {"seed": 1.0},
        {"seed": ["parent", "child"]},
        ratio=0.55,
    )

    assert expanded["seed"] == 1.0
    assert expanded["parent"] == 0.55
    assert expanded["child"] == 0.55


def test_limit_scores_keeps_highest_scoring_chunks() -> None:
    limited = limit_scores(
        {
            "low": 0.1,
            "high": 0.9,
            "middle": 0.5,
        },
        top_k=2,
    )

    assert list(limited) == ["high", "middle"]


def test_build_vector_rank_sql_uses_pgvector_distance() -> None:
    sql = build_vector_rank_sql("content_embedding")

    assert "content_embedding IS NOT NULL" in sql
    assert "content_embedding <=> %(query_embedding)s::vector" in sql

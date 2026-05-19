from app.rag.retriever import (
    RelatedEdge,
    build_vector_rank_sql,
    expand_queries,
    expand_scores_recursive,
    expand_scores_once,
    extract_query_terms,
    fuse_rrf,
    limit_chunk_ids_by_context_tokens,
    limit_scores,
    select_seed_chunks,
)


class CharTokenCounter:
    def encode(self, text: str) -> list[str]:
        return list(text)

    def decode(self, token_ids: list[str]) -> str:
        return "".join(token_ids)


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


def test_select_seed_chunks_uses_top_n_and_dynamic_threshold() -> None:
    fused = [
        ("top", 1.0),
        ("keep", 0.8),
        ("drop_by_threshold", 0.7),
        ("drop_by_top_n", 0.95),
    ]

    selected = select_seed_chunks(fused, seed_top_n=3, threshold_ratio=0.75)

    assert list(selected) == ["top", "keep"]


def test_expand_scores_recursive_respects_depth_and_ratio() -> None:
    related = {
        "seed": [("parent", "parent")],
        "parent": [("grandparent", "parent")],
        "grandparent": [("too_far", "parent")],
    }

    expanded, events = expand_scores_recursive(
        {"seed": 1.0},
        load_related_edges=lambda ids: {chunk_id: related.get(chunk_id, []) for chunk_id in ids},
        ratio=0.5,
        max_depth=2,
    )

    assert expanded == {"seed": 1.0, "parent": 0.5}
    assert [event.chunk_id for event in events] == ["parent"]


def test_expand_scores_recursive_passes_through_structural_nodes() -> None:
    related = {
        "part-1": [RelatedEdge(chunk_id="section", relation="parent", structural_only=True)],
        "section": [
            RelatedEdge(chunk_id="part-1", relation="child", structural_only=False),
            RelatedEdge(chunk_id="part-2", relation="child", structural_only=False),
        ],
    }

    expanded, events = expand_scores_recursive(
        {"part-1": 1.0},
        load_related_edges=lambda ids: {chunk_id: related.get(chunk_id, []) for chunk_id in ids},
        ratio=0.55,
        max_depth=3,
    )

    assert expanded == {"part-1": 1.0, "part-2": 0.55}
    assert [(event.chunk_id, event.relation, event.structural_only) for event in events] == [
        ("section", "parent", True),
        ("part-2", "child", False),
    ]


def test_limit_chunk_ids_by_context_tokens_keeps_high_score_within_budget() -> None:
    raw_markdown_by_id = {
        "long": "abcdef",
        "short": "xy",
        "middle": "123",
    }
    score_by_id = {"long": 0.9, "short": 0.8, "middle": 0.7}

    selected = limit_chunk_ids_by_context_tokens(
        ["long", "short", "middle"],
        raw_markdown_by_id=raw_markdown_by_id,
        score_by_id=score_by_id,
        max_context_tokens=5,
        token_counter=CharTokenCounter(),
    )

    assert selected == ["short", "middle"]


def test_build_vector_rank_sql_uses_pgvector_distance() -> None:
    sql = build_vector_rank_sql("content_embedding")

    assert "content_embedding IS NOT NULL" in sql
    assert "content_embedding <=> %(query_embedding)s::vector" in sql
    assert "LIMIT %(limit)s" in sql

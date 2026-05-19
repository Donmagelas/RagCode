from app.rag.retriever import (
    ExpandedChunk,
    RouteHit,
    SkillRetrieveTrace,
    build_route_hits,
    record_human_review,
)


def test_build_route_hits_records_route_query_and_rank() -> None:
    hits = build_route_hits(
        route="content_fts",
        query="OnBoot LoadAsync",
        chunk_ids=["chunk-a", "chunk-b"],
        scores={"chunk-a": 0.8},
    )

    assert hits == [
        RouteHit(route="content_fts", query="OnBoot LoadAsync", chunk_id="chunk-a", rank=1, score=0.8),
        RouteHit(route="content_fts", query="OnBoot LoadAsync", chunk_id="chunk-b", rank=2, score=None),
    ]


def test_record_human_review_updates_trace() -> None:
    trace = SkillRetrieveTrace(
        skill_name="lifecycle",
        query="Can OnBoot load assets?",
        expanded_queries=["Can OnBoot load assets?", "OnBoot"],
        route_results={"content_fts": []},
        rrf_results=[],
        seed_chunks=[],
        expanded_chunks=[
            ExpandedChunk(
                chunk_id="child",
                source_chunk_id="seed",
                relation="child",
                score=0.5,
            )
        ],
        final_chunk_ids=["seed", "child"],
    )

    record_human_review(trace, selected_indexes=[2], approved_chunk_ids=["child"])

    assert trace.human_review == {
        "enabled": True,
        "selected_indexes": [2],
        "approved_chunk_ids": ["child"],
    }


def test_build_expansion_events_keeps_relation_names() -> None:
    from app.rag.retriever import build_expansion_events

    events = build_expansion_events(
        {"seed": 1.0},
        {"seed": [("parent-id", "parent"), ("child-id", "child")]},
        ratio=0.55,
    )

    assert events[0].relation == "parent"
    assert events[0].score == 0.55
    assert events[1].relation == "child"

from typer.testing import CliRunner

from app.cli import app, console_safe_text, render_retrieval_trace
from app.rag.retriever import FusedScore, RouteHit, SkillRetrieveTrace


def test_cli_exposes_first_stage_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "init-db" in result.output
    assert "ingest" in result.output
    assert "retrieve" in result.output
    assert "debug-rag" in result.output
    assert "agent-run" in result.output


def test_ingest_help_exposes_model_call_switches() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["ingest", "--help"])

    assert result.exit_code == 0
    assert "--with-metadata" in result.output
    assert "--with-embeddings" in result.output


def test_agent_run_help_exposes_smoke_debug_options() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["agent-run", "--help"])

    assert result.exit_code == 0
    assert "--max-turns" in result.output
    assert "--show-tools" in result.output


def test_console_safe_text_replaces_unencodable_characters() -> None:
    assert console_safe_text("ok ✅", encoding="gbk") == "ok ?"


def test_debug_rag_help_exposes_trace_option() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["debug-rag", "--help"])

    assert result.exit_code == 0
    assert "--trace" in result.output


def test_render_retrieval_trace_includes_routes_rrf_and_human_review() -> None:
    trace = SkillRetrieveTrace(
        skill_name="lifecycle",
        query="OnBoot",
        expanded_queries=["OnBoot"],
        route_results={
            "content_fts": [
                RouteHit(
                    route="content_fts",
                    query="OnBoot",
                    chunk_id="chunk-1",
                    rank=1,
                    heading_path=["Module", "OnBoot"],
                )
            ]
        },
        rrf_results=[FusedScore(chunk_id="chunk-1", score=0.1)],
        seed_chunks=[FusedScore(chunk_id="chunk-1", score=0.1)],
        expanded_chunks=[],
        final_chunk_ids=["chunk-1"],
        human_review={"enabled": True, "selected_indexes": [1], "approved_chunk_ids": ["chunk-1"]},
    )

    rendered = render_retrieval_trace(trace)

    assert "route: content_fts" in rendered
    assert "RRF top:" in rendered
    assert "human_review:" in rendered

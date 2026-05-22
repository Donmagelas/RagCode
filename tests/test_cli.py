from typer.testing import CliRunner

import app.cli as cli_module
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
    assert "--dry-run" in result.output
    assert "--report" in result.output


def test_ingest_dry_run_outputs_report_without_database_write(tmp_path, monkeypatch) -> None:
    skill_file = tmp_path / "skill.md"
    skill_file.write_text("# Skill\n\nabcdefghij\n\n### Jump\n\nbody", encoding="utf-8")

    class FakeTokenCounter:
        def __init__(self, _model_name: str) -> None:
            pass

        def encode(self, text: str) -> list[str]:
            return list(text)

        def decode(self, token_ids: list[str]) -> str:
            return "".join(token_ids)

    def fail_if_called(*_args, **_kwargs) -> None:
        raise AssertionError("dry-run must not write database")

    monkeypatch.setattr(cli_module, "Qwen3TokenCounter", FakeTokenCounter)
    monkeypatch.setattr(cli_module, "ingest_records", fail_if_called)

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "--skills-dir", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0
    assert "Ingest report" in result.output
    assert "Dry run complete" in result.output


def test_agent_run_help_exposes_smoke_debug_options() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["agent-run", "--help"])

    assert result.exit_code == 0
    assert "--max-turns" in result.output
    assert "--show-tools" in result.output
    assert "--format" in result.output
    assert "--output" in result.output
    assert "--verify" in result.output
    assert "--max-repair-attempts" in result.output


def test_agent_run_codex_backend_outputs_structured_json(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "agent-run",
            "--backend",
            "codex",
            "--goal",
            "实现 UI",
            "--skills-dir",
            str(skills_dir),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    data = __import__("json").loads(result.output)
    assert data["backend"] == "codex"
    assert data["status"] == "partial"
    assert data["goal"] == "实现 UI"


def test_emit_output_preserves_unicode_when_writing_file(tmp_path) -> None:
    output = tmp_path / "result.json"

    cli_module._emit_output("中文内容", output=output)

    assert output.read_text(encoding="utf-8") == "中文内容"


def test_console_safe_text_replaces_unencodable_characters() -> None:
    assert console_safe_text("ok ✅", encoding="gbk") == "ok ?"


def test_debug_rag_help_exposes_trace_option() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["debug-rag", "--help"])

    assert result.exit_code == 0
    assert "--trace" in result.output


def test_prepare_context_help_exposes_backend_and_human_review() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["prepare-context", "--help"])

    assert result.exit_code == 0
    assert "--backend" in result.output
    assert "--human-review" in result.output
    assert "--format" in result.output


def test_route_skill_help_exposes_goal_option() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["route-skill", "--help"])

    assert result.exit_code == 0
    assert "--goal" in result.output


def test_eval_rag_help_exposes_cases_file_and_format() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["eval-rag", "--help"])

    assert result.exit_code == 0
    assert "--cases-file" in result.output
    assert "--format" in result.output


def test_enrich_metadata_help_exposes_limit_and_workers() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["enrich-metadata", "--help"])

    assert result.exit_code == 0
    assert "--limit" in result.output
    assert "--workers" in result.output


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

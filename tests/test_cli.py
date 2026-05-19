from typer.testing import CliRunner

from app.cli import app, console_safe_text


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

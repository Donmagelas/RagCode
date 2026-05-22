import json

from app.coding.agent import CodingAgentResult
from app.coding.result import build_builtin_run_result, build_codex_run_result, render_run_result_json


def test_build_builtin_run_result_collects_tool_side_effects() -> None:
    agent_result = CodingAgentResult(
        final_response="完成了修改",
        messages=[],
        tool_results=[
            {"tool": "read_file", "args": {"path": "a.py"}, "ok": True, "output": {"path": "a.py"}},
            {
                "tool": "edit_file",
                "args": {"path": "a.py"},
                "ok": True,
                "output": {"path": "a.py", "changed": True},
            },
            {
                "tool": "bash",
                "args": {"command": "pytest"},
                "ok": True,
                "output": {"command": "pytest", "exit_code": 0},
            },
            {"tool": "bash", "args": {"command": "ruff"}, "ok": False, "error": "failed"},
        ],
    )

    result = build_builtin_run_result(
        goal="修 UI",
        selected_skills=["ui"],
        knowledge_chunk_count=2,
        agent_result=agent_result,
    )

    assert result.backend == "builtin"
    assert result.status == "partial"
    assert result.files_read == ["a.py"]
    assert result.files_changed[0].path == "a.py"
    assert result.commands_run[0].command == "pytest"
    assert result.test_results[0].status == "passed"
    assert result.errors == ["failed"]


def test_render_run_result_json_is_backend_neutral() -> None:
    result = build_codex_run_result(
        goal="修 UI",
        selected_skills=["ui"],
        knowledge_chunk_count=3,
        context_markdown="# Context",
    )

    data = json.loads(render_run_result_json(result))

    assert data["backend"] == "codex"
    assert data["status"] == "partial"
    assert data["knowledge"]["chunk_count"] == 3
    assert data["artifacts"][0]["type"] == "context_markdown"

from app.coding.loop import execute_tool_calls
from app.tools.registry import ToolRegistry


def test_execute_tool_calls_records_results_and_changed_files() -> None:
    registry = ToolRegistry()
    registry.register("read_file", lambda *, path: {"path": path, "content": "old"})
    registry.register("edit_file", lambda *, path, old_string, new_string: {"path": path, "changed": True})
    registry.register("bash", lambda *, command: {"command": command, "exit_code": 0})

    result = execute_tool_calls(
        [
            {"tool": "read_file", "args": {"path": "main.py"}},
            {
                "tool": "edit_file",
                "args": {"path": "main.py", "old_string": "old", "new_string": "new"},
            },
            {"tool": "bash", "args": {"command": "pytest"}},
        ],
        registry=registry,
    )

    assert result["files_read"] == ["main.py"]
    assert result["files_changed"] == ["main.py"]
    assert result["commands_run"] == [{"command": "pytest", "exit_code": 0}]
    assert len(result["tool_results"]) == 3


def test_execute_tool_calls_records_tool_errors() -> None:
    registry = ToolRegistry()
    registry.register("edit_file", lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad edit")))

    result = execute_tool_calls(
        [{"tool": "edit_file", "args": {"path": "main.py"}}],
        registry=registry,
    )

    assert result["tool_results"][0]["ok"] is False
    assert "bad edit" in result["tool_results"][0]["error"]

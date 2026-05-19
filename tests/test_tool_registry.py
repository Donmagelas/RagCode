from pathlib import Path

import pytest

from app.tools.registry import ToolRegistry, create_default_tool_registry


def test_tool_registry_dispatches_registered_tool() -> None:
    registry = ToolRegistry()
    registry.register("echo", lambda *, text: {"text": text})

    assert registry.call("echo", text="hello") == {"text": "hello"}


def test_tool_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(KeyError, match="Unknown tool"):
        registry.call("missing")


def test_default_tool_registry_exposes_coding_tools(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('ok')", encoding="utf-8")
    registry = create_default_tool_registry(workspace=tmp_path)

    assert registry.call("glob_search", pattern="*.py") == ["main.py"]
    assert registry.call("read_file", path="main.py")["content"] == "print('ok')"
    assert registry.names() == [
        "bash",
        "edit_file",
        "git_diff",
        "glob_search",
        "grep_search",
        "read_file",
        "write_file",
    ]

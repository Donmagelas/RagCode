from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.tools.bash_tool import run_bash
from app.tools.file_tools import edit_file, read_file, write_file
from app.tools.git_tool import git_diff
from app.tools.search_tools import glob_search, grep_search


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, handler: Callable[..., Any]) -> None:
        """注册一个可由 coding loop 调用的工具。"""
        self._tools[name] = handler

    def call(self, name: str, **kwargs: Any) -> Any:
        """按工具名派发调用，未知工具直接拒绝。"""
        handler = self._tools.get(name)
        if handler is None:
            raise KeyError(f"Unknown tool: {name}")
        return handler(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._tools)


def create_default_tool_registry(*, workspace: str | Path) -> ToolRegistry:
    """创建绑定到当前 workspace 的默认编码工具集合。"""
    registry = ToolRegistry()
    registry.register("bash", lambda **kwargs: run_bash(workspace=workspace, **kwargs))
    registry.register("edit_file", lambda **kwargs: edit_file(workspace=workspace, **kwargs))
    registry.register("git_diff", lambda **_kwargs: git_diff(workspace=workspace))
    registry.register("glob_search", lambda **kwargs: glob_search(workspace=workspace, **kwargs))
    registry.register("grep_search", lambda **kwargs: grep_search(workspace=workspace, **kwargs))
    registry.register("read_file", lambda **kwargs: read_file(workspace=workspace, **kwargs))
    registry.register("write_file", lambda **kwargs: write_file(workspace=workspace, **kwargs))
    return registry

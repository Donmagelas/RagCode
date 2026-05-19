from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.path_utils import relative_workspace_path, resolve_workspace_path


def read_file(*, workspace: str | Path, path: str | Path) -> dict[str, Any]:
    """读取 workspace 内文件内容。"""
    resolved = resolve_workspace_path(workspace, path)
    if not resolved.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    return {
        "path": relative_workspace_path(workspace, resolved),
        "content": resolved.read_text(encoding="utf-8"),
    }


def edit_file(
    *,
    workspace: str | Path,
    path: str | Path,
    old_string: str,
    new_string: str,
) -> dict[str, Any]:
    """用 exactly-once 替换策略修改文件，避免误改重复代码片段。"""
    resolved = resolve_workspace_path(workspace, path)
    if not resolved.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    content = resolved.read_text(encoding="utf-8")
    occurrence_count = content.count(old_string)
    if occurrence_count != 1:
        raise ValueError(f"old_string must appear exactly once, got {occurrence_count} times")
    resolved.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
    return {"path": relative_workspace_path(workspace, resolved), "changed": True}


def write_file(*, workspace: str | Path, path: str | Path, content: str) -> dict[str, Any]:
    """写入 workspace 内文件；用于新建文件或明确需要整体覆盖的场景。"""
    resolved = resolve_workspace_path(workspace, path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return {"path": relative_workspace_path(workspace, resolved), "changed": True}

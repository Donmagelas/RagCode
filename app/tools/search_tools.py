from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from app.tools.path_utils import relative_workspace_path, resolve_workspace_path


def glob_search(*, workspace: str | Path, pattern: str, max_results: int = 200) -> list[str]:
    """按 glob 查找 workspace 内文件，返回相对路径。"""
    workspace_path = resolve_workspace_path(workspace, ".")
    matches = [
        relative_workspace_path(workspace_path, path)
        for path in sorted(workspace_path.glob(pattern))
        if path.is_file()
    ]
    return matches[:max_results]


def grep_search(
    *,
    workspace: str | Path,
    pattern: str,
    include: str = "*",
    max_results: int = 200,
) -> list[dict[str, Any]]:
    """在 workspace 内做正则文本搜索，返回命中行。"""
    workspace_path = resolve_workspace_path(workspace, ".")
    regex = re.compile(pattern)
    results: list[dict[str, Any]] = []
    for path in sorted(workspace_path.rglob("*")):
        if not path.is_file() or not fnmatch.fnmatch(path.name, include):
            continue
        for line_number, line in enumerate(_read_lines(path), start=1):
            if regex.search(line):
                results.append(
                    {
                        "path": relative_workspace_path(workspace_path, path),
                        "line": line_number,
                        "text": line.rstrip("\n"),
                    }
                )
                if len(results) >= max_results:
                    return results
    return results


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        return []

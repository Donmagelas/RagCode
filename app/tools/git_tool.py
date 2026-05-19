from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tools.bash_tool import run_bash


def git_diff(*, workspace: str | Path) -> dict[str, Any]:
    """返回当前 workspace 的 git diff；非 git 仓库时保留 git 的错误信息。"""
    return run_bash(workspace=workspace, command="git diff --")

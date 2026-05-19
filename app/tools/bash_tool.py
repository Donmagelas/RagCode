from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from app.tools.path_utils import resolve_workspace_path

DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bdel\s+/s\b",
    r"\brmdir\s+/s\b",
    r"\bformat\b",
    r"\bcurl\b.*\|\s*bash\b",
    r"\bwget\b.*\|\s*bash\b",
]


def is_dangerous_command(command: str) -> bool:
    """用保守规则拦截明显危险的 shell 命令。"""
    lowered = command.lower()
    return any(re.search(pattern, lowered) for pattern in DANGEROUS_PATTERNS)


def run_bash(
    *,
    workspace: str | Path,
    command: str,
    timeout_seconds: int = 120,
    max_output_chars: int = 12000,
) -> dict[str, Any]:
    """在 workspace 内执行 PowerShell 命令，并返回截断后的 stdout/stderr。"""
    if is_dangerous_command(command):
        raise ValueError(f"Dangerous command blocked: {command}")

    workspace_path = resolve_workspace_path(workspace, ".")
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "exit_code": None,
            "stdout": _truncate(exc.stdout or "", max_output_chars),
            "stderr": _truncate(exc.stderr or "", max_output_chars),
            "timed_out": True,
        }

    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": _truncate(completed.stdout, max_output_chars),
        "stderr": _truncate(completed.stderr, max_output_chars),
        "timed_out": False,
    }


def _truncate(text: str | bytes, max_chars: int) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...<truncated>...\n" + text[-half:]

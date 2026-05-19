from __future__ import annotations

from pathlib import Path


def resolve_workspace_path(workspace: str | Path, path: str | Path) -> Path:
    """解析工具路径，并阻止访问 workspace 外部文件。"""
    workspace_path = Path(workspace).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_path / candidate
    resolved = candidate.resolve()
    if not _is_relative_to(resolved, workspace_path):
        raise ValueError(f"Path is outside workspace: {path}")
    return resolved


def relative_workspace_path(workspace: str | Path, path: str | Path) -> str:
    """统一返回 POSIX 风格相对路径，方便模型消费。"""
    return Path(path).resolve().relative_to(Path(workspace).resolve()).as_posix()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True

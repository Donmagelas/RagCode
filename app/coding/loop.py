from __future__ import annotations

from typing import Any

from app.tools.registry import ToolRegistry


def execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    registry: ToolRegistry,
) -> dict[str, Any]:
    """顺序执行模型计划好的工具调用，并收集 coding loop 状态增量。"""
    tool_results: list[dict[str, Any]] = []
    files_read: list[str] = []
    files_changed: list[str] = []
    commands_run: list[dict[str, Any]] = []

    for tool_call in tool_calls:
        tool_name = str(tool_call["tool"])
        args = dict(tool_call.get("args", {}))
        try:
            output = registry.call(tool_name, **args)
        except Exception as exc:  # noqa: BLE001 - 工具错误需要回灌给模型继续修复。
            tool_results.append(
                {
                    "tool": tool_name,
                    "args": args,
                    "ok": False,
                    "error": str(exc),
                }
            )
            continue

        _record_side_effects(
            tool_name=tool_name,
            output=output,
            files_read=files_read,
            files_changed=files_changed,
            commands_run=commands_run,
        )
        tool_results.append(
            {
                "tool": tool_name,
                "args": args,
                "ok": True,
                "output": output,
            }
        )

    return {
        "tool_results": tool_results,
        "files_read": files_read,
        "files_changed": files_changed,
        "commands_run": commands_run,
    }


def _record_side_effects(
    *,
    tool_name: str,
    output: Any,
    files_read: list[str],
    files_changed: list[str],
    commands_run: list[dict[str, Any]],
) -> None:
    if not isinstance(output, dict):
        return
    path = output.get("path")
    if tool_name == "read_file" and isinstance(path, str):
        files_read.append(path)
    if tool_name in {"edit_file", "write_file"} and isinstance(path, str):
        files_changed.append(path)
    if tool_name == "bash":
        commands_run.append(output)

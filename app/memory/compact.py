from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompactResult:
    summary: str
    messages: list[dict[str, Any]]
    removed_message_count: int


def compact_messages(
    messages: list[dict[str, Any]], *, preserve_recent_messages: int = 8
) -> CompactResult:
    """压缩旧消息，保留最近消息，并避免拆开工具调用/结果对。"""
    if len(messages) <= preserve_recent_messages:
        return CompactResult(summary="", messages=messages, removed_message_count=0)

    keep_from = max(0, len(messages) - preserve_recent_messages)
    keep_from = _move_boundary_before_tool_pair(messages, keep_from)
    removed = messages[:keep_from]
    preserved = messages[keep_from:]
    return CompactResult(
        summary=_summarize_messages(removed),
        messages=preserved,
        removed_message_count=len(removed),
    )


def _move_boundary_before_tool_pair(messages: list[dict[str, Any]], keep_from: int) -> int:
    if keep_from <= 0 or keep_from >= len(messages):
        return keep_from

    first_preserved = messages[keep_from]
    previous = messages[keep_from - 1]
    if first_preserved.get("role") != "tool":
        return keep_from
    if previous.get("role") == "assistant" and previous.get("tool_use_id") == first_preserved.get(
        "tool_use_id"
    ):
        return keep_from - 1
    return keep_from


def _summarize_messages(messages: list[dict[str, Any]]) -> str:
    lines = [
        "Conversation summary:",
        f"- Scope: {len(messages)} earlier messages compacted.",
    ]
    recent_user = [
        str(message.get("content", "")).strip()
        for message in messages
        if message.get("role") == "user" and str(message.get("content", "")).strip()
    ][-3:]
    if recent_user:
        lines.append("- Recent user requests:")
        lines.extend(f"  - {_truncate(text, 160)}" for text in recent_user)
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"

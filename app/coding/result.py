from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from app.coding.agent import CodingAgentResult


RunStatus = Literal["success", "failed", "partial", "cancelled"]


@dataclass(frozen=True)
class ChangedFile:
    path: str
    change_type: str = "modified"


@dataclass(frozen=True)
class CommandResult:
    command: str
    exit_code: int | None
    status: str
    duration_ms: int | None = None


@dataclass(frozen=True)
class TestResult:
    name: str
    status: str
    command: str


@dataclass(frozen=True)
class DiffSummary:
    added: int = 0
    deleted: int = 0
    files: int = 0


@dataclass(frozen=True)
class KnowledgeRunSummary:
    chunk_count: int
    skills: list[str]
    chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunArtifact:
    type: str
    content: str = ""
    path: str = ""


@dataclass(frozen=True)
class CodingRunResult:
    backend: str
    status: RunStatus
    goal: str
    selected_skills: list[str]
    knowledge: KnowledgeRunSummary
    summary: str
    files_read: list[str] = field(default_factory=list)
    files_changed: list[ChangedFile] = field(default_factory=list)
    commands_run: list[CommandResult] = field(default_factory=list)
    test_results: list[TestResult] = field(default_factory=list)
    diff_summary: DiffSummary | None = None
    artifacts: list[RunArtifact] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_backend_result: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def build_builtin_run_result(
    *,
    goal: str,
    selected_skills: list[str],
    knowledge_chunk_count: int,
    agent_result: CodingAgentResult,
) -> CodingRunResult:
    """把内置 coding loop 的原始结果包装成后端无关结构。"""
    files_read: list[str] = []
    files_changed: list[ChangedFile] = []
    commands_run: list[CommandResult] = []
    errors: list[str] = []

    for tool_result in agent_result.tool_results:
        if not tool_result.get("ok", False):
            errors.append(str(tool_result.get("error", "")))
            continue
        output = tool_result.get("output")
        if not isinstance(output, dict):
            continue
        tool_name = str(tool_result.get("tool", ""))
        path = output.get("path")
        if tool_name == "read_file" and isinstance(path, str):
            files_read.append(path)
        if tool_name in {"edit_file", "write_file"} and isinstance(path, str):
            files_changed.append(ChangedFile(path=path, change_type=_change_type(tool_name)))
        if tool_name == "bash":
            commands_run.append(_command_result(output, tool_result))

    test_results = [
        TestResult(name="command", status=command.status, command=command.command)
        for command in commands_run
    ]
    latest_commands = {command.command: command for command in commands_run}
    failed_commands = [command for command in latest_commands.values() if command.status != "passed"]
    if failed_commands:
        status: RunStatus = "failed"
    elif errors:
        status = "partial"
    else:
        status = "success"
    return CodingRunResult(
        backend="builtin",
        status=status,
        goal=goal,
        selected_skills=selected_skills,
        knowledge=KnowledgeRunSummary(chunk_count=knowledge_chunk_count, skills=selected_skills),
        summary=agent_result.final_response,
        files_read=files_read,
        files_changed=files_changed,
        commands_run=commands_run,
        test_results=test_results,
        errors=errors,
        raw_backend_result={
            "message_count": len(agent_result.messages),
            "tool_call_count": len(agent_result.tool_results),
        },
    )


def build_codex_run_result(
    *,
    goal: str,
    selected_skills: list[str],
    knowledge_chunk_count: int,
    context_markdown: str,
) -> CodingRunResult:
    """第一阶段 Codex 适配器只产出上下文包，不直接启动 Codex 编码。"""
    return CodingRunResult(
        backend="codex",
        status="partial",
        goal=goal,
        selected_skills=selected_skills,
        knowledge=KnowledgeRunSummary(chunk_count=knowledge_chunk_count, skills=selected_skills),
        summary="Codex backend is prepared with framework knowledge context; execution is external.",
        artifacts=[RunArtifact(type="context_markdown", content=context_markdown)],
        warnings=["Codex backend is not executed by this project in the current stage."],
    )


def render_run_result_json(result: CodingRunResult) -> str:
    """输出机器可读 JSON，供 Codex 或其他编码端统一消费。"""
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)


def _change_type(tool_name: str) -> str:
    return "created_or_replaced" if tool_name == "write_file" else "modified"


def _command_result(output: dict[str, Any], tool_result: dict[str, Any]) -> CommandResult:
    command = str(output.get("command") or tool_result.get("args", {}).get("command", ""))
    exit_code = output.get("exit_code")
    parsed_exit_code = int(exit_code) if isinstance(exit_code, int) else None
    status = "passed" if parsed_exit_code == 0 else "failed"
    return CommandResult(command=command, exit_code=parsed_exit_code, status=status)

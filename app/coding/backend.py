from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.coding.agent import CodingAgent, CodingAgentResult, CodingModel
from app.coding.result import CodingRunResult, build_builtin_run_result, build_codex_run_result
from app.context.package import KnowledgeContextPackage
from app.tools.registry import ToolRegistry


@dataclass(frozen=True)
class CodingRequest:
    goal: str
    workspace: str
    knowledge_context: KnowledgeContextPackage
    conversation_summary: str
    constraints: list[str]
    verification_commands: list[str]


class CodingBackend(Protocol):
    name: str

    def run(self, request: CodingRequest) -> CodingRunResult:
        """执行编码端并返回统一结构化结果。"""


class PreparedCodexBackend:
    name = "codex"

    def run(self, request: CodingRequest) -> CodingRunResult:
        """当前阶段只准备 Codex 可消费上下文，不直接启动 Codex。"""
        return build_codex_run_result(
            goal=request.goal,
            selected_skills=request.knowledge_context.selected_skills,
            knowledge_chunk_count=len(request.knowledge_context.retrieved),
            context_markdown=request.knowledge_context.context_markdown,
        )


class BuiltinCodingBackend:
    name = "builtin"

    def __init__(
        self,
        *,
        model: CodingModel,
        registry: ToolRegistry,
        max_turns: int = 8,
        max_repair_attempts: int = 1,
    ) -> None:
        self._model = model
        self._registry = registry
        self._max_turns = max_turns
        self._max_repair_attempts = max_repair_attempts

    def run(self, request: CodingRequest) -> CodingRunResult:
        """执行内置模型工具循环，并在结束后运行配置的验证命令。"""
        agent = CodingAgent(model=self._model, registry=self._registry, max_turns=self._max_turns)
        agent_result = agent.run(
            user_goal=request.goal,
            context_markdown=_build_builtin_context_markdown(request),
        )
        verification_results = run_verification_commands(
            registry=self._registry,
            commands=request.verification_commands,
        )
        all_tool_results = [*agent_result.tool_results, *verification_results]
        final_response = agent_result.final_response
        messages = [*agent_result.messages]

        for _attempt in range(self._max_repair_attempts):
            if not _has_failed_verification(verification_results):
                break
            repair_result = agent.run(
                user_goal=_build_repair_goal(request.goal, verification_results),
                context_markdown=_build_builtin_context_markdown(request),
            )
            messages.extend(repair_result.messages)
            all_tool_results.extend(repair_result.tool_results)
            final_response = repair_result.final_response
            verification_results = run_verification_commands(
                registry=self._registry,
                commands=request.verification_commands,
            )
            all_tool_results.extend(verification_results)

        combined_result = CodingAgentResult(
            final_response=final_response,
            messages=messages,
            tool_results=all_tool_results,
        )
        return build_builtin_run_result(
            goal=request.goal,
            selected_skills=request.knowledge_context.selected_skills,
            knowledge_chunk_count=len(request.knowledge_context.retrieved),
            agent_result=combined_result,
        )


def run_verification_commands(
    *, registry: ToolRegistry, commands: list[str]
) -> list[dict[str, object]]:
    """通过 bash 工具顺序执行验证命令，并按工具结果格式返回。"""
    results: list[dict[str, object]] = []
    for command in commands:
        try:
            output = registry.call("bash", command=command)
        except Exception as exc:  # noqa: BLE001 - 验证失败也要进入统一结果。
            results.append(
                {
                    "tool": "bash",
                    "args": {"command": command},
                    "ok": False,
                    "error": str(exc),
                }
            )
            continue
        results.append(
            {
                "tool": "bash",
                "args": {"command": command},
                "ok": True,
                "output": output,
            }
        )
    return results


def _has_failed_verification(results: list[dict[str, object]]) -> bool:
    for result in results:
        if not result.get("ok", False):
            return True
        output = result.get("output")
        if not isinstance(output, dict):
            continue
        if output.get("timed_out") is True:
            return True
        exit_code = output.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return True
        if exit_code is None:
            return True
    return False


def _build_repair_goal(goal: str, verification_results: list[dict[str, object]]) -> str:
    details = "\n\n".join(_format_verification_failure(result) for result in verification_results)
    return (
        f"{goal}\n\n"
        "验证命令失败，请根据下面的验证结果继续查找和修复代码，然后等待下一轮验证：\n\n"
        f"{details}"
    )


def _format_verification_failure(result: dict[str, object]) -> str:
    if not result.get("ok", False):
        return f"tool_error: {result.get('error', '')}"
    output = result.get("output")
    if not isinstance(output, dict):
        return "invalid verification output"
    command = output.get("command", "")
    exit_code = output.get("exit_code", "")
    stdout = str(output.get("stdout", ""))
    stderr = str(output.get("stderr", ""))
    return f"command: {command}\nexit_code: {exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"


def _build_builtin_context_markdown(request: CodingRequest) -> str:
    sections = [request.knowledge_context.context_markdown]
    if request.conversation_summary:
        sections.append("## Conversation Summary\n\n" + request.conversation_summary)
    if request.constraints:
        sections.append("## Constraints\n\n" + "\n".join(f"- {item}" for item in request.constraints))
    if request.verification_commands:
        sections.append(
            "## Verification Commands\n\n"
            + "\n".join(f"- `{command}`" for command in request.verification_commands)
        )
    return "\n\n".join(section for section in sections if section.strip())

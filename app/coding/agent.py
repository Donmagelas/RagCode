from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from app.tools.registry import ToolRegistry
from app.tools.tool_schemas import agent_tool_schemas


@dataclass(frozen=True)
class ModelTurn:
    content: str
    tool_calls: list[dict[str, Any]]


@dataclass(frozen=True)
class CodingAgentResult:
    final_response: str
    messages: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]


class CodingModel(Protocol):
    def next_turn(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        """生成下一轮模型响应。"""


class CodingAgent:
    def __init__(self, *, model: CodingModel, registry: ToolRegistry, max_turns: int = 8) -> None:
        self._model = model
        self._registry = registry
        self._max_turns = max_turns

    def run(self, *, user_goal: str, context_markdown: str = "") -> CodingAgentResult:
        """执行模型工具循环；工具结果会作为 tool 消息继续交给模型。"""
        messages = _initial_messages(user_goal=user_goal, context_markdown=context_markdown)
        tool_results: list[dict[str, Any]] = []

        for _turn_index in range(self._max_turns):
            model_turn = self._model.next_turn(
                messages=[dict(message) for message in messages],
                tools=agent_tool_schemas(),
            )
            messages.append(_assistant_message(model_turn))
            if not model_turn.tool_calls:
                return CodingAgentResult(
                    final_response=model_turn.content,
                    messages=messages,
                    tool_results=tool_results,
                )

            for tool_call in model_turn.tool_calls:
                result = _execute_model_tool_call(tool_call, self._registry)
                tool_results.append(result)
                messages.append(_tool_message(tool_call["id"], result))

        return CodingAgentResult(
            final_response="模型工具循环达到最大轮数，已停止。",
            messages=messages,
            tool_results=tool_results,
        )


def _initial_messages(*, user_goal: str, context_markdown: str) -> list[dict[str, Any]]:
    system_content = (
        "你是面向自研游戏框架的 CodeAgent。编码前只能依据已提供的框架 Markdown 原文，"
        "如果只看到 Skill manifest、还缺少具体 API 或生命周期规则，必须先调用 Skill 获取原文片段。"
        "需要查看或修改代码时使用工具，修改前必须 read_file，修改后尽量运行验证命令。"
    )
    if context_markdown:
        system_content += "\n\n已提供上下文：\n" + context_markdown
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_goal},
    ]


def _assistant_message(model_turn: ModelTurn) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": model_turn.content}
    if model_turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call.get("args", {}), ensure_ascii=False),
                },
            }
            for tool_call in model_turn.tool_calls
        ]
    return message


def _execute_model_tool_call(tool_call: dict[str, Any], registry: ToolRegistry) -> dict[str, Any]:
    tool_name = str(tool_call["name"])
    args = dict(tool_call.get("args", {}))
    try:
        output = registry.call(tool_name, **args)
    except Exception as exc:  # noqa: BLE001 - 工具错误必须回灌给模型继续修。
        return {"tool": tool_name, "args": args, "ok": False, "error": str(exc)}
    return {"tool": tool_name, "args": args, "ok": True, "output": output}


def _tool_message(tool_call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result, ensure_ascii=False),
    }

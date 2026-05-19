from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.coding.agent import ModelTurn


class OpenAICompatibleCodingModel:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def next_turn(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        """调用 OpenAI-compatible chat.completions，并转换成内部 ModelTurn。"""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools or None,
            temperature=0,
        )
        return model_turn_from_openai_message(response.choices[0].message)


def model_turn_from_openai_message(message: Any) -> ModelTurn:
    """解析 OpenAI tool_calls 消息，容忍模型输出非法 JSON 参数。"""
    tool_calls = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        raw_arguments = getattr(tool_call.function, "arguments", "{}") or "{}"
        try:
            args = json.loads(raw_arguments)
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        tool_calls.append(
            {
                "id": tool_call.id,
                "name": tool_call.function.name,
                "args": args,
            }
        )
    return ModelTurn(content=getattr(message, "content", None) or "", tool_calls=tool_calls)

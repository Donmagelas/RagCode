from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

METADATA_KEYS = [
    "module_type",
    "component_name",
    "api_name",
    "usage_type",
    "tags",
    "searchable_keywords",
    "summary",
]


class MetadataExtractor:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def extract(self, raw_markdown: str, heading_path: list[str]) -> dict[str, Any]:
        """用 LLM 提取检索元数据；返回固定 JSON schema。"""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是框架知识文档的元数据抽取器。只输出 JSON，不要解释。"
                        "字段固定为 module_type, component_name, api_name, usage_type, "
                        "tags, searchable_keywords, summary。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "heading_path": heading_path,
                            "raw_markdown": raw_markdown,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {}
        return normalize_metadata(parsed)


def normalize_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """把 LLM 输出规整成固定 schema，避免入库字段漂移。"""
    return {
        "module_type": _string_value(value.get("module_type")),
        "component_name": _string_value(value.get("component_name")),
        "api_name": _string_value(value.get("api_name")),
        "usage_type": _string_value(value.get("usage_type")),
        "tags": _list_value(value.get("tags")),
        "searchable_keywords": _list_value(value.get("searchable_keywords")),
        "summary": _string_value(value.get("summary")),
    }


def _string_value(value: Any) -> str:
    return "" if value is None else str(value)


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]

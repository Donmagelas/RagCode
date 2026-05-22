from __future__ import annotations

import json
import re
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
    def __init__(self, *, api_key: str, base_url: str, model: str, batch_size: int = 8) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=90)
        self._model = model
        self._batch_size = batch_size

    def extract(self, raw_markdown: str, heading_path: list[str]) -> dict[str, Any]:
        """用 LLM 提取检索元数据；返回固定 JSON schema。"""
        return self.extract_many([{"raw_markdown": raw_markdown, "heading_path": heading_path}])[0]

    def extract_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """批量提取语义元数据，减少大量 chunk 入库时的 LLM 往返次数。"""
        return list(self.iter_extract_many(items))

    def iter_extract_many(self, items: list[dict[str, Any]]):
        """按批次产出提取结果，方便 CLI 在长任务中持续显示进度。"""
        results: list[dict[str, Any]] = []
        for batch in _chunk_items(items, self._batch_size):
            results = self._extract_batch(batch)
            yield from results

    def _extract_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload = [
            {
                "index": index,
                "heading_path": item["heading_path"],
                "raw_markdown": item["raw_markdown"],
            }
            for index, item in enumerate(items)
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是自研游戏框架知识库的元数据抽取器。必须只输出一个 JSON 对象，"
                        "第一字符必须是 {，最后字符必须是 }，不要 Markdown，不要解释。"
                        "输出格式固定为：{\"items\":[{\"index\":0,\"module_type\":\"\","
                        "\"component_name\":\"\",\"api_name\":\"\",\"usage_type\":\"\","
                        "\"tags\":[],\"searchable_keywords\":[],\"summary\":\"\"}]}。"
                        "items 数量必须与输入数量一致，index 必须对应输入 index。"
                    ),
                },
                {
                    "role": "user",
                    "content": "请为以下 chunks 提取检索元数据：\n"
                    + json.dumps({"items": payload}, ensure_ascii=False),
                },
            ],
            temperature=0,
            max_tokens=4096,
        )
        content = response.choices[0].message.content or "{}"
        parsed = extract_json_object(content)
        parsed_items = parsed.get("items")
        if not isinstance(parsed_items, list):
            return [normalize_metadata({}) for _item in items]

        by_index = {
            int(item.get("index")): normalize_metadata(item)
            for item in parsed_items
            if isinstance(item, dict) and _is_int_like(item.get("index"))
        }
        return [by_index.get(index, normalize_metadata({})) for index in range(len(items))]


def extract_json_object(text: str) -> dict[str, Any]:
    """从模型输出中提取 JSON 对象，兼容偶发的 ```json 代码块包装。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
        if match:
            stripped = match.group(1)
    elif not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            stripped = stripped[start : end + 1]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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


def _chunk_items(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True

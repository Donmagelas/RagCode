from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI

from app.models.metadata_extractor import extract_json_object
from app.routing.skill_router import SkillManifest


@dataclass(frozen=True)
class SkillSelectionResult:
    selected_skills: list[str]
    reason: str = ""


class SkillSelector(Protocol):
    def select_skills(
        self,
        *,
        goal: str,
        manifests: list[SkillManifest],
        conversation_summary: str = "",
    ) -> SkillSelectionResult | dict[str, Any]:
        """根据用户目标和 skill manifest 选择需要检索的 skill。"""


class OpenAICompatibleSkillSelector:
    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
        self._model = model

    def select_skills(
        self,
        *,
        goal: str,
        manifests: list[SkillManifest],
        conversation_summary: str = "",
    ) -> SkillSelectionResult:
        """只暴露轻量 manifest，让模型输出固定 JSON 选择结果。"""
        prompt = build_skill_selection_prompt(
            goal=goal,
            manifests=manifests,
            conversation_summary=conversation_summary,
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 CodeAgent 的 skill 路由器。只能根据 manifest 选择 skill。"
                        "只输出 JSON，不要解释。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        return parse_skill_selection_json(
            content,
            known_skill_names={manifest.skill_name for manifest in manifests},
        )


def build_skill_selection_prompt(
    *,
    goal: str,
    manifests: list[SkillManifest],
    conversation_summary: str = "",
) -> str:
    """构建模型选 skill 的输入；只包含 manifest，不包含完整知识正文。"""
    manifest_items = [
        {
            "skill_name": manifest.skill_name,
            "description": manifest.description,
            "framework_name": manifest.framework_name,
            "framework_version": manifest.framework_version,
            "tags": manifest.tags,
            "path": str(manifest.path),
        }
        for manifest in manifests
    ]
    return json.dumps(
        {
            "task": "select framework knowledge skills for the user goal",
            "output_schema": {"selected_skills": ["skill_name"], "reason": ""},
            "user_goal": goal,
            "conversation_summary": conversation_summary,
            "skill_manifests": manifest_items,
        },
        ensure_ascii=False,
        indent=2,
    )


def parse_skill_selection_json(text: str, *, known_skill_names: set[str]) -> SkillSelectionResult:
    """解析模型输出，过滤不存在的 skill，避免后续 RAG 指向错误文档。"""
    parsed = extract_json_object(text)
    selected = parsed.get("selected_skills", [])
    if not isinstance(selected, list):
        selected = []
    seen: set[str] = set()
    selected_skills: list[str] = []
    for item in selected:
        skill_name = str(item)
        if skill_name in known_skill_names and skill_name not in seen:
            seen.add(skill_name)
            selected_skills.append(skill_name)
    return SkillSelectionResult(
        selected_skills=selected_skills,
        reason=str(parsed.get("reason", "")),
    )


def manifest_from_mapping(value: dict[str, Any]) -> SkillManifest:
    """把 LangGraph state 中的可序列化 manifest 还原为 SkillManifest。"""
    return SkillManifest(
        skill_name=str(value.get("skill_name", "")),
        description=str(value.get("description", "")),
        framework_name=str(value.get("framework_name", "")),
        framework_version=str(value.get("framework_version", "")),
        tags=[str(tag) for tag in value.get("tags", []) if tag],
        path=Path(str(value.get("path", ""))),
    )

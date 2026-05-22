from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedKnowledge:
    skill_name: str
    file_path: str
    heading_path: list[str]
    raw_markdown: str
    score: float


@dataclass(frozen=True)
class KnowledgeContextPackage:
    goal: str
    selected_skills: list[str]
    retrieved: list[RetrievedKnowledge]
    backend: str = "builtin"
    trace: dict[str, Any] | None = None

    @property
    def context_markdown(self) -> str:
        return render_context_markdown(self)


def build_context_package(
    *,
    goal: str,
    selected_skills: list[str],
    retrieved: list[RetrievedKnowledge],
    backend: str = "builtin",
    trace: dict[str, Any] | None = None,
) -> KnowledgeContextPackage:
    """构建 Codex 和内置 coding loop 都能消费的统一知识上下文包。"""
    return KnowledgeContextPackage(
        goal=goal,
        selected_skills=selected_skills,
        retrieved=retrieved,
        backend=backend,
        trace=trace,
    )


def render_context_markdown(package: KnowledgeContextPackage) -> str:
    """渲染给 coding backend 使用的 Markdown 上下文。"""
    lines = [
        "# User Goal",
        "",
        package.goal,
        "",
        "# Selected Skills",
        "",
    ]
    lines.extend(f"- {skill}" for skill in package.selected_skills)
    lines.extend(["", "# Framework Knowledge", ""])
    for item in package.retrieved:
        lines.extend(
            [
                f"## skill: {item.skill_name}",
                "",
                f"source: {item.file_path}",
                f"heading_path: {' > '.join(item.heading_path)}",
                f"score: {item.score:.6f}",
                "",
                "```md",
                item.raw_markdown,
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "# Coding Constraints",
            "",
            "- 框架 API 只能依据上面的 Markdown 原文使用。",
            "- 如果缺少框架知识，先重新检索，不要猜 API。",
            "- 修改前先阅读相关代码。",
            "- 修改后运行项目验证命令。",
        ]
    )
    return "\n".join(lines)


def render_context_json(package: KnowledgeContextPackage) -> str:
    """渲染机器可读上下文，供后续 Codex/API 集成使用。"""
    return json.dumps(asdict(package), ensure_ascii=False, indent=2)

from __future__ import annotations

from app.routing.skill_router import SkillManifest


def route_skills_by_manifest(goal: str, manifests: list[SkillManifest], *, limit: int = 3) -> list[str]:
    """第一版轻量 skill 路由：用 manifest 文本做关键词匹配，后续可替换成模型路由。"""
    goal_lower = goal.lower()
    scored: list[tuple[int, str]] = []
    for manifest in manifests:
        score = 0
        haystack = " ".join(
            [
                manifest.skill_name,
                manifest.description,
                manifest.framework_name,
                manifest.framework_version,
                *manifest.tags,
            ]
        ).lower()
        for token in _query_tokens(goal_lower):
            if token and token in haystack:
                score += 1
        if score > 0:
            scored.append((score, manifest.skill_name))
    return [skill for _score, skill in sorted(scored, reverse=True)[:limit]]


def _query_tokens(text: str) -> list[str]:
    return [token for token in text.replace("_", " ").split() if len(token) >= 2]

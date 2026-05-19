from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SkillManifest:
    skill_name: str
    description: str
    framework_name: str
    framework_version: str
    tags: list[str]
    path: Path


def discover_skill_manifests(skills_dir: str | Path) -> list[SkillManifest]:
    """扫描目录式和文件式 skill 文档，并保证 skill_name 全局唯一。"""
    root = Path(skills_dir)
    candidates = _discover_markdown_skill_files(root)
    manifests = [_manifest_from_file(path) for path in candidates]

    seen: dict[str, Path] = {}
    for manifest in manifests:
        previous = seen.get(manifest.skill_name)
        if previous is not None:
            raise ValueError(
                f"Duplicate skill_name '{manifest.skill_name}': {previous} and {manifest.path}"
            )
        seen[manifest.skill_name] = manifest.path

    return manifests


def format_skill_manifest_text(manifests: list[SkillManifest]) -> str:
    """只暴露 skill 轻量 manifest，不泄露完整 Markdown 正文。"""
    lines = ["可用 Skill："]
    for manifest in manifests:
        lines.append(
            "- "
            + " | ".join(
                part
                for part in [
                    f"name={manifest.skill_name}",
                    f"description={manifest.description}",
                    f"framework={manifest.framework_name}",
                    f"version={manifest.framework_version}",
                ]
                if part and not part.endswith("=")
            )
        )
    return "\n".join(lines)


def _discover_markdown_skill_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Skills directory does not exist: {root}")

    # 先收集目录式 */SKILL.md，再收集根目录文件式 *.md，保持稳定排序便于调试。
    directory_skills = sorted(path for path in root.glob("*/SKILL.md") if path.is_file())
    file_skills = sorted(path for path in root.glob("*.md") if path.is_file())
    return directory_skills + file_skills


def _manifest_from_file(path: Path) -> SkillManifest:
    text = path.read_text(encoding="utf-8")
    frontmatter, _body = split_frontmatter(text)
    name = _string_value(frontmatter.get("name")) or path.stem
    tags = frontmatter.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        tags = []

    return SkillManifest(
        skill_name=name,
        description=_string_value(frontmatter.get("description")),
        framework_name=_string_value(frontmatter.get("framework_name")),
        framework_version=_string_value(frontmatter.get("framework_version")),
        tags=[str(tag) for tag in tags],
        path=path,
    )


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """解析 Markdown frontmatter；没有 frontmatter 时返回空元数据。"""
    if not text.startswith("---\n"):
        return {}, text

    end_marker = "\n---\n"
    end = text.find(end_marker, 4)
    if end == -1:
        return {}, text

    raw_frontmatter = text[4:end]
    body = text[end + len(end_marker) :]
    parsed = yaml.safe_load(raw_frontmatter) or {}
    if not isinstance(parsed, dict):
        return {}, body
    return parsed, body


def _string_value(value: Any) -> str:
    return "" if value is None else str(value)

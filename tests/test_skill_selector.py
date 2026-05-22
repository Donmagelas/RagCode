from app.routing.model_skill_selector import (
    SkillSelectionResult,
    build_skill_selection_prompt,
    parse_skill_selection_json,
)
from app.routing.skill_router import SkillManifest


def test_parse_skill_selection_json_keeps_known_skills_only() -> None:
    result = parse_skill_selection_json(
        '{"selected_skills":["ui","missing"],"reason":"需要 UI"}',
        known_skill_names={"ui", "animation"},
    )

    assert result == SkillSelectionResult(selected_skills=["ui"], reason="需要 UI")


def test_build_skill_selection_prompt_exposes_manifest_not_full_doc() -> None:
    prompt = build_skill_selection_prompt(
        goal="实现背包 UI",
        manifests=[
            SkillManifest(
                skill_name="ui",
                description="UI 窗口和控件",
                framework_name="agk",
                framework_version="v1",
                tags=["UI"],
                path="ui/SKILL.md",
            )
        ],
        conversation_summary="",
    )

    assert "selected_skills" in prompt
    assert "UI 窗口和控件" in prompt
    assert "Markdown 原文" not in prompt

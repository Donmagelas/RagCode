from pathlib import Path

from app.routing.skill_router import SkillManifest
from app.routing.skill_selection import route_skills_by_manifest


def test_route_skills_by_manifest_matches_goal_terms() -> None:
    manifests = [
        SkillManifest(
            skill_name="05_UI系统",
            description="UI 窗口 面板 控件",
            framework_name="aurora",
            framework_version="v1",
            tags=["窗口"],
            path=Path("ui.md"),
        ),
        SkillManifest(
            skill_name="09_网络同步",
            description="NetworkTick Ghost",
            framework_name="aurora",
            framework_version="v1",
            tags=[],
            path=Path("net.md"),
        ),
    ]

    selected = route_skills_by_manifest("实现 UI 窗口", manifests)

    assert selected == ["05_UI系统"]

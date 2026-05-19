from pathlib import Path

from app.routing.skill_router import discover_skill_manifests, format_skill_manifest_text


def test_discover_skill_manifests_supports_file_and_directory_skills(tmp_path: Path) -> None:
    file_skill = tmp_path / "05_UI系统.md"
    file_skill.write_text("# UI 系统\n\nUI 文档。", encoding="utf-8")

    dir_skill_dir = tmp_path / "animation"
    dir_skill_dir.mkdir()
    dir_skill = dir_skill_dir / "SKILL.md"
    dir_skill.write_text(
        "\n".join(
            [
                "---",
                "name: animation",
                "description: 动画系统",
                "framework_name: aurora_gamekit",
                "framework_version: v1",
                "tags:",
                "  - 动画",
                "---",
                "# 动画",
            ]
        ),
        encoding="utf-8",
    )

    manifests = discover_skill_manifests(tmp_path)

    by_name = {manifest.skill_name: manifest for manifest in manifests}
    assert set(by_name) == {"05_UI系统", "animation"}
    assert by_name["05_UI系统"].path == file_skill
    assert by_name["05_UI系统"].description == ""
    assert by_name["animation"].path == dir_skill
    assert by_name["animation"].description == "动画系统"
    assert by_name["animation"].tags == ["动画"]


def test_discover_skill_manifests_rejects_duplicate_skill_names(tmp_path: Path) -> None:
    (tmp_path / "ui.md").write_text(
        "---\nname: ui\n---\n# UI A\n",
        encoding="utf-8",
    )
    other_dir = tmp_path / "ui_dir"
    other_dir.mkdir()
    (other_dir / "SKILL.md").write_text(
        "---\nname: ui\n---\n# UI B\n",
        encoding="utf-8",
    )

    try:
        discover_skill_manifests(tmp_path)
    except ValueError as exc:
        assert "Duplicate skill_name" in str(exc)
    else:
        raise AssertionError("duplicate skill_name should fail")


def test_format_skill_manifest_text_exposes_only_lightweight_fields(tmp_path: Path) -> None:
    skill_file = tmp_path / "ui.md"
    skill_file.write_text(
        "---\nname: ui\ndescription: UI 系统\nframework_name: aurora\nframework_version: v1\n---\n# Secret API\n",
        encoding="utf-8",
    )
    manifests = discover_skill_manifests(tmp_path)

    text = format_skill_manifest_text(manifests)

    assert "ui" in text
    assert "UI 系统" in text
    assert "Secret API" not in text

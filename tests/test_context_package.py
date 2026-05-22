from app.context.package import RetrievedKnowledge, build_context_package, render_context_markdown


def test_build_context_package_renders_framework_knowledge() -> None:
    package = build_context_package(
        goal="实现背包 UI",
        selected_skills=["05_UI系统"],
        retrieved=[
            RetrievedKnowledge(
                skill_name="05_UI系统",
                file_path="aurora_gamekit_rag_md_docs/05_UI系统.md",
                heading_path=["UI 系统", "UIPanel"],
                raw_markdown="## UIPanel\n\n面板说明。",
                score=0.12,
            )
        ],
    )

    rendered = render_context_markdown(package)

    assert package.goal == "实现背包 UI"
    assert package.selected_skills == ["05_UI系统"]
    assert "# User Goal" in rendered
    assert "## skill: 05_UI系统" in rendered
    assert "面板说明。" in rendered
    assert "框架 API 只能依据上面的 Markdown 原文使用" in rendered

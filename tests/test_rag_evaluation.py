from app.rag.evaluation import (
    EvaluationCaseResult,
    RagEvaluationCase,
    build_evaluation_report,
    evaluate_case_result,
    load_evaluation_cases,
    render_evaluation_report,
)
from app.rag.retriever import RetrievedChunk


def test_evaluate_case_result_marks_doc_path_quote_and_distractor() -> None:
    case = RagEvaluationCase(
        case_id="001",
        skill_name="01_启动与生命周期",
        query="OnBoot 里可以加载资源吗？",
        expected_doc="01_启动与生命周期.md",
        expected_heading_path="启动与生命周期 > Module > OnBoot > 执行时机 > 可访问系统",
        expected_quote="此时不可访问 Scene、World、AssetDatabase、RenderDevice。",
        distractor="OnLoad 可以访问资源系统",
    )
    chunks = [
        RetrievedChunk(
            id="chunk-1",
            doc_id="doc-1",
            skill_name="01_启动与生命周期",
            file_path=r"aurora_gamekit_rag_md_docs\01_启动与生命周期.md",
            heading="可访问系统",
            heading_path=["启动与生命周期", "Module", "OnBoot", "执行时机", "可访问系统"],
            sort_order=10,
            raw_markdown="###### 可访问系统\n\n此时不可访问 Scene、World、AssetDatabase、RenderDevice。",
            score=0.1,
        )
    ]

    result = evaluate_case_result(case, chunks)

    assert result.passed is True
    assert result.doc_hit is True
    assert result.heading_path_hit is True
    assert result.quote_hit is True
    assert result.distractor_hit is False
    assert result.top_heading_path == "启动与生命周期 > Module > OnBoot > 执行时机 > 可访问系统"


def test_build_evaluation_report_counts_pass_rate() -> None:
    results = [
        EvaluationCaseResult(
            case_id="001",
            query="q1",
            expected_doc="a.md",
            expected_heading_path="A",
            expected_quote="quote",
            top_doc="a.md",
            top_heading_path="A",
            doc_hit=True,
            heading_path_hit=True,
            quote_hit=True,
            distractor_hit=False,
            passed=True,
        ),
        EvaluationCaseResult(
            case_id="002",
            query="q2",
            expected_doc="b.md",
            expected_heading_path="B",
            expected_quote="quote",
            top_doc="c.md",
            top_heading_path="C",
            doc_hit=False,
            heading_path_hit=False,
            quote_hit=False,
            distractor_hit=False,
            passed=False,
        ),
    ]

    report = build_evaluation_report(results)

    assert report.total == 2
    assert report.passed == 1
    assert report.pass_rate == 0.5
    rendered = render_evaluation_report(report)
    assert "RAG evaluation report" in rendered
    assert "001" in rendered
    assert "002" in rendered


def test_evaluate_case_result_ignores_inline_code_backticks_for_quote() -> None:
    case = RagEvaluationCase(
        case_id="003",
        skill_name="00_术语与设计约定",
        query="AssetRef 和 AssetHandle 有什么区别？",
        expected_doc="00_术语与设计约定.md",
        expected_heading_path="术语与设计约定 > AssetHandle > AssetRef",
        expected_quote="AssetRef<T> 只保存资源虚拟路径和资源类型，不触发加载，也不增加引用计数。",
        distractor="",
    )
    chunks = [
        RetrievedChunk(
            id="chunk-3",
            doc_id="doc-3",
            skill_name="00_术语与设计约定",
            file_path=r"aurora_gamekit_rag_md_docs\00_术语与设计约定.md",
            heading="与 AssetHandle 的差异",
            heading_path=["术语与设计约定", "AssetHandle", "AssetRef"],
            sort_order=3,
            raw_markdown=(
                "###### 与 AssetHandle 的差异\n\n"
                "`AssetRef<T>` 只保存资源虚拟路径和资源类型，不触发加载，也不增加引用计数。"
            ),
            score=0.2,
        )
    ]

    result = evaluate_case_result(case, chunks)

    assert result.quote_hit is True
    assert result.passed is True


def test_load_evaluation_cases_reads_generic_jsonl(tmp_path) -> None:
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(
        (
            '{"case_id":"001","skill_name":"ui","query":"按钮点击",'
            '"expected_doc":"05_UI系统.md","expected_heading_path":"UI 系统 > Widget",'
            '"expected_quote":"不会被 EventBus 订阅","distractor":"全局事件"}\n'
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(cases_file)

    assert cases == [
        RagEvaluationCase(
            case_id="001",
            skill_name="ui",
            query="按钮点击",
            expected_doc="05_UI系统.md",
            expected_heading_path="UI 系统 > Widget",
            expected_quote="不会被 EventBus 订阅",
            distractor="全局事件",
        )
    ]


def test_evaluate_case_result_reports_highest_score_chunk_as_top() -> None:
    case = RagEvaluationCase(
        case_id="007",
        skill_name="12_物理碰撞",
        query="Trigger 会阻挡物体吗？",
        expected_doc="12_物理碰撞.md",
        expected_heading_path="物理碰撞 > Trigger > TriggerCollider > isTrigger > 行为 > 不产生物理阻挡",
        expected_quote="不产生物理阻挡",
    )
    chunks = [
        RetrievedChunk(
            id="root",
            doc_id="doc",
            skill_name="12_物理碰撞",
            file_path=r"aurora_gamekit_rag_md_docs\12_物理碰撞.md",
            heading="物理碰撞",
            heading_path=["物理碰撞"],
            sort_order=1,
            raw_markdown="# 物理碰撞",
            score=0.03,
        ),
        RetrievedChunk(
            id="hit",
            doc_id="doc",
            skill_name="12_物理碰撞",
            file_path=r"aurora_gamekit_rag_md_docs\12_物理碰撞.md",
            heading="不产生物理阻挡",
            heading_path=["物理碰撞", "Trigger", "TriggerCollider", "isTrigger", "行为", "不产生物理阻挡"],
            sort_order=20,
            raw_markdown="###### 不产生物理阻挡\n\n当 Collider 的 `isTrigger = true` 时，不产生物理阻挡。",
            score=0.06,
        ),
    ]

    result = evaluate_case_result(case, chunks)

    assert result.top_heading_path == "物理碰撞 > Trigger > TriggerCollider > isTrigger > 行为 > 不产生物理阻挡"

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class RagEvaluationCase:
    case_id: str
    skill_name: str
    query: str
    expected_doc: str
    expected_heading_path: str
    expected_quote: str
    distractor: str = ""


@dataclass(frozen=True)
class EvaluationCaseResult:
    case_id: str
    query: str
    expected_doc: str
    expected_heading_path: str
    expected_quote: str
    top_doc: str
    top_heading_path: str
    doc_hit: bool
    heading_path_hit: bool
    quote_hit: bool
    distractor_hit: bool
    passed: bool


@dataclass(frozen=True)
class EvaluationReport:
    total: int
    passed: int
    pass_rate: float
    results: list[EvaluationCaseResult]


def evaluate_case_result(
    sample_case: RagEvaluationCase, chunks: list[RetrievedChunk]
) -> EvaluationCaseResult:
    """评估单条 RAG 用例是否命中文档、标题路径和关键原文。"""
    # retrieve_chunks 最终会按文档顺序输出给模型；评估报告里的 top 应展示最高分命中块。
    top_chunk = max(chunks, key=lambda chunk: chunk.score) if chunks else None
    top_doc = _doc_name(top_chunk.file_path) if top_chunk else ""
    top_heading_path = _join_heading_path(top_chunk.heading_path) if top_chunk else ""
    expected_heading_path = _normalize_text(sample_case.expected_heading_path)
    expected_quote = _normalize_text(sample_case.expected_quote)
    distractor = _normalize_text(sample_case.distractor)

    doc_hit = any(_doc_name(chunk.file_path) == sample_case.expected_doc for chunk in chunks)
    heading_path_hit = any(
        expected_heading_path == _normalize_text(_join_heading_path(chunk.heading_path))
        or expected_heading_path in _normalize_text(_join_heading_path(chunk.heading_path))
        for chunk in chunks
    )
    quote_hit = any(expected_quote in _normalize_text(chunk.raw_markdown) for chunk in chunks)
    # 干扰项是测试集中记录的误召回线索，命中它通常说明上下文混入了不该优先给模型的片段。
    distractor_hit = bool(distractor) and any(
        distractor in _normalize_text(chunk.raw_markdown) for chunk in chunks
    )
    passed = doc_hit and heading_path_hit and quote_hit and not distractor_hit

    return EvaluationCaseResult(
        case_id=sample_case.case_id,
        query=sample_case.query,
        expected_doc=sample_case.expected_doc,
        expected_heading_path=sample_case.expected_heading_path,
        expected_quote=sample_case.expected_quote,
        top_doc=top_doc,
        top_heading_path=top_heading_path,
        doc_hit=doc_hit,
        heading_path_hit=heading_path_hit,
        quote_hit=quote_hit,
        distractor_hit=distractor_hit,
        passed=passed,
    )


def load_evaluation_cases(path: str | Path, *, limit: int | None = None) -> list[RagEvaluationCase]:
    """从通用 JSON/JSONL 文件读取评估用例，不绑定任何特定测试 Markdown 格式。"""
    cases_path = Path(path)
    text = cases_path.read_text(encoding="utf-8")
    if cases_path.suffix.lower() == ".jsonl":
        raw_items = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        raw_items = parsed["cases"] if isinstance(parsed, dict) and "cases" in parsed else parsed
    cases = [_case_from_mapping(item) for item in raw_items]
    return cases[:limit] if limit is not None else cases


def build_evaluation_report(results: list[EvaluationCaseResult]) -> EvaluationReport:
    """汇总多条评估结果，输出总数、通过数和通过率。"""
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    pass_rate = passed / total if total else 0.0
    return EvaluationReport(total=total, passed=passed, pass_rate=pass_rate, results=results)


def render_evaluation_report(report: EvaluationReport) -> str:
    """渲染给 CLI 阅读的简洁评估表。"""
    lines = [
        "RAG evaluation report",
        f"total={report.total} passed={report.passed} pass_rate={report.pass_rate:.2%}",
        "",
        "case | pass | doc | path | quote | distractor | top",
        "--- | --- | --- | --- | --- | --- | ---",
    ]
    for result in report.results:
        lines.append(
            " | ".join(
                [
                    result.case_id,
                    _mark(result.passed),
                    _mark(result.doc_hit),
                    _mark(result.heading_path_hit),
                    _mark(result.quote_hit),
                    _mark(not result.distractor_hit),
                    f"{result.top_doc} / {result.top_heading_path}",
                ]
            )
        )
    return "\n".join(lines)


def render_evaluation_json(report: EvaluationReport) -> str:
    """渲染机器可读 JSON，便于后续接入调参脚本。"""
    return json.dumps(asdict(report), ensure_ascii=False, indent=2)


def _doc_name(file_path: str) -> str:
    return Path(file_path).name


def _join_heading_path(heading_path: list[str]) -> str:
    return " > ".join(heading_path)


def _normalize_text(text: str) -> str:
    # 评估关键原文时忽略 Markdown inline code 的反引号，避免格式标记影响内容命中判断。
    return " ".join(text.replace("`", "").split())


def _mark(value: bool) -> str:
    return "yes" if value else "no"


def _case_from_mapping(item: dict[str, object]) -> RagEvaluationCase:
    required_fields = [
        "case_id",
        "skill_name",
        "query",
        "expected_doc",
        "expected_heading_path",
        "expected_quote",
    ]
    missing = [field for field in required_fields if not item.get(field)]
    if missing:
        raise ValueError(f"Missing evaluation case fields: {', '.join(missing)}")
    return RagEvaluationCase(
        case_id=str(item["case_id"]),
        skill_name=str(item["skill_name"]),
        query=str(item["query"]),
        expected_doc=str(item["expected_doc"]),
        expected_heading_path=str(item["expected_heading_path"]),
        expected_quote=str(item["expected_quote"]),
        distractor=str(item.get("distractor", "")),
    )

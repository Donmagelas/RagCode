from __future__ import annotations

import os
import sys
from enum import Enum
from pathlib import Path

import psycopg
import typer
from alembic import command
from alembic.config import Config

from app.context.package import (
    RetrievedKnowledge,
    build_context_package,
    render_context_json,
    render_context_markdown,
)
from app.coding.backend import BuiltinCodingBackend, CodingRequest, PreparedCodexBackend
from app.coding.result import (
    render_run_result_json,
)
from app.models.chat_model import OpenAICompatibleCodingModel
from app.models.embeddings import EmbeddingClient
from app.models.metadata_extractor import MetadataExtractor
from app.rag.evaluation import (
    build_evaluation_report,
    evaluate_case_result,
    load_evaluation_cases,
    render_evaluation_json,
    render_evaluation_report,
)
from app.rag.ingest import (
    apply_embeddings,
    apply_metadata_extraction,
    apply_record_cache,
    build_ingest_records,
    build_ingest_report,
    ingest_records,
    load_record_cache,
    render_ingest_report,
)
from app.rag.metadata_enrich import enrich_missing_metadata
from app.rag.human_review import parse_selected_indexes
from app.rag.retriever import SkillRetrieveTrace, record_human_review, retrieve_chunks
from app.rag.token_counter import Qwen3TokenCounter
from app.routing.skill_selection import route_skills_by_manifest
from app.routing.skill_router import discover_skill_manifests, format_skill_manifest_text
from app.settings import load_app_settings
from app.tools.registry import create_default_tool_registry
from app.tools.skill_tool import SkillTool

app = typer.Typer(help="CodeAgent first-stage RAG and coding assistant CLI.")


class ContextFormat(str, Enum):
    markdown = "markdown"
    json = "json"


class EvalFormat(str, Enum):
    table = "table"
    json = "json"


class AgentRunFormat(str, Enum):
    text = "text"
    json = "json"


class CodingBackend(str, Enum):
    builtin = "builtin"
    codex = "codex"


@app.command("init-db")
def init_db(config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径")) -> None:
    """初始化 PostgreSQL 扩展和基础表结构。"""
    settings = load_app_settings(config)
    database_url = _psycopg_url(settings.database.url)

    with psycopg.connect(
        database_url,
        autocommit=True,
        connect_timeout=settings.database.connect_timeout_seconds,
    ) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    alembic_config = Config("alembic.ini")
    alembic_config.set_main_option("sqlalchemy.url", settings.database.url)
    command.upgrade(alembic_config, "head")

    typer.echo("Database initialized.")


@app.command()
def ingest(
    skills_dir: Path = typer.Option(..., "--skills-dir", help="skill 文档目录"),
    prune: bool = typer.Option(False, "--prune", help="删除已不存在的旧文档"),
    with_metadata: bool = typer.Option(False, "--with-metadata", help="调用 LLM 提取元数据"),
    with_embeddings: bool = typer.Option(False, "--with-embeddings", help="调用 embedding 生成向量"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只解析并输出报告，不写入数据库"),
    report: bool = typer.Option(False, "--report", help="输出 chunk、token 和 warning 汇总报告"),
    config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径"),
) -> None:
    """解析 skill 文档并写入知识库。"""
    settings = load_app_settings(config)
    token_counter = Qwen3TokenCounter(settings.models.tokenizer_model)
    records = build_ingest_records(
        skills_dir,
        max_chunk_tokens=settings.rag.max_chunk_tokens,
        chunk_overlap_tokens=settings.rag.chunk_overlap_tokens,
        min_chunk_tokens=settings.rag.min_chunk_tokens,
        token_counter=token_counter,
    )
    ingest_report = build_ingest_report(records)
    if dry_run:
        typer.echo(render_ingest_report(ingest_report))
        typer.echo("Dry run complete; database was not changed.")
        return

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key and (with_metadata or with_embeddings):
        apply_record_cache(
            records,
            load_record_cache(
                settings.database.url,
                connect_timeout_seconds=settings.database.connect_timeout_seconds,
            ),
        )
    if api_key and with_metadata:
        metadata_extractor = MetadataExtractor(
            api_key=api_key,
            base_url=settings.models.dashscope_base_url,
            model=settings.models.chat_model,
            batch_size=settings.models.metadata_batch_size,
        )
        apply_metadata_extraction(
            records,
            metadata_extractor,
            on_progress=lambda index, total, heading_path: typer.echo(
                f"metadata {index}/{total}: {heading_path}"
            ),
        )
    elif with_metadata:
        typer.echo("DASHSCOPE_API_KEY is not set; skip metadata extraction.")

    if api_key and with_embeddings:
        embedding_client = EmbeddingClient(
            api_key=api_key,
            base_url=settings.models.dashscope_base_url,
            model=settings.models.embedding_model,
            embedding_dim=settings.models.embedding_dim,
        )
        apply_embeddings(records, embedding_client)
    elif with_embeddings:
        typer.echo("DASHSCOPE_API_KEY is not set; skip embedding generation.")
    if report:
        typer.echo(render_ingest_report(ingest_report))
    ingest_records(
        settings.database.url,
        records,
        prune=prune,
        connect_timeout_seconds=settings.database.connect_timeout_seconds,
    )
    typer.echo(f"Ingested {len(records.docs)} docs and {len(records.chunks)} chunks.")


@app.command()
def retrieve(
    skill: str = typer.Option(...),
    query: str = typer.Option(...),
    config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径"),
) -> None:
    """对指定 skill 执行内部 RAG。"""
    settings = load_app_settings(config)
    embeddings = _query_embeddings_if_available(query, settings)
    token_counter = Qwen3TokenCounter(settings.models.tokenizer_model)
    chunks = retrieve_chunks(
        settings.database.url,
        skill_name=skill,
        query=query,
        top_k=settings.rag.max_final_chunks,
        rrf_k=settings.rag.rrf_k,
        retriever_top_k=settings.rag.retriever_top_k,
        seed_top_n=settings.rag.seed_top_n,
        seed_threshold_ratio=settings.rag.seed_threshold_ratio,
        expand_threshold_ratio=settings.rag.expand_threshold_ratio,
        query_expansion_max_terms=settings.rag.query_expansion_max_terms,
        max_depth=settings.rag.max_depth,
        max_context_tokens=settings.rag.max_context_tokens,
        token_counter=token_counter,
        query_embedding=embeddings,
        metadata_query_embedding=embeddings,
        connect_timeout_seconds=settings.database.connect_timeout_seconds,
    )
    for chunk in chunks:
        typer.echo(f"## {chunk.skill_name} / {' > '.join(chunk.heading_path)}")
        typer.echo(f"score: {chunk.score:.6f}")
        typer.echo(chunk.raw_markdown)
        typer.echo("")


@app.command("debug-rag")
def debug_rag(
    skill: str = typer.Option(...),
    query: str = typer.Option(...),
    human_review: bool = typer.Option(False, "--human-review", help="开启人工确认"),
    trace: bool = typer.Option(False, "--trace", help="输出四路召回、RRF 和结构扩展过程"),
    config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径"),
) -> None:
    """输出 RAG 调试信息。"""
    settings = load_app_settings(config)
    embeddings = _query_embeddings_if_available(query, settings)
    token_counter = Qwen3TokenCounter(settings.models.tokenizer_model)
    retrieve_result = retrieve_chunks(
        settings.database.url,
        skill_name=skill,
        query=query,
        top_k=settings.rag.max_final_chunks,
        rrf_k=settings.rag.rrf_k,
        retriever_top_k=settings.rag.retriever_top_k,
        seed_top_n=settings.rag.seed_top_n,
        seed_threshold_ratio=settings.rag.seed_threshold_ratio,
        expand_threshold_ratio=settings.rag.expand_threshold_ratio,
        query_expansion_max_terms=settings.rag.query_expansion_max_terms,
        max_depth=settings.rag.max_depth,
        max_context_tokens=settings.rag.max_context_tokens,
        token_counter=token_counter,
        query_embedding=embeddings,
        metadata_query_embedding=embeddings,
        return_trace=trace,
        connect_timeout_seconds=settings.database.connect_timeout_seconds,
    )
    if trace:
        chunks = retrieve_result.chunks
        retrieve_trace = retrieve_result.trace
    else:
        chunks = retrieve_result
        retrieve_trace = None
    typer.echo(f"debug-rag skill={skill} query={query} human_review={human_review}")
    typer.echo(f"chunks={len(chunks)}")
    for index, chunk in enumerate(chunks, start=1):
        typer.echo(f"{index}. score={chunk.score:.6f} path={' > '.join(chunk.heading_path)}")
        typer.echo(f"   file={chunk.file_path}")
    if human_review and chunks:
        selected_text = typer.prompt(
            "输入要保留的 chunk 序号，逗号分隔；直接回车表示全部保留",
            default="",
            show_default=False,
        )
        selected_indexes = parse_selected_indexes(selected_text, max_index=len(chunks))
        if selected_indexes:
            chunks = [chunks[index - 1] for index in selected_indexes]
        if retrieve_trace is not None:
            record_human_review(
                retrieve_trace,
                selected_indexes=selected_indexes,
                approved_chunk_ids=[chunk.id for chunk in chunks],
            )
        typer.echo(f"approved_chunks={len(chunks)}")
        for index, chunk in enumerate(chunks, start=1):
            typer.echo(f"approved {index}. {' > '.join(chunk.heading_path)}")
    if retrieve_trace is not None:
        typer.echo(render_retrieval_trace(retrieve_trace))


def render_retrieval_trace(trace: SkillRetrieveTrace) -> str:
    """渲染 RAG trace，帮助调试四路召回、RRF 和结构扩展。"""
    lines: list[str] = []
    lines.append("trace:")
    lines.append(f"expanded_queries: {', '.join(trace.expanded_queries)}")
    for route in ["content_fts", "metadata_fts", "content_vector", "metadata_vector"]:
        hits = trace.route_results.get(route, [])
        lines.append(f"route: {route}")
        for hit in hits[:10]:
            label = " > ".join(hit.heading_path) if hit.heading_path else hit.chunk_id
            lines.append(f"{hit.rank}. [{hit.query}] {label}")

    lines.append("RRF top:")
    for index, item in enumerate(trace.rrf_results[:10], start=1):
        lines.append(f"{index}. {item.chunk_id} score={item.score:.6f}")

    lines.append("seed:")
    for index, item in enumerate(trace.seed_chunks[:10], start=1):
        lines.append(f"{index}. {item.chunk_id} score={item.score:.6f}")

    lines.append("expanded:")
    for item in trace.expanded_chunks[:20]:
        label = " > ".join(item.heading_path) if item.heading_path else item.chunk_id
        lines.append(f"+ {item.relation}: {label} <- {item.source_chunk_id} score={item.score:.6f}")

    lines.append("final:")
    for index, chunk_id in enumerate(trace.final_chunk_ids, start=1):
        lines.append(f"{index}. {chunk_id}")

    if trace.human_review is not None:
        lines.append("human_review:")
        lines.append(f"selected_indexes: {trace.human_review['selected_indexes']}")
        lines.append(f"approved_chunk_ids: {trace.human_review['approved_chunk_ids']}")
    return "\n".join(lines)


@app.command("route-skill")
def route_skill(
    goal: str = typer.Option(..., "--goal", help="用户需求"),
    skills_dir: Path = typer.Option(
        Path("aurora_gamekit_rag_md_docs"), "--skills-dir", help="skill 文档目录"
    ),
    limit: int = typer.Option(3, "--limit", help="最多返回 skill 数"),
) -> None:
    """根据 skill manifest 做轻量路由，输出候选 skill。"""
    manifests = discover_skill_manifests(skills_dir)
    selected = route_skills_by_manifest(goal, manifests, limit=limit)
    for skill_name in selected:
        typer.echo(skill_name)


@app.command("prepare-context")
def prepare_context(
    goal: str = typer.Option(..., "--goal", help="用户编码需求"),
    skill: list[str] | None = typer.Option(None, "--skill", help="要使用的 skill，可重复传入"),
    skills_dir: Path = typer.Option(
        Path("aurora_gamekit_rag_md_docs"), "--skills-dir", help="skill 文档目录"
    ),
    auto_route: bool = typer.Option(False, "--auto-route", help="根据 manifest 自动选择 skill"),
    human_review: bool = typer.Option(False, "--human-review", help="生成上下文前人工确认 chunk"),
    backend: CodingBackend = typer.Option(CodingBackend.builtin, "--backend", help="目标编程后端"),
    output_format: ContextFormat = typer.Option(
        ContextFormat.markdown, "--format", help="上下文输出格式"
    ),
    output: Path | None = typer.Option(None, "--output", help="写入输出文件"),
    config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径"),
) -> None:
    """生成 Codex 和内置 coding loop 都可消费的知识上下文包。"""
    settings = load_app_settings(config)
    selected_skills = list(skill or [])
    if auto_route and not selected_skills:
        selected_skills = route_skills_by_manifest(goal, discover_skill_manifests(skills_dir))
    if not selected_skills:
        raise typer.BadParameter("At least one --skill is required unless --auto-route is enabled.")

    skill_tool = SkillTool(settings=settings, api_key=os.getenv("DASHSCOPE_API_KEY"))
    retrieved: list[RetrievedKnowledge] = []
    for skill_name in selected_skills:
        result = skill_tool.retrieve(skill=skill_name, query=goal)
        chunks = result["chunks"]
        if human_review and chunks:
            chunks = _review_serialized_chunks(chunks)
        for chunk in chunks:
            retrieved.append(
                RetrievedKnowledge(
                    skill_name=str(chunk["skill_name"]),
                    file_path=str(chunk.get("file_path", "")),
                    heading_path=list(chunk["heading_path"]),
                    raw_markdown=str(chunk["raw_markdown"]),
                    score=float(chunk["score"]),
                )
            )

    package = build_context_package(
        goal=goal,
        selected_skills=selected_skills,
        retrieved=retrieved,
        backend=backend.value,
    )
    rendered = (
        render_context_json(package)
        if output_format == ContextFormat.json
        else render_context_markdown(package)
    )
    if output is not None:
        output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        typer.echo(rendered)


@app.command("eval-rag")
def eval_rag(
    cases_file: Path = typer.Option(..., "--cases-file", help="通用 RAG 评估用例 JSON/JSONL"),
    limit: int | None = typer.Option(None, "--limit", help="最多评估多少条用例"),
    output_format: EvalFormat = typer.Option(EvalFormat.table, "--format", help="输出格式"),
    output: Path | None = typer.Option(None, "--output", help="写入输出文件"),
    config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径"),
) -> None:
    """用测试用例批量评估当前 RAG 检索质量。"""
    settings = load_app_settings(config)
    token_counter = Qwen3TokenCounter(settings.models.tokenizer_model)
    cases = load_evaluation_cases(cases_file, limit=limit)
    results = []
    for sample_case in cases:
        query_embedding = _query_embeddings_if_available(sample_case.query, settings)
        chunks = retrieve_chunks(
            settings.database.url,
            skill_name=sample_case.skill_name,
            query=sample_case.query,
            top_k=settings.rag.max_final_chunks,
            rrf_k=settings.rag.rrf_k,
            retriever_top_k=settings.rag.retriever_top_k,
            seed_top_n=settings.rag.seed_top_n,
            seed_threshold_ratio=settings.rag.seed_threshold_ratio,
            expand_threshold_ratio=settings.rag.expand_threshold_ratio,
            query_expansion_max_terms=settings.rag.query_expansion_max_terms,
            max_depth=settings.rag.max_depth,
            max_context_tokens=settings.rag.max_context_tokens,
            token_counter=token_counter,
            query_embedding=query_embedding,
            metadata_query_embedding=query_embedding,
            connect_timeout_seconds=settings.database.connect_timeout_seconds,
        )
        results.append(evaluate_case_result(sample_case, chunks))

    report = build_evaluation_report(results)
    rendered = (
        render_evaluation_json(report)
        if output_format == EvalFormat.json
        else render_evaluation_report(report)
    )
    if output is not None:
        output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        typer.echo(rendered)


@app.command("enrich-metadata")
def enrich_metadata(
    skill: str | None = typer.Option(None, "--skill", help="只处理指定 skill"),
    limit: int | None = typer.Option(None, "--limit", help="最多处理多少个 chunk"),
    workers: int = typer.Option(4, "--workers", help="并发 LLM 请求数"),
    config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径"),
) -> None:
    """对已入库 chunk 做可恢复的 LLM 语义 metadata 增强。"""
    settings = load_app_settings(config)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise typer.BadParameter("DASHSCOPE_API_KEY is not set.")
    embedding_client = EmbeddingClient(
        api_key=api_key,
        base_url=settings.models.dashscope_base_url,
        model=settings.models.embedding_model,
        embedding_dim=settings.models.embedding_dim,
    )
    result = enrich_missing_metadata(
        settings.database.url,
        api_key=api_key,
        base_url=settings.models.dashscope_base_url,
        chat_model=settings.models.chat_model,
        embedding_client=embedding_client,
        skill_name=skill,
        limit=limit,
        workers=workers,
        connect_timeout_seconds=settings.database.connect_timeout_seconds,
        on_progress=typer.echo,
    )
    typer.echo(
        (
            f"metadata enrich complete: selected={result.selected} "
            f"enriched={result.enriched} failed={result.failed} embedded={result.embedded}"
        )
    )


@app.command("agent-run")
def agent_run(
    goal: str = typer.Option(..., "--goal", help="用户编码需求"),
    skill: list[str] | None = typer.Option(None, "--skill", help="要使用的 skill，可重复传入"),
    skills_dir: Path = typer.Option(
        Path("aurora_gamekit_rag_md_docs"), "--skills-dir", help="skill 文档目录"
    ),
    workspace: Path = typer.Option(Path("."), "--workspace", help="要修改的项目 workspace"),
    backend: CodingBackend = typer.Option(CodingBackend.builtin, "--backend", help="编程后端"),
    max_turns: int = typer.Option(8, "--max-turns", help="模型工具循环最大轮数"),
    max_repair_attempts: int = typer.Option(
        1, "--max-repair-attempts", help="验证失败后最多继续修复次数"
    ),
    verify: list[str] | None = typer.Option(None, "--verify", help="验证命令，可重复传入"),
    show_tools: bool = typer.Option(False, "--show-tools", help="输出工具调用调试摘要"),
    output_format: AgentRunFormat = typer.Option(AgentRunFormat.text, "--format", help="输出格式"),
    output: Path | None = typer.Option(None, "--output", help="写入输出文件"),
    config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径"),
) -> None:
    """运行 Skill RAG + 模型工具编码循环。"""
    settings = load_app_settings(config)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key and backend == CodingBackend.builtin:
        raise typer.BadParameter("DASHSCOPE_API_KEY is not set.")

    skill_tool = SkillTool(settings=settings, api_key=api_key)
    retrieved_knowledge: list[RetrievedKnowledge] = []
    manifests = discover_skill_manifests(skills_dir)
    manifest_text = format_skill_manifest_text(manifests)
    for skill_name in skill or []:
        result = skill_tool.retrieve(skill=skill_name, query=goal)
        for chunk in result["chunks"]:
            retrieved_knowledge.append(
                RetrievedKnowledge(
                    skill_name=str(chunk["skill_name"]),
                    file_path=str(chunk.get("file_path", "")),
                    heading_path=list(chunk["heading_path"]),
                    raw_markdown=str(chunk["raw_markdown"]),
                    score=float(chunk["score"]),
                )
            )

    if backend == CodingBackend.codex:
        package = build_context_package(
            goal=goal,
            selected_skills=list(skill or []),
            retrieved=retrieved_knowledge,
            backend=backend.value,
        )
        run_result = PreparedCodexBackend().run(
            CodingRequest(
                goal=goal,
                workspace=str(workspace),
                knowledge_context=package,
                conversation_summary="",
                constraints=[],
                verification_commands=list(verify or []),
            )
        )
        rendered = render_run_result_json(run_result) if output_format == AgentRunFormat.json else run_result.summary
        _emit_output(rendered, output=output)
        return

    registry = create_default_tool_registry(workspace=workspace)
    registry.register("Skill", lambda **kwargs: skill_tool.retrieve(**kwargs))
    package = build_context_package(
        goal=goal,
        selected_skills=list(skill or []),
        retrieved=retrieved_knowledge,
        backend=backend.value,
    )
    builtin_backend = BuiltinCodingBackend(
        model=OpenAICompatibleCodingModel(
            api_key=api_key,
            base_url=settings.models.dashscope_base_url,
            model=settings.models.chat_model,
        ),
        registry=registry,
        max_turns=max_turns,
        max_repair_attempts=max_repair_attempts,
    )
    run_result = builtin_backend.run(
        CodingRequest(
            goal=goal,
            workspace=str(workspace),
            knowledge_context=package,
            conversation_summary="",
            constraints=[manifest_text] if manifest_text else [],
            verification_commands=list(verify or []),
        )
    )
    if show_tools:
        for index, command in enumerate(run_result.commands_run, start=1):
            typer.echo(f"command[{index}] {command.command} {command.status}")
        for index, error in enumerate(run_result.errors, start=1):
            typer.echo(console_safe_text(f"error[{index}] {error}"))
    rendered = render_run_result_json(run_result) if output_format == AgentRunFormat.json else run_result.summary
    _emit_output(rendered, output=output)


def console_safe_text(text: str, encoding: str | None = None) -> str:
    """把模型输出转换成当前控制台可打印的文本，避免 Windows GBK 编码崩溃。"""
    target_encoding = encoding or sys.stdout.encoding or "utf-8"
    return text.encode(target_encoding, errors="replace").decode(target_encoding, errors="replace")


def _emit_output(text: str, *, output: Path | None) -> None:
    if output is not None:
        output.write_text(text, encoding="utf-8", newline="\n")
    else:
        typer.echo(console_safe_text(text))


def _review_serialized_chunks(chunks: list[dict[str, object]]) -> list[dict[str, object]]:
    for index, chunk in enumerate(chunks, start=1):
        typer.echo(f"{index}. score={float(chunk['score']):.6f} path={' > '.join(chunk['heading_path'])}")
    selected_text = typer.prompt(
        "输入要保留的 chunk 序号，逗号分隔；直接回车表示全部保留",
        default="",
        show_default=False,
    )
    selected_indexes = parse_selected_indexes(selected_text, max_index=len(chunks))
    if not selected_indexes:
        return chunks
    return [chunks[index - 1] for index in selected_indexes]


def _psycopg_url(database_url: str) -> str:
    """psycopg 不识别 SQLAlchemy 的 postgresql+psycopg 前缀，这里做一次转换。"""
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def _query_embeddings_if_available(query: str, settings) -> list[float] | None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    embedding_client = EmbeddingClient(
        api_key=api_key,
        base_url=settings.models.dashscope_base_url,
        model=settings.models.embedding_model,
        embedding_dim=settings.models.embedding_dim,
    )
    return embedding_client.embed_texts([query])[0]


def main() -> None:
    app()


if __name__ == "__main__":
    main()

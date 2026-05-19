from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
import typer
from alembic import command
from alembic.config import Config

from app.coding.agent import CodingAgent
from app.models.chat_model import OpenAICompatibleCodingModel
from app.models.embeddings import EmbeddingClient
from app.models.metadata_extractor import MetadataExtractor
from app.rag.ingest import (
    apply_embeddings,
    apply_metadata_extraction,
    apply_record_cache,
    build_ingest_records,
    ingest_records,
    load_record_cache,
)
from app.rag.human_review import parse_selected_indexes
from app.rag.retriever import SkillRetrieveTrace, record_human_review, retrieve_chunks
from app.rag.token_counter import Qwen3TokenCounter
from app.routing.skill_router import discover_skill_manifests, format_skill_manifest_text
from app.settings import load_app_settings
from app.tools.registry import create_default_tool_registry
from app.tools.skill_tool import SkillTool

app = typer.Typer(help="CodeAgent first-stage RAG and coding assistant CLI.")


@app.command("init-db")
def init_db(config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径")) -> None:
    """初始化 PostgreSQL 扩展和基础表结构。"""
    settings = load_app_settings(config)
    database_url = _psycopg_url(settings.database.url)

    with psycopg.connect(database_url, autocommit=True) as conn:
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
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key and (with_metadata or with_embeddings):
        apply_record_cache(records, load_record_cache(settings.database.url))
    if api_key and with_metadata:
        metadata_extractor = MetadataExtractor(
            api_key=api_key,
            base_url=settings.models.dashscope_base_url,
            model=settings.models.chat_model,
        )
        apply_metadata_extraction(records, metadata_extractor)
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
    ingest_records(settings.database.url, records, prune=prune)
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


@app.command("agent-run")
def agent_run(
    goal: str = typer.Option(..., "--goal", help="用户编码需求"),
    skill: list[str] | None = typer.Option(None, "--skill", help="要使用的 skill，可重复传入"),
    skills_dir: Path = typer.Option(
        Path("aurora_gamekit_rag_md_docs"), "--skills-dir", help="skill 文档目录"
    ),
    workspace: Path = typer.Option(Path("."), "--workspace", help="要修改的项目 workspace"),
    max_turns: int = typer.Option(8, "--max-turns", help="模型工具循环最大轮数"),
    show_tools: bool = typer.Option(False, "--show-tools", help="输出工具调用调试摘要"),
    config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径"),
) -> None:
    """运行 Skill RAG + 模型工具编码循环。"""
    settings = load_app_settings(config)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise typer.BadParameter("DASHSCOPE_API_KEY is not set.")

    skill_tool = SkillTool(settings=settings, api_key=api_key)
    context_parts: list[str] = []
    manifests = discover_skill_manifests(skills_dir)
    context_parts.append(format_skill_manifest_text(manifests))
    for skill_name in skill or []:
        result = skill_tool.retrieve(skill=skill_name, query=goal)
        context_parts.append(result["context_markdown"])

    registry = create_default_tool_registry(workspace=workspace)
    registry.register("Skill", lambda **kwargs: skill_tool.retrieve(**kwargs))
    agent = CodingAgent(
        model=OpenAICompatibleCodingModel(
            api_key=api_key,
            base_url=settings.models.dashscope_base_url,
            model=settings.models.chat_model,
        ),
        registry=registry,
        max_turns=max_turns,
    )
    result = agent.run(user_goal=goal, context_markdown="\n\n".join(context_parts))
    if show_tools:
        for index, tool_result in enumerate(result.tool_results, start=1):
            status = "ok" if tool_result["ok"] else "error"
            typer.echo(f"tool[{index}] {tool_result['tool']} {status}")
            if not tool_result["ok"]:
                typer.echo(console_safe_text(f"  {tool_result['error']}"))
    typer.echo(console_safe_text(result.final_response))


def console_safe_text(text: str, encoding: str | None = None) -> str:
    """把模型输出转换成当前控制台可打印的文本，避免 Windows GBK 编码崩溃。"""
    target_encoding = encoding or sys.stdout.encoding or "utf-8"
    return text.encode(target_encoding, errors="replace").decode(target_encoding, errors="replace")


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

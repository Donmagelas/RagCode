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
from app.rag.retriever import retrieve_chunks
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
    records = build_ingest_records(skills_dir)
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
    chunks = retrieve_chunks(
        settings.database.url,
        skill_name=skill,
        query=query,
        top_k=settings.rag.max_final_chunks,
        rrf_k=settings.rag.rrf_k,
        query_expansion_max_terms=settings.rag.query_expansion_max_terms,
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
    config: Path = typer.Option(Path("config/default.yaml"), help="配置文件路径"),
) -> None:
    """输出 RAG 调试信息。"""
    settings = load_app_settings(config)
    embeddings = _query_embeddings_if_available(query, settings)
    chunks = retrieve_chunks(
        settings.database.url,
        skill_name=skill,
        query=query,
        top_k=settings.rag.max_final_chunks,
        rrf_k=settings.rag.rrf_k,
        query_expansion_max_terms=settings.rag.query_expansion_max_terms,
        query_embedding=embeddings,
        metadata_query_embedding=embeddings,
    )
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
        typer.echo(f"approved_chunks={len(chunks)}")
        for index, chunk in enumerate(chunks, start=1):
            typer.echo(f"approved {index}. {' > '.join(chunk.heading_path)}")


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

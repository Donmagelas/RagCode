from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class DatabaseSettings(BaseModel):
    url: str = "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/codeagent"


class ModelSettings(BaseModel):
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen3.6-plus"
    embedding_model: str = "qwen3-vl-embedding"
    embedding_dim: int = 1024
    tokenizer_model: str = "Qwen/Qwen3-Embedding-0.6B"


class RagSettings(BaseModel):
    max_chunk_tokens: int = 1200
    chunk_overlap_tokens: int = 120
    min_chunk_tokens: int = 80
    rrf_k: int = 60
    retriever_top_k: int = 30
    seed_top_n: int = 8
    seed_threshold_ratio: float = 0.75
    expand_threshold_ratio: float = 0.55
    query_expansion_max_terms: int = 4
    max_depth: int = 3
    max_final_chunks: int = 12
    max_context_tokens: int = 12000


class AppSettings(BaseModel):
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    rag: RagSettings = Field(default_factory=RagSettings)


def load_app_settings(config_path: str | Path = "config/default.yaml") -> AppSettings:
    """读取非敏感默认配置；敏感信息后续由 .env 覆盖。"""
    load_dotenv()
    path = Path(config_path)
    if not path.exists():
        return _apply_env_overrides(AppSettings())

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return _apply_env_overrides(AppSettings.model_validate(raw))


def _apply_env_overrides(settings: AppSettings) -> AppSettings:
    data = settings.model_dump()

    if database_url := os.getenv("DATABASE_URL"):
        data["database"]["url"] = database_url
    if base_url := os.getenv("DASHSCOPE_BASE_URL"):
        data["models"]["dashscope_base_url"] = base_url
    if chat_model := os.getenv("CHAT_MODEL"):
        data["models"]["chat_model"] = chat_model
    if embedding_model := os.getenv("EMBEDDING_MODEL"):
        data["models"]["embedding_model"] = embedding_model
    if embedding_dim := os.getenv("EMBEDDING_DIM"):
        data["models"]["embedding_dim"] = int(embedding_dim)
    if tokenizer_model := os.getenv("TOKENIZER_MODEL"):
        data["models"]["tokenizer_model"] = tokenizer_model

    return AppSettings.model_validate(data)

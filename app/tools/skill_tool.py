from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.models.embeddings import EmbeddingClient
from app.rag.retriever import retrieve_chunks
from app.rag.token_counter import Qwen3TokenCounter


class SkillTool:
    def __init__(
        self,
        *,
        settings: Any,
        api_key: str | None = None,
        retriever: Callable[..., list[Any]] = retrieve_chunks,
    ) -> None:
        self._settings = settings
        self._api_key = api_key
        self._retriever = retriever

    def retrieve(self, *, skill: str, query: str, max_chunks: int | None = None) -> dict[str, Any]:
        """在指定 skill 文档内部执行 RAG，并返回可直接放入上下文的 Markdown 原文。"""
        query_embedding = self._query_embedding(query)
        top_k = max_chunks or self._settings.rag.max_final_chunks
        chunks = self._retriever(
            database_url=self._settings.database.url,
            skill_name=skill,
            query=query,
            top_k=top_k,
            rrf_k=self._settings.rag.rrf_k,
            retriever_top_k=self._settings.rag.retriever_top_k,
            seed_top_n=self._settings.rag.seed_top_n,
            seed_threshold_ratio=self._settings.rag.seed_threshold_ratio,
            expand_threshold_ratio=self._settings.rag.expand_threshold_ratio,
            query_expansion_max_terms=self._settings.rag.query_expansion_max_terms,
            max_depth=self._settings.rag.max_depth,
            max_context_tokens=self._settings.rag.max_context_tokens,
            token_counter=Qwen3TokenCounter(self._settings.models.tokenizer_model),
            query_embedding=query_embedding,
            metadata_query_embedding=query_embedding,
        )
        serialized_chunks = [_serialize_chunk(chunk) for chunk in chunks]
        return {
            "skill": skill,
            "query": query,
            "chunks": serialized_chunks,
            "context_markdown": "\n\n".join(chunk["raw_markdown"] for chunk in serialized_chunks),
        }

    def _query_embedding(self, query: str) -> list[float] | None:
        if not self._api_key:
            return None
        embedding_client = EmbeddingClient(
            api_key=self._api_key,
            base_url=self._settings.models.dashscope_base_url,
            model=self._settings.models.embedding_model,
            embedding_dim=self._settings.models.embedding_dim,
        )
        return embedding_client.embed_texts([query])[0]


def _serialize_chunk(chunk: Any) -> dict[str, Any]:
    """把 retriever 返回对象转成 LangGraph 状态里稳定可序列化的 dict。"""
    return {
        "skill_name": chunk.skill_name,
        "heading_path": list(chunk.heading_path),
        "raw_markdown": chunk.raw_markdown,
        "score": float(chunk.score),
    }

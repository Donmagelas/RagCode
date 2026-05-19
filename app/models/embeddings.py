from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlsplit

import httpx
from openai import OpenAI

MULTIMODAL_EMBEDDING_PATH = "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"


class EmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        embedding_dim: int,
        batch_size: int = 8,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._embedding_dim = embedding_dim
        self._batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """调用 embedding 接口，并校验返回维度。"""
        if not texts:
            return []
        if self._model == "qwen3-vl-embedding":
            embeddings = self._embed_texts_with_dashscope_multimodal(texts)
        else:
            embeddings = self._embed_texts_with_openai_compatible(texts)
        self._validate_dimensions(embeddings)
        return embeddings

    def _embed_texts_with_openai_compatible(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        return [item.embedding for item in response.data]

    def _embed_texts_with_dashscope_multimodal(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in chunk_texts(texts, batch_size=self._batch_size):
            response = httpx.post(
                dashscope_multimodal_embedding_url(self._base_url),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=build_multimodal_embedding_payload(
                    model=self._model,
                    texts=batch,
                    dimension=self._embedding_dim,
                ),
                timeout=120,
            )
            if response.is_error:
                detail = response.text[:500]
                raise RuntimeError(f"DashScope embedding request failed: {response.status_code} {detail}")
            data = response.json()
            embeddings.extend(item["embedding"] for item in data["output"]["embeddings"])
        return embeddings

    def _validate_dimensions(self, embeddings: list[list[float]]) -> None:
        for embedding in embeddings:
            if len(embedding) != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self._embedding_dim}, got {len(embedding)}"
                )


def build_multimodal_embedding_payload(
    *, model: str, texts: Sequence[str], dimension: int
) -> dict[str, object]:
    """构建 DashScope qwen3-vl-embedding 原生请求体。"""
    return {
        "model": model,
        "input": {"contents": [{"text": text} for text in texts]},
        "parameters": {"dimension": dimension},
    }


def chunk_texts(texts: Sequence[str], *, batch_size: int) -> list[list[str]]:
    """按批次切分文本，避免单次 embedding 请求过大。"""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    return [list(texts[index : index + batch_size]) for index in range(0, len(texts), batch_size)]


def dashscope_multimodal_embedding_url(base_url: str) -> str:
    """从 OpenAI-compatible base_url 派生 DashScope 原生 embedding endpoint。"""
    parsed = urlsplit(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return origin + MULTIMODAL_EMBEDDING_PATH


def format_pgvector(values: Sequence[float]) -> str:
    """把 float 序列格式化成 pgvector 可接收的文本。"""
    return "[" + ",".join(str(float(value)) for value in values) + "]"

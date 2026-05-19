from __future__ import annotations

from typing import Protocol


class TokenCounter(Protocol):
    """RAG 切分只依赖 encode/decode，方便测试替换 tokenizer。"""

    def encode(self, text: str) -> list[int]:
        """把文本转为 token id。"""

    def decode(self, token_ids: list[int]) -> str:
        """把 token id 还原为文本。"""


class HeuristicTokenCounter:
    """兜底近似计数器；生产 ingest 应优先使用 Qwen3TokenCounter。"""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="ignore")


class Qwen3TokenCounter:
    """使用 Qwen3 开源 tokenizer 做近似 token 计数。"""

    def __init__(self, model_name: str, *, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._tokenizer = None

    def encode(self, text: str) -> list[int]:
        return self._load_tokenizer().encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self._load_tokenizer().decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _load_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen3 tokenizer requires transformers. "
                    "Install project dependencies before ingest."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=self.local_files_only,
            )
        return self._tokenizer

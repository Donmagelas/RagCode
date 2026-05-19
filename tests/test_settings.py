from pathlib import Path

from app.settings import load_app_settings


def test_load_app_settings_reads_yaml_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "default.yaml"
    config_file.write_text(
        "\n".join(
            [
                "database:",
                "  url: postgresql+psycopg://postgres:secret@127.0.0.1:5432/codeagent",
                "models:",
                "  dashscope_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1",
                "  chat_model: qwen3.6-plus",
                "  embedding_model: qwen3-vl-embedding",
                "  embedding_dim: 1024",
                "  tokenizer_model: Qwen/Qwen3-Embedding-0.6B",
                "rag:",
                "  max_chunk_tokens: 1200",
                "  chunk_overlap_tokens: 120",
                "  min_chunk_tokens: 80",
                "  rrf_k: 60",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_app_settings(config_file)

    assert settings.database.url.endswith("/codeagent")
    assert settings.models.chat_model == "qwen3.6-plus"
    assert settings.models.embedding_model == "qwen3-vl-embedding"
    assert settings.models.embedding_dim == 1024
    assert settings.models.tokenizer_model == "Qwen/Qwen3-Embedding-0.6B"
    assert settings.rag.max_chunk_tokens == 1200
    assert settings.rag.chunk_overlap_tokens == 120
    assert settings.rag.min_chunk_tokens == 80
    assert settings.rag.rrf_k == 60


def test_load_app_settings_allows_environment_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    config_file = tmp_path / "default.yaml"
    config_file.write_text("database:\n  url: postgresql+psycopg://default\n", encoding="utf-8")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://env")
    monkeypatch.setenv("CHAT_MODEL", "env-chat")
    monkeypatch.setenv("EMBEDDING_MODEL", "env-embedding")
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    monkeypatch.setenv("TOKENIZER_MODEL", "env-tokenizer")

    settings = load_app_settings(config_file)

    assert settings.database.url == "postgresql+psycopg://env"
    assert settings.models.chat_model == "env-chat"
    assert settings.models.embedding_model == "env-embedding"
    assert settings.models.embedding_dim == 1024
    assert settings.models.tokenizer_model == "env-tokenizer"

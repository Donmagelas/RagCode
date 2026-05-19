from pathlib import Path

from app.rag.ingest import (
    _delete_stale_chunks,
    apply_embeddings,
    apply_metadata_extraction,
    apply_record_cache,
    build_ingest_records,
)


def test_build_ingest_records_parses_file_style_skill(tmp_path: Path) -> None:
    skill_file = tmp_path / "05_UI系统.md"
    skill_file.write_text("# UI 系统\n\nUI 根说明。\n\n## Widget\n\nWidget 说明。", encoding="utf-8")

    records = build_ingest_records(tmp_path)

    assert len(records.docs) == 1
    doc = records.docs[0]
    assert doc["skill_name"] == "05_UI系统"
    assert doc["file_path"].endswith("05_UI系统.md")
    assert doc["content_hash"]
    assert {chunk["heading"] for chunk in records.chunks} == {"UI 系统", "Widget"}
    assert all(chunk["doc_id"] == doc["id"] for chunk in records.chunks)
    assert all("metadata_text" in chunk for chunk in records.chunks)


def test_apply_embeddings_sets_content_and_metadata_vectors(tmp_path: Path) -> None:
    skill_file = tmp_path / "ui.md"
    skill_file.write_text("# UI\n\nUI 内容。", encoding="utf-8")
    records = build_ingest_records(tmp_path)

    class FakeEmbeddingClient:
        def embed_texts(self, texts):
            return [[float(index), float(index + 1)] for index, _text in enumerate(texts)]

    apply_embeddings(records, FakeEmbeddingClient())

    assert records.chunks[0]["content_embedding"] == [0.0, 1.0]
    assert records.chunks[0]["metadata_embedding"] == [0.0, 1.0]


def test_apply_metadata_extraction_enriches_metadata_text(tmp_path: Path) -> None:
    skill_file = tmp_path / "ui.md"
    skill_file.write_text("# UI\n\n`OpenPanelAsync()` 打开面板。", encoding="utf-8")
    records = build_ingest_records(tmp_path)

    class FakeMetadataExtractor:
        def extract(self, raw_markdown, heading_path):
            return {
                "module_type": "UI系统",
                "component_name": "UIPanel",
                "api_name": "OpenPanelAsync",
                "usage_type": "api",
                "tags": ["面板"],
                "searchable_keywords": ["打开面板"],
                "summary": "打开 UI 面板",
            }

    apply_metadata_extraction(records, FakeMetadataExtractor())

    assert records.chunks[0]["metadata_json"]["api_name"] == "OpenPanelAsync"
    assert "打开面板" in records.chunks[0]["metadata_text"]
    assert "OpenPanelAsync" in records.chunks[0]["metadata_text"]


def test_apply_record_cache_reuses_metadata_and_embeddings(tmp_path: Path) -> None:
    skill_file = tmp_path / "ui.md"
    skill_file.write_text("# UI\n\nUI 内容。", encoding="utf-8")
    records = build_ingest_records(tmp_path)
    chunk = records.chunks[0]

    apply_record_cache(
        records,
        {
            chunk["content_hash"]: {
                "metadata_json": {"api_name": "CachedApi"},
                "metadata_text": "CachedApi cached metadata",
                "content_embedding": [1.0, 2.0],
                "metadata_embedding": [3.0, 4.0],
            }
        },
    )

    assert chunk["metadata_json"]["api_name"] == "CachedApi"
    assert chunk["metadata_text"] == "CachedApi cached metadata"
    assert chunk["content_embedding"] == [1.0, 2.0]
    assert chunk["metadata_embedding"] == [3.0, 4.0]


def test_delete_stale_chunks_keeps_only_current_chunk_ids() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.calls = []

        def execute(self, sql, params=None):
            self.calls.append((sql, params))

    conn = FakeConnection()

    _delete_stale_chunks(conn, "doc_1", ["chunk_1", "chunk_2"])

    sql, params = conn.calls[0]
    assert "DELETE FROM doc_chunks" in sql
    assert params == ("doc_1", ["chunk_1", "chunk_2"])

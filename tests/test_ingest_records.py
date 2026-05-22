from pathlib import Path

from app.rag.ingest import (
    _delete_stale_chunks,
    apply_embeddings,
    apply_metadata_extraction,
    apply_record_cache,
    build_ingest_records,
    build_ingest_report,
    render_ingest_report,
)


class CharTokenCounter:
    def encode(self, text: str) -> list[str]:
        return list(text)

    def decode(self, token_ids: list[str]) -> str:
        return "".join(token_ids)


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


def test_build_ingest_records_uses_token_chunking_options(tmp_path: Path) -> None:
    skill_file = tmp_path / "long.md"
    skill_file.write_text("# Long\n\nabcdefghij", encoding="utf-8")

    records = build_ingest_records(
        tmp_path,
        max_chunk_tokens=4,
        chunk_overlap_tokens=1,
        min_chunk_tokens=1,
        token_counter=CharTokenCounter(),
    )

    assert records.chunks[0]["own_content"] == ""
    assert records.chunks[0]["node_type"] == "section"
    assert records.chunks[0]["structural_only"] is True
    assert all(chunk["node_type"] == "part" for chunk in records.chunks[1:])
    assert all(chunk["structural_only"] is False for chunk in records.chunks[1:])
    assert [chunk["own_content"] for chunk in records.chunks[1:]] == ["abcd", "defg", "ghij"]


def test_build_ingest_report_summarizes_chunks_tokens_and_warnings(tmp_path: Path) -> None:
    skill_file = tmp_path / "long.md"
    skill_file.write_text("# Long\n\nabcdefghij\n\n### Jump\n\nbody", encoding="utf-8")

    records = build_ingest_records(
        tmp_path,
        max_chunk_tokens=4,
        chunk_overlap_tokens=1,
        min_chunk_tokens=1,
        token_counter=CharTokenCounter(),
    )

    report = build_ingest_report(records)
    rendered = render_ingest_report(report)

    assert report.doc_count == 1
    assert report.chunk_count == len(records.chunks)
    assert report.part_count >= 1
    assert report.structural_only_count >= 1
    assert report.max_chunk_tokens >= 1
    assert report.warning_count >= 1
    assert "Ingest report" in rendered
    assert "heading_level_jump" in rendered


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


def test_apply_metadata_extraction_runs_when_cached_embeddings_exist(tmp_path: Path) -> None:
    skill_file = tmp_path / "ui.md"
    skill_file.write_text("# UI\n\n`OpenPanelAsync()` 打开面板。", encoding="utf-8")
    records = build_ingest_records(tmp_path)
    chunk = records.chunks[0]
    chunk["content_embedding"] = [1.0, 2.0]
    chunk["metadata_embedding"] = [3.0, 4.0]

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

    assert chunk["metadata_json"]["api_name"] == "OpenPanelAsync"
    assert chunk["content_embedding"] == [1.0, 2.0]
    assert chunk["metadata_embedding"] is None


def test_apply_metadata_extraction_skips_existing_semantic_metadata(tmp_path: Path) -> None:
    skill_file = tmp_path / "ui.md"
    skill_file.write_text("# UI\n\n`OpenPanelAsync()` 打开面板。", encoding="utf-8")
    records = build_ingest_records(tmp_path)
    chunk = records.chunks[0]
    chunk["metadata_json"]["summary"] = "已经提取过"

    class FailingMetadataExtractor:
        def extract(self, raw_markdown, heading_path):
            raise AssertionError("semantic metadata should be reused")

    apply_metadata_extraction(records, FailingMetadataExtractor())

    assert chunk["metadata_json"]["summary"] == "已经提取过"


def test_apply_metadata_extraction_skips_structural_only_chunks(tmp_path: Path) -> None:
    skill_file = tmp_path / "ui.md"
    skill_file.write_text("# UI\n\nabcdefghij", encoding="utf-8")
    records = build_ingest_records(
        tmp_path,
        max_chunk_tokens=4,
        chunk_overlap_tokens=1,
        min_chunk_tokens=1,
        token_counter=CharTokenCounter(),
    )

    called_markdown: list[str] = []

    class FakeMetadataExtractor:
        def extract(self, raw_markdown, heading_path):
            called_markdown.append(raw_markdown)
            return {
                "module_type": "",
                "component_name": "",
                "api_name": "",
                "usage_type": "",
                "tags": [],
                "searchable_keywords": [],
                "summary": "part metadata",
            }

    apply_metadata_extraction(records, FakeMetadataExtractor())

    assert records.chunks[0]["structural_only"] is True
    assert records.chunks[0]["metadata_json"].get("summary") is None
    assert "# UI" not in called_markdown
    assert called_markdown == ["abcd", "defg", "ghij"]


def test_apply_metadata_extraction_uses_batch_extractor(tmp_path: Path) -> None:
    skill_file = tmp_path / "ui.md"
    skill_file.write_text("# UI\n\nA\n\n## Button\n\nB", encoding="utf-8")
    records = build_ingest_records(tmp_path)

    class FakeBatchMetadataExtractor:
        def extract_many(self, items):
            return [
                {
                    "module_type": "UI系统",
                    "component_name": "",
                    "api_name": "",
                    "usage_type": "doc",
                    "tags": [],
                    "searchable_keywords": [],
                    "summary": "batch metadata",
                }
                for _item in items
            ]

    apply_metadata_extraction(records, FakeBatchMetadataExtractor())

    assert all(chunk["metadata_json"]["summary"] == "batch metadata" for chunk in records.chunks)


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

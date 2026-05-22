from app.rag.metadata_enrich import MetadataTarget, semantic_metadata_sql


def test_semantic_metadata_sql_checks_expected_fields() -> None:
    sql = semantic_metadata_sql("c.metadata_json")

    assert "summary" in sql
    assert "api_name" in sql
    assert "module_type" in sql
    assert "searchable_keywords" in sql


def test_metadata_target_carries_original_metadata_for_merge() -> None:
    target = MetadataTarget(
        id="chunk-1",
        metadata_json={"skill_name": "ui"},
        raw_markdown="# UI",
        heading_path=["UI"],
    )

    assert target.metadata_json["skill_name"] == "ui"
    assert target.heading_path == ["UI"]

from app.models.metadata_extractor import normalize_metadata


def test_normalize_metadata_keeps_fixed_schema_and_lists() -> None:
    metadata = normalize_metadata(
        {
            "module_type": "UI系统",
            "component_name": None,
            "api_name": "OpenPanelAsync",
            "usage_type": "api",
            "tags": "UI",
            "searchable_keywords": ["面板", 123],
            "summary": "打开面板",
            "unexpected": "ignored",
        }
    )

    assert metadata == {
        "module_type": "UI系统",
        "component_name": "",
        "api_name": "OpenPanelAsync",
        "usage_type": "api",
        "tags": ["UI"],
        "searchable_keywords": ["面板", "123"],
        "summary": "打开面板",
    }

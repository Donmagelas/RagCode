from app.models.metadata_extractor import (
    MetadataExtractor,
    extract_json_object,
    normalize_metadata,
)


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


def test_extract_json_object_accepts_fenced_json() -> None:
    parsed = extract_json_object(
        '说明文字\n```json\n{"api_name":"OpenPanelAsync","tags":["UI"]}\n```'
    )

    assert parsed["api_name"] == "OpenPanelAsync"
    assert parsed["tags"] == ["UI"]


def test_metadata_extractor_single_extract_reads_batched_items_response() -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            class Message:
                content = (
                    '{"items":[{"index":0,"module_type":"UI系统","component_name":"UIPanel",'
                    '"api_name":"OpenPanelAsync","usage_type":"api","tags":["UI"],'
                    '"searchable_keywords":["打开面板"],"summary":"打开 UI 面板"}]}'
                )

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    extractor = MetadataExtractor(
        api_key="test",
        base_url="https://example.test/v1",
        model="qwen",
        batch_size=1,
    )
    extractor._client.chat.completions = FakeCompletions()

    metadata = extractor.extract("raw", ["UI"])

    assert metadata["api_name"] == "OpenPanelAsync"
    assert metadata["searchable_keywords"] == ["打开面板"]

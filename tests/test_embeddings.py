from app.models.embeddings import build_multimodal_embedding_payload, chunk_texts, format_pgvector


def test_format_pgvector_serializes_float_list() -> None:
    assert format_pgvector([1, 0.5, -2]) == "[1.0,0.5,-2.0]"


def test_build_multimodal_embedding_payload_uses_dimension_and_contents() -> None:
    payload = build_multimodal_embedding_payload(
        model="qwen3-vl-embedding",
        texts=["OnBoot", "UI"],
        dimension=1024,
    )

    assert payload == {
        "model": "qwen3-vl-embedding",
        "input": {"contents": [{"text": "OnBoot"}, {"text": "UI"}]},
        "parameters": {"dimension": 1024},
    }


def test_chunk_texts_splits_by_batch_size() -> None:
    assert list(chunk_texts(["a", "b", "c"], batch_size=2)) == [["a", "b"], ["c"]]

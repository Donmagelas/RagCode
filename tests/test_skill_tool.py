from app.tools.skill_tool import SkillTool


def test_skill_tool_returns_markdown_context_from_retriever() -> None:
    class FakeSettings:
        class Rag:
            max_final_chunks = 3
            rrf_k = 60
            retriever_top_k = 30
            seed_top_n = 8
            seed_threshold_ratio = 0.75
            expand_threshold_ratio = 0.55
            query_expansion_max_terms = 2
            max_depth = 3
            max_context_tokens = 12000

        class Models:
            dashscope_base_url = "https://example.test"
            embedding_model = "fake"
            embedding_dim = 1024
            tokenizer_model = "fake-tokenizer"

        class Database:
            url = "postgresql://example"
            connect_timeout_seconds = 5

        rag = Rag()
        models = Models()
        database = Database()

    class FakeChunk:
        skill_name = "ui"
        heading_path = ["UI", "Window"]
        raw_markdown = "# Window\n\nUse WindowBase."
        score = 0.42

    calls = []

    def fake_retriever(**kwargs):
        calls.append(kwargs)
        return [FakeChunk()]

    tool = SkillTool(settings=FakeSettings(), retriever=fake_retriever)

    result = tool.retrieve(skill="ui", query="how to open window", max_chunks=2)

    assert result["skill"] == "ui"
    assert result["chunks"][0]["raw_markdown"] == "# Window\n\nUse WindowBase."
    assert result["context_markdown"] == "# Window\n\nUse WindowBase."
    assert calls[0]["skill_name"] == "ui"
    assert calls[0]["query"] == "how to open window"
    assert calls[0]["top_k"] == 2
    assert calls[0]["retriever_top_k"] == 30
    assert calls[0]["seed_top_n"] == 8
    assert calls[0]["seed_threshold_ratio"] == 0.75
    assert calls[0]["expand_threshold_ratio"] == 0.55
    assert calls[0]["max_depth"] == 3
    assert calls[0]["max_context_tokens"] == 12000
    assert calls[0]["connect_timeout_seconds"] == 5

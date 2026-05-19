from app.tools.skill_tool import SkillTool


def test_skill_tool_returns_markdown_context_from_retriever() -> None:
    class FakeSettings:
        class Rag:
            max_final_chunks = 3
            rrf_k = 60
            query_expansion_max_terms = 2

        class Models:
            dashscope_base_url = "https://example.test"
            embedding_model = "fake"
            embedding_dim = 1024

        class Database:
            url = "postgresql://example"

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

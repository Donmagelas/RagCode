from app.graph.workflow import build_rag_graph


def test_build_rag_graph_compiles() -> None:
    graph = build_rag_graph()

    assert graph is not None


def test_rag_graph_retrieves_selected_skills() -> None:
    class FakeSkillTool:
        def retrieve(self, *, skill, query):
            return {
                "skill": skill,
                "query": query,
                "chunks": [
                    {
                        "skill_name": skill,
                        "heading_path": [skill, "API"],
                        "raw_markdown": f"# {skill}",
                        "score": 1.0,
                    }
                ],
                "context_markdown": f"# {skill}",
            }

    graph = build_rag_graph(skill_tool=FakeSkillTool())
    result = graph.invoke(
        {
            "conversation_id": "c1",
            "task_id": "t1",
            "messages": [],
            "user_goal": " use ui ",
            "selected_skills": ["ui", "animation"],
            "retrieved_chunks": [],
            "approved_chunks": [],
            "files_read": [],
            "files_changed": [],
            "commands_run": [],
            "test_results": [],
            "open_questions": [],
            "conversation_summary": "",
            "final_summary": "",
        }
    )

    assert [chunk["skill_name"] for chunk in result["retrieved_chunks"]] == ["ui", "animation"]
    assert result["final_summary"] == "approved_chunks=2"


def test_rag_graph_executes_pending_tool_calls() -> None:
    class FakeRegistry:
        def call(self, name, **kwargs):
            assert name == "read_file"
            return {"path": kwargs["path"], "content": "hello"}

    graph = build_rag_graph(tool_registry=FakeRegistry())
    result = graph.invoke(
        {
            "conversation_id": "c1",
            "task_id": "t1",
            "messages": [],
            "user_goal": "read file",
            "selected_skills": [],
            "retrieved_chunks": [],
            "approved_chunks": [],
            "pending_tool_calls": [{"tool": "read_file", "args": {"path": "main.py"}}],
            "tool_results": [],
            "files_read": [],
            "files_changed": [],
            "commands_run": [],
            "test_results": [],
            "open_questions": [],
            "conversation_summary": "",
            "final_summary": "",
        }
    )

    assert result["files_read"] == ["main.py"]
    assert result["tool_results"][0]["ok"] is True

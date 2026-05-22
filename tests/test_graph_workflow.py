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


def test_graph_uses_model_skill_selector_and_packs_knowledge_context() -> None:
    class FakeSelector:
        def select_skills(self, *, goal, manifests, conversation_summary):
            assert goal == "use ui"
            return {"selected_skills": ["ui"], "reason": "需要 UI"}

    class FakeSkillTool:
        def retrieve(self, *, skill, query):
            return {
                "chunks": [
                    {
                        "skill_name": skill,
                        "file_path": "ui.md",
                        "heading_path": ["UI"],
                        "raw_markdown": "# UI",
                        "score": 1.0,
                    }
                ]
            }

    class FakeBackend:
        def run(self, request):
            assert request.knowledge_context.selected_skills == ["ui"]
            return {
                "backend": "fake",
                "status": "success",
                "goal": request.goal,
                "summary": "ok",
            }

    graph = build_rag_graph(
        skill_manifest_provider=lambda _state: [{"skill_name": "ui", "description": "UI"}],
        skill_selector=FakeSelector(),
        skill_tool=FakeSkillTool(),
        coding_backend=FakeBackend(),
    )
    result = graph.invoke(
        {
            "conversation_id": "c1",
            "task_id": "t1",
            "messages": [],
            "user_goal": " use ui ",
            "selected_skills": [],
            "retrieved_chunks": [],
            "approved_chunks": [],
            "pending_tool_calls": [],
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

    assert result["selected_skills"] == ["ui"]
    assert result["skill_selection_reason"] == "需要 UI"
    assert result["knowledge_context_package"]["selected_skills"] == ["ui"]
    assert result["final_result"]["backend"] == "fake"


def test_graph_compact_context_calls_summary_store() -> None:
    stored = []

    def store_summary(*, conversation_id, summary, removed_message_count):
        stored.append(
            {
                "conversation_id": conversation_id,
                "summary": summary,
                "removed_message_count": removed_message_count,
            }
        )

    graph = build_rag_graph(summary_store=store_summary)
    result = graph.invoke(
        {
            "conversation_id": "c1",
            "task_id": "t1",
            "messages": [
                {"role": "user", "content": f"old {index}"}
                for index in range(10)
            ],
            "user_goal": "hello",
            "selected_skills": [],
            "retrieved_chunks": [],
            "approved_chunks": [],
            "pending_tool_calls": [],
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

    assert result["conversation_summary"]
    assert stored[0]["conversation_id"] == "c1"
    assert stored[0]["removed_message_count"] == 2


def test_build_rag_graph_accepts_checkpointer() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_rag_graph(checkpointer=MemorySaver())

    result = graph.invoke(
        {
            "conversation_id": "c1",
            "task_id": "t1",
            "messages": [],
            "user_goal": "hello",
            "selected_skills": [],
            "retrieved_chunks": [],
            "approved_chunks": [],
            "pending_tool_calls": [],
            "tool_results": [],
            "files_read": [],
            "files_changed": [],
            "commands_run": [],
            "test_results": [],
            "open_questions": [],
            "conversation_summary": "",
            "final_summary": "",
        },
        config={"configurable": {"thread_id": "t1"}},
    )

    assert result["final_summary"]

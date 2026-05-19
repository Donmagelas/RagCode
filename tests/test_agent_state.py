from app.memory.state import create_initial_agent_state


def test_create_initial_agent_state_contains_first_stage_memory_fields() -> None:
    state = create_initial_agent_state(
        conversation_id="conversation-1",
        task_id="task-1",
        user_goal="实现 UI 面板",
    )

    assert state["conversation_id"] == "conversation-1"
    assert state["task_id"] == "task-1"
    assert state["user_goal"] == "实现 UI 面板"
    assert state["selected_skills"] == []
    assert state["retrieved_chunks"] == []
    assert state["approved_chunks"] == []
    assert state["pending_tool_calls"] == []
    assert state["tool_results"] == []
    assert state["conversation_summary"] == ""

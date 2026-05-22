from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class AgentState(TypedDict):
    conversation_id: str
    task_id: str
    messages: list[dict[str, Any]]
    user_goal: str
    selected_skills: list[str]
    skill_manifests: NotRequired[list[dict[str, Any]]]
    skill_selection_reason: NotRequired[str]
    retrieved_chunks: list[dict[str, Any]]
    approved_chunks: list[dict[str, Any]]
    knowledge_context_package: NotRequired[dict[str, Any]]
    coding_request: NotRequired[dict[str, Any]]
    coding_result: NotRequired[dict[str, Any]]
    final_result: NotRequired[dict[str, Any]]
    backend: NotRequired[str]
    workspace_path: NotRequired[str]
    verification_commands: NotRequired[list[str]]
    pending_tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    files_read: list[str]
    files_changed: list[str]
    commands_run: list[dict[str, Any]]
    test_results: list[dict[str, Any]]
    open_questions: list[str]
    conversation_summary: str
    final_summary: str


def create_initial_agent_state(
    *, conversation_id: str, task_id: str, user_goal: str
) -> AgentState:
    """创建 LangGraph 初始状态；所有列表字段避免共享可变默认值。"""
    return {
        "conversation_id": conversation_id,
        "task_id": task_id,
        "messages": [],
        "user_goal": user_goal,
        "selected_skills": [],
        "skill_manifests": [],
        "skill_selection_reason": "",
        "retrieved_chunks": [],
        "approved_chunks": [],
        "knowledge_context_package": {},
        "coding_request": {},
        "coding_result": {},
        "final_result": {},
        "backend": "codex",
        "workspace_path": ".",
        "verification_commands": [],
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

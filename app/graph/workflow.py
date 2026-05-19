from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.coding.loop import execute_tool_calls
from app.memory.compact import compact_messages
from app.memory.state import AgentState


def build_rag_graph(*, skill_tool: Any | None = None, tool_registry: Any | None = None):
    """构建第一阶段 RAG 调试图骨架。"""
    graph = StateGraph(AgentState)
    graph.add_node("normalize_query", normalize_query)
    graph.add_node("skill_disclosure", skill_disclosure)
    graph.add_node("rag_retrieve", _rag_retrieve_node(skill_tool))
    graph.add_node("pack_context", pack_context)
    graph.add_node("coding_loop", _coding_loop_node(tool_registry))
    graph.add_node("compact_context", compact_context)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("normalize_query")
    graph.add_edge("normalize_query", "skill_disclosure")
    graph.add_edge("skill_disclosure", "rag_retrieve")
    graph.add_edge("rag_retrieve", "pack_context")
    graph.add_edge("pack_context", "coding_loop")
    graph.add_edge("coding_loop", "compact_context")
    graph.add_edge("compact_context", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def normalize_query(state: AgentState) -> dict[str, object]:
    """第一阶段先保留原始目标，后续再扩展 query normalize。"""
    return {"user_goal": state["user_goal"].strip()}


def skill_disclosure(state: AgentState) -> dict[str, object]:
    """Skill 路由后续接模型；当前保留已有 selected_skills。"""
    return {"selected_skills": state["selected_skills"]}


def _rag_retrieve_node(skill_tool: Any | None):
    def node(state: AgentState) -> dict[str, object]:
        return rag_retrieve(state, skill_tool=skill_tool)

    return node


def rag_retrieve(state: AgentState, *, skill_tool: Any | None = None) -> dict[str, object]:
    """按已选 skill 调用 SkillTool；未注入工具时保留现有检索结果。"""
    if skill_tool is None or not state["selected_skills"]:
        return {"retrieved_chunks": state["retrieved_chunks"]}

    retrieved_chunks: list[dict[str, Any]] = []
    for skill in state["selected_skills"]:
        result = skill_tool.retrieve(skill=skill, query=state["user_goal"])
        retrieved_chunks.extend(result["chunks"])
    return {"retrieved_chunks": retrieved_chunks}


def pack_context(state: AgentState) -> dict[str, object]:
    """把检索结果默认视为已确认上下文。"""
    approved = state["approved_chunks"] or state["retrieved_chunks"]
    return {"approved_chunks": approved}


def _coding_loop_node(tool_registry: Any | None):
    def node(state: AgentState) -> dict[str, object]:
        return coding_loop(state, tool_registry=tool_registry)

    return node


def coding_loop(state: AgentState, *, tool_registry: Any | None = None) -> dict[str, object]:
    """执行已经计划好的工具调用；真正的模型多轮决策后续接入。"""
    pending_tool_calls = state.get("pending_tool_calls", [])
    if tool_registry is None or not pending_tool_calls:
        return {}

    result = execute_tool_calls(pending_tool_calls, registry=tool_registry)
    return {
        "tool_results": [*state.get("tool_results", []), *result["tool_results"]],
        "files_read": [*state["files_read"], *result["files_read"]],
        "files_changed": [*state["files_changed"], *result["files_changed"]],
        "commands_run": [*state["commands_run"], *result["commands_run"]],
        "pending_tool_calls": [],
    }


def compact_context(state: AgentState) -> dict[str, object]:
    """基础上下文压缩，避免 messages 无限增长。"""
    result = compact_messages(state["messages"], preserve_recent_messages=8)
    if result.removed_message_count == 0:
        return {}
    return {
        "messages": result.messages,
        "conversation_summary": result.summary,
    }


def finalize(state: AgentState) -> dict[str, object]:
    """生成第一阶段图执行摘要。"""
    return {"final_summary": f"approved_chunks={len(state['approved_chunks'])}"}

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from app.coding.backend import CodingRequest
from app.context.package import RetrievedKnowledge, build_context_package
from app.coding.loop import execute_tool_calls
from app.coding.result import CodingRunResult
from app.memory.compact import compact_messages
from app.memory.state import AgentState
from app.routing.model_skill_selector import SkillSelectionResult, manifest_from_mapping


def build_rag_graph(
    *,
    skill_tool: Any | None = None,
    skill_selector: Any | None = None,
    skill_manifest_provider: Any | None = None,
    coding_backend: Any | None = None,
    tool_registry: Any | None = None,
    summary_store: Any | None = None,
    checkpointer: Any | None = None,
):
    """构建知识准备到抽象编码后端的 LangGraph 工作流。"""
    graph = StateGraph(AgentState)
    graph.add_node("normalize_query", normalize_query)
    graph.add_node("load_skill_manifests", _load_skill_manifests_node(skill_manifest_provider))
    graph.add_node("model_select_skills", _model_select_skills_node(skill_selector))
    graph.add_node("rag_retrieve", _rag_retrieve_node(skill_tool))
    graph.add_node("pack_knowledge_context", pack_knowledge_context)
    graph.add_node("compact_context", _compact_context_node(summary_store))
    graph.add_node("coding_backend", _coding_backend_node(coding_backend, tool_registry))
    graph.add_node("finalize", finalize)

    graph.set_entry_point("normalize_query")
    graph.add_edge("normalize_query", "load_skill_manifests")
    graph.add_edge("load_skill_manifests", "model_select_skills")
    graph.add_edge("model_select_skills", "rag_retrieve")
    graph.add_edge("rag_retrieve", "pack_knowledge_context")
    graph.add_edge("pack_knowledge_context", "compact_context")
    graph.add_edge("compact_context", "coding_backend")
    graph.add_edge("coding_backend", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def normalize_query(state: AgentState) -> dict[str, object]:
    """第一阶段先保留原始目标，后续再扩展 query normalize。"""
    return {"user_goal": state["user_goal"].strip()}


def _load_skill_manifests_node(skill_manifest_provider: Any | None):
    def node(state: AgentState) -> dict[str, object]:
        return load_skill_manifests(state, skill_manifest_provider=skill_manifest_provider)

    return node


def load_skill_manifests(
    state: AgentState, *, skill_manifest_provider: Any | None = None
) -> dict[str, object]:
    """加载轻量 skill manifest；没有 provider 时沿用 state 中已有内容。"""
    if skill_manifest_provider is None:
        return {"skill_manifests": state.get("skill_manifests", [])}
    if callable(skill_manifest_provider):
        manifests = skill_manifest_provider(state)
    elif hasattr(skill_manifest_provider, "load"):
        manifests = skill_manifest_provider.load(state)
    else:
        manifests = []
    return {"skill_manifests": manifests}


def _model_select_skills_node(skill_selector: Any | None):
    def node(state: AgentState) -> dict[str, object]:
        return model_select_skills(state, skill_selector=skill_selector)

    return node


def model_select_skills(state: AgentState, *, skill_selector: Any | None = None) -> dict[str, object]:
    """使用模型根据 skill manifest 选择需要检索的框架知识文档。"""
    if skill_selector is None:
        return {
            "selected_skills": state.get("selected_skills", []),
            "skill_selection_reason": state.get("skill_selection_reason", ""),
        }
    manifests = [
        manifest_from_mapping(item)
        for item in state.get("skill_manifests", [])
        if isinstance(item, dict)
    ]
    result = skill_selector.select_skills(
        goal=state["user_goal"],
        manifests=manifests,
        conversation_summary=state.get("conversation_summary", ""),
    )
    if isinstance(result, SkillSelectionResult):
        return {"selected_skills": result.selected_skills, "skill_selection_reason": result.reason}
    return {
        "selected_skills": list(result.get("selected_skills", [])),
        "skill_selection_reason": str(result.get("reason", "")),
    }


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
    """兼容旧调用名，把检索结果默认视为已确认上下文。"""
    return pack_knowledge_context(state)


def pack_knowledge_context(state: AgentState) -> dict[str, object]:
    """把已确认 chunk 打包成编码后端无关的知识上下文包。"""
    approved = state["approved_chunks"] or state["retrieved_chunks"]
    retrieved = [
        RetrievedKnowledge(
            skill_name=str(chunk.get("skill_name", "")),
            file_path=str(chunk.get("file_path", "")),
            heading_path=list(chunk.get("heading_path", [])),
            raw_markdown=str(chunk.get("raw_markdown", "")),
            score=float(chunk.get("score", 0.0)),
        )
        for chunk in approved
    ]
    package = build_context_package(
        goal=state["user_goal"],
        selected_skills=state.get("selected_skills", []),
        retrieved=retrieved,
        backend=state.get("backend", "codex"),
    )
    return {"approved_chunks": approved, "knowledge_context_package": asdict(package)}


def _coding_backend_node(coding_backend: Any | None, tool_registry: Any | None):
    def node(state: AgentState) -> dict[str, object]:
        return coding_backend_node(state, coding_backend=coding_backend, tool_registry=tool_registry)

    return node


def coding_backend_node(
    state: AgentState,
    *,
    coding_backend: Any | None = None,
    tool_registry: Any | None = None,
) -> dict[str, object]:
    """调用抽象编码后端；未注入后端时保留旧工具调用 smoke 能力。"""
    if coding_backend is None:
        return coding_loop(state, tool_registry=tool_registry)

    package = build_context_package(
        goal=state["knowledge_context_package"]["goal"],
        selected_skills=list(state["knowledge_context_package"]["selected_skills"]),
        retrieved=[
            RetrievedKnowledge(
                skill_name=str(item["skill_name"]),
                file_path=str(item.get("file_path", "")),
                heading_path=list(item["heading_path"]),
                raw_markdown=str(item["raw_markdown"]),
                score=float(item["score"]),
            )
            for item in state["knowledge_context_package"].get("retrieved", [])
        ],
        backend=str(state["knowledge_context_package"].get("backend", "codex")),
        trace=state["knowledge_context_package"].get("trace"),
    )
    request = CodingRequest(
        goal=state["user_goal"],
        workspace=state.get("workspace_path", "."),
        knowledge_context=package,
        conversation_summary=state.get("conversation_summary", ""),
        constraints=[],
        verification_commands=state.get("verification_commands", []),
    )
    result = coding_backend.run(request)
    result_dict = _as_plain_dict(result)
    return {
        "coding_request": {
            "goal": request.goal,
            "workspace": request.workspace,
            "selected_skills": request.knowledge_context.selected_skills,
        },
        "coding_result": result_dict,
    }


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


def _compact_context_node(summary_store: Any | None):
    def node(state: AgentState) -> dict[str, object]:
        return compact_context(state, summary_store=summary_store)

    return node


def compact_context(state: AgentState, *, summary_store: Any | None = None) -> dict[str, object]:
    """基础上下文压缩，避免 messages 无限增长。"""
    result = compact_messages(state["messages"], preserve_recent_messages=8)
    if result.removed_message_count == 0:
        return {}
    if summary_store is not None:
        summary_store(
            conversation_id=state["conversation_id"],
            summary=result.summary,
            removed_message_count=result.removed_message_count,
        )
    return {
        "messages": result.messages,
        "conversation_summary": result.summary,
    }


def finalize(state: AgentState) -> dict[str, object]:
    """生成统一最终结果摘要。"""
    final_result = state.get("coding_result", {})
    if final_result:
        return {
            "final_result": final_result,
            "final_summary": str(final_result.get("summary", "")),
        }
    return {"final_summary": f"approved_chunks={len(state['approved_chunks'])}"}


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, CodingRunResult):
        return asdict(value)
    return {"backend": "unknown", "status": "partial", "summary": str(value)}

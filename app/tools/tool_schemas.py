from __future__ import annotations

from typing import Any


def agent_tool_schemas() -> list[dict[str, Any]]:
    """完整 agent 工具 schema：先暴露 Skill，再暴露编码工具。"""
    return [_skill_tool_schema(), *coding_tool_schemas()]


def coding_tool_schemas() -> list[dict[str, Any]]:
    """返回 OpenAI-compatible tool schema，供模型选择编码工具。"""
    return [
        _function_tool(
            "glob_search",
            "按 glob 模式查找项目文件，返回相对路径列表。",
            {
                "pattern": {"type": "string", "description": "例如 **/*.py"},
                "max_results": {"type": "integer", "description": "最多返回数量"},
            },
            ["pattern"],
        ),
        _function_tool(
            "grep_search",
            "在项目文件中搜索正则表达式，返回命中行。",
            {
                "pattern": {"type": "string", "description": "正则表达式"},
                "include": {"type": "string", "description": "文件名过滤，例如 *.py"},
                "max_results": {"type": "integer", "description": "最多返回数量"},
            },
            ["pattern"],
        ),
        _function_tool(
            "read_file",
            "读取项目内文件内容。",
            {"path": {"type": "string", "description": "workspace 内相对路径"}},
            ["path"],
        ),
        _function_tool(
            "edit_file",
            "用 old_string 到 new_string 的 exactly-once 替换修改文件。",
            {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
            },
            ["path", "old_string", "new_string"],
        ),
        _function_tool(
            "write_file",
            "新建或整体覆盖项目内文件。",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        _function_tool(
            "bash",
            "在 workspace 内运行验证、构建或检查命令。",
            {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer"},
            },
            ["command"],
        ),
        _function_tool(
            "git_diff",
            "查看当前 workspace 的 git diff。",
            {},
            [],
        ),
    ]


def _skill_tool_schema() -> dict[str, Any]:
    return _function_tool(
        "Skill",
        "按需读取某个 skill 文档内部的框架 Markdown 原文片段。",
        {
            "skill": {"type": "string", "description": "skill 名称，例如 ui 或 animation"},
            "query": {"type": "string", "description": "要在该 skill 内部检索的问题"},
            "max_chunks": {"type": "integer", "description": "最多返回 chunk 数"},
        },
        ["skill", "query"],
    )


def _function_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }

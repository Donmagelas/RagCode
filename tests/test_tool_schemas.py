from app.tools.tool_schemas import agent_tool_schemas, coding_tool_schemas


def test_coding_tool_schemas_include_expected_tool_names() -> None:
    schemas = coding_tool_schemas()

    names = [schema["function"]["name"] for schema in schemas]

    assert names == [
        "glob_search",
        "grep_search",
        "read_file",
        "edit_file",
        "write_file",
        "bash",
        "git_diff",
    ]


def test_edit_file_schema_requires_old_and_new_string() -> None:
    edit_schema = next(
        schema for schema in coding_tool_schemas() if schema["function"]["name"] == "edit_file"
    )

    assert edit_schema["function"]["parameters"]["required"] == ["path", "old_string", "new_string"]


def test_agent_tool_schemas_include_skill_tool_first() -> None:
    schemas = agent_tool_schemas()

    assert schemas[0]["function"]["name"] == "Skill"
    assert schemas[0]["function"]["parameters"]["required"] == ["skill", "query"]

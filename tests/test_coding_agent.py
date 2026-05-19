from app.coding.agent import CodingAgent, ModelTurn, _initial_messages
from app.tools.registry import ToolRegistry


def test_coding_agent_feeds_tool_results_back_to_model() -> None:
    registry = ToolRegistry()
    registry.register("read_file", lambda *, path: {"path": path, "content": "print('ok')"})

    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def next_turn(self, *, messages, tools):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return ModelTurn(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "read_file",
                            "args": {"path": "main.py"},
                        }
                    ],
                )
            return ModelTurn(content="已读取 main.py", tool_calls=[])

    model = FakeModel()
    agent = CodingAgent(model=model, registry=registry)

    result = agent.run(user_goal="read main.py", context_markdown="# UI")

    assert result.final_response == "已读取 main.py"
    assert result.tool_results[0]["ok"] is True
    assert model.calls[1][-1]["role"] == "tool"
    assert "print('ok')" in model.calls[1][-1]["content"]


def test_coding_agent_returns_tool_errors_to_model() -> None:
    registry = ToolRegistry()
    registry.register("edit_file", lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad edit")))

    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, *, messages, tools):
            self.calls += 1
            if self.calls == 1:
                return ModelTurn(
                    content="",
                    tool_calls=[{"id": "call_1", "name": "edit_file", "args": {"path": "main.py"}}],
                )
            return ModelTurn(content="我会修正 edit 参数", tool_calls=[])

    agent = CodingAgent(model=FakeModel(), registry=registry)

    result = agent.run(user_goal="edit main.py")

    assert result.tool_results[0]["ok"] is False
    assert "bad edit" in result.tool_results[0]["error"]


def test_initial_messages_tell_model_to_use_skill_for_framework_details() -> None:
    messages = _initial_messages(user_goal="实现 UI", context_markdown="可用 Skill：ui")

    assert "调用 Skill" in messages[0]["content"]
    assert "可用 Skill：ui" in messages[0]["content"]

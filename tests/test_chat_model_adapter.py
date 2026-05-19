from app.models.chat_model import model_turn_from_openai_message


def test_model_turn_from_openai_message_parses_tool_calls() -> None:
    class FakeFunction:
        name = "read_file"
        arguments = '{"path":"main.py"}'

    class FakeToolCall:
        id = "call_1"
        function = FakeFunction()

    class FakeMessage:
        content = "need file"
        tool_calls = [FakeToolCall()]

    turn = model_turn_from_openai_message(FakeMessage())

    assert turn.content == "need file"
    assert turn.tool_calls == [{"id": "call_1", "name": "read_file", "args": {"path": "main.py"}}]


def test_model_turn_from_openai_message_handles_bad_arguments() -> None:
    class FakeFunction:
        name = "read_file"
        arguments = "{bad json"

    class FakeToolCall:
        id = "call_1"
        function = FakeFunction()

    class FakeMessage:
        content = None
        tool_calls = [FakeToolCall()]

    turn = model_turn_from_openai_message(FakeMessage())

    assert turn.content == ""
    assert turn.tool_calls[0]["args"] == {}

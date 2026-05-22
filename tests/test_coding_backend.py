from app.coding.agent import ModelTurn
from app.coding.backend import BuiltinCodingBackend, CodingRequest, PreparedCodexBackend
from app.context.package import RetrievedKnowledge, build_context_package


def test_prepared_codex_backend_returns_unified_result() -> None:
    package = build_context_package(
        goal="实现背包 UI",
        selected_skills=["ui"],
        retrieved=[
            RetrievedKnowledge(
                skill_name="ui",
                file_path="ui.md",
                heading_path=["UI"],
                raw_markdown="# UI",
                score=1.0,
            )
        ],
        backend="codex",
    )
    request = CodingRequest(
        goal="实现背包 UI",
        workspace=".",
        knowledge_context=package,
        conversation_summary="",
        constraints=[],
        verification_commands=[],
    )

    result = PreparedCodexBackend().run(request)

    assert result.backend == "codex"
    assert result.knowledge.chunk_count == 1
    assert result.knowledge.skills == ["ui"]
    assert result.artifacts[0].type == "context_markdown"


def test_builtin_backend_runs_tool_loop_and_verification_commands() -> None:
    package = build_context_package(
        goal="修改 UI 文案",
        selected_skills=["ui"],
        retrieved=[
            RetrievedKnowledge(
                skill_name="ui",
                file_path="ui.md",
                heading_path=["UI"],
                raw_markdown="# UI\n\n使用 SetText 修改文本。",
                score=1.0,
            )
        ],
        backend="builtin",
    )
    request = CodingRequest(
        goal="修改 UI 文案",
        workspace=".",
        knowledge_context=package,
        conversation_summary="之前确认使用 UI skill",
        constraints=["修改前必须 read_file"],
        verification_commands=["pytest tests/test_ui.py"],
    )
    registry = RecordingRegistry()
    model = ScriptedModel(
        [
            ModelTurn(
                content="先读文件再修改",
                tool_calls=[
                    {
                        "id": "call-read",
                        "name": "read_file",
                        "args": {"path": "ui.py"},
                    },
                    {
                        "id": "call-edit",
                        "name": "edit_file",
                        "args": {
                            "path": "ui.py",
                            "old_string": "old",
                            "new_string": "new",
                        },
                    },
                ],
            ),
            ModelTurn(content="完成修改", tool_calls=[]),
        ]
    )

    result = BuiltinCodingBackend(model=model, registry=registry, max_repair_attempts=0).run(request)

    assert result.backend == "builtin"
    assert result.status == "success"
    assert result.files_read == ["ui.py"]
    assert result.files_changed[0].path == "ui.py"
    assert [command.command for command in result.commands_run] == ["pytest tests/test_ui.py"]
    assert result.test_results[0].status == "passed"
    assert registry.calls[-1] == ("bash", {"command": "pytest tests/test_ui.py"})


def test_builtin_backend_marks_failed_verification_as_failed() -> None:
    package = build_context_package(
        goal="修改 UI 文案",
        selected_skills=["ui"],
        retrieved=[],
        backend="builtin",
    )
    request = CodingRequest(
        goal="修改 UI 文案",
        workspace=".",
        knowledge_context=package,
        conversation_summary="",
        constraints=[],
        verification_commands=["pytest"],
    )
    registry = RecordingRegistry(bash_exit_codes=[1])
    model = ScriptedModel([ModelTurn(content="完成修改", tool_calls=[])])

    result = BuiltinCodingBackend(model=model, registry=registry, max_repair_attempts=0).run(request)

    assert result.status == "failed"
    assert result.commands_run[0].command == "pytest"
    assert result.commands_run[0].status == "failed"


def test_builtin_backend_retries_after_failed_verification() -> None:
    package = build_context_package(
        goal="修复 UI 测试",
        selected_skills=["ui"],
        retrieved=[],
        backend="builtin",
    )
    request = CodingRequest(
        goal="修复 UI 测试",
        workspace=".",
        knowledge_context=package,
        conversation_summary="",
        constraints=[],
        verification_commands=["pytest"],
    )
    registry = RecordingRegistry(bash_exit_codes=[1, 0])
    model = ScriptedModel(
        [
            ModelTurn(content="先完成一次修改", tool_calls=[]),
            ModelTurn(
                content="根据失败结果继续修复",
                tool_calls=[
                    {
                        "id": "call-edit",
                        "name": "edit_file",
                        "args": {
                            "path": "ui.py",
                            "old_string": "broken",
                            "new_string": "fixed",
                        },
                    }
                ],
            ),
            ModelTurn(content="修复完成", tool_calls=[]),
        ]
    )

    result = BuiltinCodingBackend(model=model, registry=registry, max_repair_attempts=1).run(request)

    assert result.status == "success"
    assert [command.status for command in result.commands_run] == ["failed", "passed"]
    assert result.files_changed[0].path == "ui.py"


class ScriptedModel:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = list(turns)

    def next_turn(self, *, messages, tools):
        assert tools
        return self._turns.pop(0)


class RecordingRegistry:
    def __init__(self, *, bash_exit_codes: list[int] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._bash_exit_codes = list(bash_exit_codes or [0])

    def call(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name == "read_file":
            return {"path": kwargs["path"], "content": "old"}
        if name == "edit_file":
            return {"path": kwargs["path"], "changed": True}
        if name == "bash":
            exit_code = self._bash_exit_codes.pop(0) if self._bash_exit_codes else 0
            return {
                "command": kwargs["command"],
                "exit_code": exit_code,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            }
        raise AssertionError(f"unexpected tool: {name}")

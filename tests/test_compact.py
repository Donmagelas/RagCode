from app.memory.compact import compact_messages


def test_compact_messages_keeps_recent_messages_and_summarizes_old_ones() -> None:
    messages = [
        {"role": "user", "content": "需求 A"},
        {"role": "assistant", "content": "计划 A"},
        {"role": "user", "content": "需求 B"},
        {"role": "assistant", "content": "计划 B"},
    ]

    result = compact_messages(messages, preserve_recent_messages=2)

    assert result.removed_message_count == 2
    assert "需求 A" in result.summary
    assert result.messages == messages[-2:]


def test_compact_messages_does_not_split_tool_pair_at_boundary() -> None:
    messages = [
        {"role": "user", "content": "开始"},
        {"role": "assistant", "content": "调用工具", "tool_use_id": "tool-1"},
        {"role": "tool", "content": "工具结果", "tool_use_id": "tool-1"},
        {"role": "assistant", "content": "完成"},
    ]

    result = compact_messages(messages, preserve_recent_messages=2)

    assert result.messages[0]["role"] == "assistant"
    assert result.messages[0]["tool_use_id"] == "tool-1"
    assert result.messages[1]["role"] == "tool"
    assert result.messages[1]["tool_use_id"] == "tool-1"

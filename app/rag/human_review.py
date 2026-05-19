from __future__ import annotations


def parse_selected_indexes(text: str, *, max_index: int) -> list[int]:
    """解析人工确认输入的 chunk 序号，序号从 1 开始。"""
    indexes: list[int] = []
    seen: set[int] = set()
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part.isdigit():
            continue
        value = int(part)
        if value < 1 or value > max_index or value in seen:
            continue
        indexes.append(value)
        seen.add(value)
    return indexes

from app.rag.markdown_parser import parse_markdown_document


class CharTokenCounter:
    def encode(self, text: str) -> list[str]:
        return list(text)

    def decode(self, token_ids: list[str]) -> str:
        return "".join(token_ids)


def test_parse_markdown_document_builds_heading_tree_and_own_content() -> None:
    markdown = """文档介绍。

# UI 系统

UI 根说明。

## UIPanel

面板说明。

```python
# 这里不是标题
print("hello")
```

### 打开面板

打开说明。

## Widget

Widget 说明。
"""

    chunks = parse_markdown_document("ui", "ui.md", markdown)

    intro = chunks[0]
    ui = chunks[1]
    panel = chunks[2]
    open_panel = chunks[3]
    widget = chunks[4]

    assert intro.heading == "__document_intro__"
    assert intro.heading_level == 0
    assert intro.own_content.strip() == "文档介绍。"

    assert ui.heading == "UI 系统"
    assert ui.heading_path == ["UI 系统"]
    assert "UI 根说明。" in ui.own_content
    assert "面板说明。" not in ui.own_content

    assert panel.parent_id == ui.id
    assert panel.next_sibling_id == widget.id
    assert widget.prev_sibling_id == panel.id
    assert open_panel.parent_id == panel.id
    assert "# 这里不是标题" in panel.own_content


def test_parse_markdown_document_splits_long_own_content_into_part_children() -> None:
    markdown = "# 长章节\n\n" + "\n".join(f"第 {index} 行内容。" for index in range(12))

    chunks = parse_markdown_document(
        "long",
        "long.md",
        markdown,
        max_chunk_chars=36,
        min_chunk_chars=1,
    )

    parent = chunks[0]
    parts = chunks[1:]

    assert parent.heading == "长章节"
    assert parent.own_content == ""
    assert parent.node_type == "section"
    assert parent.structural_only is True
    assert len(parts) > 1
    assert all(part.parent_id == parent.id for part in parts)
    assert all(part.node_type == "part" for part in parts)
    assert all(part.structural_only is False for part in parts)
    assert all(part.heading.startswith("长章节 part ") for part in parts)
    assert "第 0 行内容。" in parts[0].own_content


def test_parse_markdown_document_splits_long_own_content_by_token_counter() -> None:
    markdown = "# 长章节\n\nabcdefghij"

    chunks = parse_markdown_document(
        "long",
        "long.md",
        markdown,
        max_chunk_tokens=4,
        chunk_overlap_tokens=1,
        min_chunk_tokens=1,
        token_counter=CharTokenCounter(),
    )

    parent = chunks[0]
    parts = chunks[1:]

    assert parent.own_content == ""
    assert [part.own_content for part in parts] == ["abcd", "defg", "ghij"]
def test_parse_markdown_document_ignores_frontmatter_content() -> None:
    markdown = """---
name: ui
description: UI skill
framework_name: aurora
framework_version: v1
---

# UI System

Use WindowBase to create UI windows.
"""

    chunks = parse_markdown_document("ui", "ui.md", markdown)

    assert len(chunks) == 1
    assert chunks[0].heading == "UI System"
    assert "WindowBase" in chunks[0].raw_markdown
    assert "framework_name" not in chunks[0].raw_markdown

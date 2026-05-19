from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocChunk:
    id: str
    doc_id: str
    file_path: str
    heading: str
    heading_level: int
    heading_path: list[str]
    sort_order: int
    parent_id: str | None = None
    prev_sibling_id: str | None = None
    next_sibling_id: str | None = None
    own_content: str = ""
    raw_markdown: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

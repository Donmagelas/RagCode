from __future__ import annotations

from dataclasses import dataclass

from markdown_it import MarkdownIt

from app.rag.schemas import DocChunk
from app.rag.token_counter import HeuristicTokenCounter, TokenCounter


@dataclass
class _Heading:
    title: str
    level: int
    start_line: int
    end_line: int


def parse_markdown_document(
    doc_id: str,
    file_path: str,
    markdown: str,
    *,
    max_chunk_tokens: int | None = None,
    chunk_overlap_tokens: int = 0,
    min_chunk_tokens: int = 80,
    token_counter: TokenCounter | None = None,
    max_chunk_chars: int | None = 6000,
    min_chunk_chars: int = 80,
) -> list[DocChunk]:
    """按 Markdown 标题树构建 chunk；代码块里的 # 不会被当作标题。"""
    markdown = _strip_frontmatter(markdown)
    lines = markdown.splitlines()
    headings = _extract_headings(markdown)
    chunks: list[DocChunk] = []
    stack: list[DocChunk] = []

    if headings and _slice_lines(lines, 0, headings[0].start_line).strip():
        intro = DocChunk(
            id=_chunk_id(doc_id, 0),
            doc_id=doc_id,
            file_path=file_path,
            heading="__document_intro__",
            heading_level=0,
            heading_path=["__document_intro__"],
            sort_order=0,
            own_content=_slice_lines(lines, 0, headings[0].start_line),
            raw_markdown=_slice_lines(lines, 0, headings[0].start_line),
        )
        chunks.append(intro)

    for index, heading in enumerate(headings):
        while stack and stack[-1].heading_level >= heading.level:
            stack.pop()

        parent = stack[-1] if stack else None
        heading_path = [*parent.heading_path, heading.title] if parent else [heading.title]
        next_heading_start = (
            headings[index + 1].start_line if index + 1 < len(headings) else len(lines)
        )
        own_content = _slice_lines(lines, heading.end_line, next_heading_start).strip("\n")
        heading_markdown = _slice_lines(lines, heading.start_line, heading.end_line).strip("\n")

        chunk = DocChunk(
            id=_chunk_id(doc_id, len(chunks)),
            doc_id=doc_id,
            file_path=file_path,
            heading=heading.title,
            heading_level=heading.level,
            heading_path=heading_path,
            sort_order=len(chunks),
            parent_id=parent.id if parent else None,
            own_content=own_content,
            raw_markdown=_join_markdown(heading_markdown, own_content),
        )

        _link_sibling(chunks, chunk)
        chunks.append(chunk)
        stack.append(chunk)

        if max_chunk_tokens is not None:
            counter = token_counter or HeuristicTokenCounter()
            if len(counter.encode(own_content)) > max_chunk_tokens:
                _split_long_chunk_by_tokens(
                    chunks,
                    chunk,
                    heading_markdown,
                    max_chunk_tokens,
                    chunk_overlap_tokens,
                    min_chunk_tokens,
                    counter,
                )
        elif max_chunk_chars is not None and len(own_content) > max_chunk_chars:
            _split_long_chunk(chunks, chunk, heading_markdown, max_chunk_chars, min_chunk_chars)

    return chunks


def _extract_headings(markdown: str) -> list[_Heading]:
    parser = MarkdownIt("commonmark")
    tokens = parser.parse(markdown)
    headings: list[_Heading] = []

    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        inline = tokens[index + 1] if index + 1 < len(tokens) else None
        title = inline.content.strip() if inline is not None and inline.type == "inline" else ""
        level = int(token.tag[1])
        headings.append(
            _Heading(
                title=title,
                level=level,
                start_line=token.map[0],
                end_line=token.map[1],
            )
        )

    return headings


def _strip_frontmatter(markdown: str) -> str:
    """去掉 Markdown frontmatter，frontmatter 只作为文档级元数据使用。"""
    if not markdown.startswith("---\n"):
        return markdown
    end_marker = "\n---\n"
    end = markdown.find(end_marker, 4)
    if end == -1:
        return markdown
    return markdown[end + len(end_marker) :]


def _split_long_chunk(
    chunks: list[DocChunk],
    parent: DocChunk,
    heading_markdown: str,
    max_chunk_chars: int,
    min_chunk_chars: int,
) -> None:
    # 父节点保留结构，正文转移到 part 子节点，符合“只返回 own_content”的设计。
    original = parent.own_content
    parent.own_content = ""
    parent.raw_markdown = heading_markdown

    pieces = _split_text(original, max_chunk_chars, min_chunk_chars)
    previous_part: DocChunk | None = None
    for part_index, piece in enumerate(pieces, start=1):
        part = DocChunk(
            id=_chunk_id(parent.doc_id, len(chunks)),
            doc_id=parent.doc_id,
            file_path=parent.file_path,
            heading=f"{parent.heading} part {part_index}",
            heading_level=parent.heading_level + 1,
            heading_path=[*parent.heading_path, f"part {part_index}"],
            sort_order=len(chunks),
            parent_id=parent.id,
            prev_sibling_id=previous_part.id if previous_part else None,
            own_content=piece,
            raw_markdown=piece,
            metadata=dict(parent.metadata),
        )
        if previous_part is not None:
            previous_part.next_sibling_id = part.id
        chunks.append(part)
        previous_part = part


def _split_long_chunk_by_tokens(
    chunks: list[DocChunk],
    parent: DocChunk,
    heading_markdown: str,
    max_chunk_tokens: int,
    chunk_overlap_tokens: int,
    min_chunk_tokens: int,
    token_counter: TokenCounter,
) -> None:
    # 父节点保留标题结构，正文按 token 窗口转移到 part 子节点。
    original = parent.own_content
    parent.own_content = ""
    parent.raw_markdown = heading_markdown

    pieces = _split_text_by_tokens(
        original,
        max_chunk_tokens=max_chunk_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
        token_counter=token_counter,
    )
    previous_part: DocChunk | None = None
    for part_index, piece in enumerate(pieces, start=1):
        part = DocChunk(
            id=_chunk_id(parent.doc_id, len(chunks)),
            doc_id=parent.doc_id,
            file_path=parent.file_path,
            heading=f"{parent.heading} part {part_index}",
            heading_level=parent.heading_level + 1,
            heading_path=[*parent.heading_path, f"part {part_index}"],
            sort_order=len(chunks),
            parent_id=parent.id,
            prev_sibling_id=previous_part.id if previous_part else None,
            own_content=piece,
            raw_markdown=piece,
            metadata=dict(parent.metadata),
        )
        if previous_part is not None:
            previous_part.next_sibling_id = part.id
        chunks.append(part)
        previous_part = part


def _split_text(text: str, max_chars: int, min_chars: int) -> list[str]:
    paragraphs = [part for part in text.splitlines() if part.strip()]
    pieces: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        candidate = "\n".join([*current, paragraph]).strip()
        if current and len(candidate) > max_chars and len("\n".join(current)) >= min_chars:
            pieces.append("\n".join(current).strip())
            current = [paragraph]
        else:
            current.append(paragraph)

    if current:
        pieces.append("\n".join(current).strip())
    return pieces or [text]


def _split_text_by_tokens(
    text: str,
    *,
    max_chunk_tokens: int,
    chunk_overlap_tokens: int,
    min_chunk_tokens: int,
    token_counter: TokenCounter,
) -> list[str]:
    token_ids = token_counter.encode(text)
    if len(token_ids) <= max_chunk_tokens:
        return [text]

    overlap = max(0, min(chunk_overlap_tokens, max_chunk_tokens - 1))
    step = max_chunk_tokens - overlap
    pieces: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_chunk_tokens, len(token_ids))
        piece_ids = token_ids[start:end]
        if len(piece_ids) >= min_chunk_tokens or not pieces:
            piece = token_counter.decode(piece_ids).strip()
            if piece:
                pieces.append(piece)
        if end == len(token_ids):
            break
        start += step
    return pieces or [text]


def _link_sibling(existing: list[DocChunk], chunk: DocChunk) -> None:
    for previous in reversed(existing):
        if previous.parent_id == chunk.parent_id and previous.heading_level == chunk.heading_level:
            chunk.prev_sibling_id = previous.id
            previous.next_sibling_id = chunk.id
            return


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start:end])


def _join_markdown(heading_markdown: str, own_content: str) -> str:
    return "\n\n".join(part for part in [heading_markdown, own_content] if part)


def _chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}#chunk-{index:04d}"

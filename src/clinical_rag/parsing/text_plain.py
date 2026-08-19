from __future__ import annotations

import re
from pathlib import Path

from clinical_rag.schemas import (
    ExtractionMethod,
    MediaType,
    ParsedDocument,
    ParsedPage,
    RawDocument,
    TextBlock,
)

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUM_HEADING = re.compile(r"^[0-9]+(?:\.[0-9]+)*[.\)]\s+\S.{0,100}$")
_CAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 ,/&()\-]{8,100}$")


def _is_heading(line: str, markdown: bool) -> tuple[bool, int, str]:
    stripped = line.strip()
    if not stripped:
        return False, 0, stripped
    if markdown:
        m = _MD_HEADING.match(stripped)
        if m:
            return True, len(m.group(1)), m.group(2).strip()
    if _NUM_HEADING.match(stripped) or _CAPS_HEADING.match(stripped):
        return True, 1, stripped
    return False, 0, stripped


def blocks_from_text(text: str, markdown: bool) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            blocks.append(TextBlock(kind="paragraph", text="\n".join(para).strip()))
            para.clear()

    for line in text.splitlines():
        is_h, level, title = _is_heading(line, markdown)
        if is_h:
            flush_para()
            blocks.append(TextBlock(kind="heading", text=title, heading_level=level))
        elif line.strip():
            para.append(line.rstrip())
        else:
            flush_para()
    flush_para()
    return blocks


def parse_plain(raw: RawDocument) -> ParsedDocument:
    text = Path(raw.path).read_text(encoding="utf-8", errors="replace")
    markdown = raw.media_type is MediaType.md or raw.filename.lower().endswith(".md")
    blocks = blocks_from_text(text, markdown=markdown)
    page = ParsedPage(
        page_number=1,
        text=text,
        extraction_method=ExtractionMethod.text,
        blocks=blocks,
    )
    return ParsedDocument(
        doc_id=raw.doc_id,
        document_name=raw.document_name,
        source_url=raw.source_url,
        media_type=raw.media_type,
        filename=raw.filename,
        pages=[page],
    )

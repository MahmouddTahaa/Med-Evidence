from __future__ import annotations

from clinical_rag.chunking.base import (
    UNKNOWN,
    DraftChunk,
    count_tokens,
    fallback_blocks,
    window_texts,
)
from clinical_rag.schemas import ChunkConfig, ParsedDocument


def _sections(parsed: ParsedDocument) -> list[DraftChunk]:
    sections: list[DraftChunk] = []
    title = UNKNOWN
    buf: list[str] = []
    page_number = parsed.pages[0].page_number if parsed.pages else 1
    method = parsed.pages[0].extraction_method if parsed.pages else None

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text and method is not None:
            sections.append(
                DraftChunk(
                    text=text,
                    section_title=title,
                    page_number=page_number,
                    extraction_method=method,
                )
            )
        buf.clear()

    for page in parsed.pages:
        for block in fallback_blocks(page):
            if block.kind == "heading":
                flush()
                title = block.text.strip() or UNKNOWN
                page_number = page.page_number
                method = page.extraction_method
                buf.append(block.text.strip())
            else:
                if not buf:
                    page_number = page.page_number
                    method = page.extraction_method
                buf.append(block.text)
    flush()
    return sections


def chunk_section_aware(parsed: ParsedDocument, cfg: ChunkConfig) -> list[DraftChunk]:
    packed: list[DraftChunk] = []
    acc: DraftChunk | None = None
    for section in _sections(parsed):
        if acc is None:
            acc = section
            continue
        combined = f"{acc.text}\n\n{section.text}"
        if count_tokens(combined) <= cfg.target_tokens:
            acc = DraftChunk(
                text=combined,
                section_title=acc.section_title,
                page_number=acc.page_number,
                extraction_method=acc.extraction_method,
            )
        else:
            packed.extend(_emit(acc, cfg))
            acc = section
    if acc is not None:
        packed.extend(_emit(acc, cfg))
    return packed


def _emit(draft: DraftChunk, cfg: ChunkConfig) -> list[DraftChunk]:
    if count_tokens(draft.text) <= cfg.max_tokens:
        return [draft]
    return [
        DraftChunk(
            text=part,
            section_title=draft.section_title,
            page_number=draft.page_number,
            extraction_method=draft.extraction_method,
        )
        for part in window_texts(draft.text, cfg.target_tokens, cfg.overlap_ratio)
        if part.strip()
    ]

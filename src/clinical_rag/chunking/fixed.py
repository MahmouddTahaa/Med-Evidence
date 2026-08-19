from __future__ import annotations

from clinical_rag.chunking.base import DraftChunk, decode_tokens, token_stream, window_ranges
from clinical_rag.schemas import ChunkConfig, ParsedDocument


def chunk_fixed(parsed: ParsedDocument, cfg: ChunkConfig) -> list[DraftChunk]:
    tokens, pages, methods, titles = token_stream(parsed)
    drafts: list[DraftChunk] = []
    for start, end in window_ranges(len(tokens), cfg.target_tokens, cfg.overlap_ratio):
        text = decode_tokens(tokens[start:end]).strip()
        if not text:
            continue
        drafts.append(
            DraftChunk(
                text=text,
                section_title=titles[start] if titles else "(unknown)",
                page_number=pages[start] if pages else None,
                extraction_method=methods[start] if methods else parsed.pages[0].extraction_method,
            )
        )
    return drafts

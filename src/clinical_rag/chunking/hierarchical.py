from __future__ import annotations

from clinical_rag.chunking.base import DraftChunk, decode_tokens, token_stream, window_ranges
from clinical_rag.schemas import ChunkConfig, ParsedDocument


def chunk_hierarchical(parsed: ParsedDocument, cfg: ChunkConfig, doc_id: str) -> list[DraftChunk]:
    tokens, pages, methods, titles = token_stream(parsed)
    children: list[DraftChunk] = []
    for p_i, (p_start, p_end) in enumerate(
        window_ranges(len(tokens), cfg.parent_tokens, cfg.overlap_ratio), start=1
    ):
        parent_id = f"{doc_id}-parent-{p_i:04d}"
        parent_tokens = tokens[p_start:p_end]
        for c_start, c_end in window_ranges(len(parent_tokens), cfg.child_tokens, cfg.overlap_ratio):
            abs_start = p_start + c_start
            text = decode_tokens(parent_tokens[c_start:c_end]).strip()
            if not text:
                continue
            children.append(
                DraftChunk(
                    text=text,
                    section_title=titles[abs_start] if titles else "(unknown)",
                    page_number=pages[abs_start] if pages else None,
                    extraction_method=methods[abs_start],
                    parent_chunk_id=parent_id,
                )
            )
    return children

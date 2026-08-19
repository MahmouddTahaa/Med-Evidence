"""Turn retrieved hits into numbered citation blocks for a generator prompt.

No LLM, no system prompt. Generation owns wrapping; this owns the citation shape.
"""

from __future__ import annotations

from clinical_rag.schemas import SmokeHit

RetrievedChunk = SmokeHit


def citation_blocks(hits: list[SmokeHit], *, max_chars: int | None = None) -> str:
    """Format hits as `[n] document · section · page · chunk_id` plus body text."""
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        loc = []
        if hit.page_number is not None:
            loc.append(f"page {hit.page_number}")
        loc.append(hit.chunk_id)
        header = (
            f"[{i}] {hit.document_name} · {hit.section_title} · " + " · ".join(loc)
        )
        body = hit.text or ""
        if max_chars is not None and max_chars >= 0 and len(body) > max_chars:
            body = body[:max_chars].rstrip() + "…"
        parts.append(f"{header}\n{body}".rstrip())
    return "\n\n".join(parts)

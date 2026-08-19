"""Pack remaining slots with other windows of the rank-1 (document, section).

Rank 1 is frozen. Later windows of the same heading are often missed by dense /
BM25 / cross-encoder because they no longer contain the heading string. Gold is
filename+section union, so this packing matches the label policy rather than
changing it. Off unless RetrievalConfig.sibling_fill is set.
"""

from __future__ import annotations

from typing import Any

from clinical_rag.indexing.payload import hit_from_meta
from clinical_rag.schemas import SmokeHit


def fill_section_siblings(
    hits: list[SmokeHit],
    chunks: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[SmokeHit]:
    """Keep hit 0; insert same document_name + section_title windows after it."""
    if not hits or top_k < 1:
        return []
    first = hits[0]
    section = (first.section_title or "").strip().lower()
    if not section:
        return hits[:top_k]

    existing = {h.chunk_id: h for h in hits}
    sibling_ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    doc = first.document_name
    for row in chunks:
        cid = str(row.get("chunk_id") or "")
        if not cid:
            continue
        by_id[cid] = row
        if str(row.get("document_name") or "") != doc:
            continue
        if str(row.get("section_title") or "").strip().lower() != section:
            continue
        sibling_ids.append(cid)

    out: list[SmokeHit] = [first]
    used = {first.chunk_id}
    for cid in sibling_ids:
        if len(out) >= top_k:
            break
        if cid in used:
            continue
        if cid in existing:
            out.append(existing[cid])
        else:
            row = by_id[cid]
            out.append(hit_from_meta(row, text=str(row.get("text") or ""), score=first.score))
        used.add(cid)

    for hit in hits[1:]:
        if len(out) >= top_k:
            break
        if hit.chunk_id in used:
            continue
        out.append(hit)
        used.add(hit.chunk_id)
    return out[:top_k]

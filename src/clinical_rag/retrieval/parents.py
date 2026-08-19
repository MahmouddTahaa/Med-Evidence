"""Pull other windows of a retrieved parent into the candidate pool.

Unlike sibling fill (which freezes rank 1 and packs), this only *exposes*
same-parent chunks to the ranker. Hierarchical jobs group by parent_chunk_id;
section_aware falls back to (document_name, section_title).
"""

from __future__ import annotations

from typing import Any

from clinical_rag.indexing.payload import hit_from_meta
from clinical_rag.schemas import SmokeHit

_ParentKey = tuple[str, ...]


def _parent_key(row: dict[str, Any]) -> _ParentKey:
    pid = str(row.get("parent_chunk_id") or "").strip()
    if pid:
        return ("parent", pid)
    doc = str(row.get("document_name") or "")
    sec = str(row.get("section_title") or "").strip().lower()
    if doc and sec:
        return ("section", doc, sec)
    return ("chunk", str(row.get("chunk_id") or ""))


def expand_parent_children(
    hits: list[SmokeHit],
    chunks: list[dict[str, Any]],
    *,
    limit: int,
) -> list[SmokeHit]:
    """Insert other children of each retrieved parent; do not reorder by score."""
    if not hits or limit < 1:
        return []

    by_id = {str(row.get("chunk_id") or ""): row for row in chunks if row.get("chunk_id")}
    groups: dict[_ParentKey, list[dict[str, Any]]] = {}
    for row in chunks:
        cid = str(row.get("chunk_id") or "")
        if not cid:
            continue
        groups.setdefault(_parent_key(row), []).append(row)

    existing = {h.chunk_id: h for h in hits}
    out: list[SmokeHit] = []
    used: set[str] = set()

    def _append_row(row: dict[str, Any], score: float) -> None:
        cid = str(row.get("chunk_id") or "")
        if not cid or cid in used or len(out) >= limit:
            return
        if cid in existing:
            out.append(existing[cid])
        else:
            out.append(hit_from_meta(row, text=str(row.get("text") or ""), score=score))
        used.add(cid)

    for hit in hits:
        if len(out) >= limit:
            break
        row = by_id.get(hit.chunk_id)
        key = _parent_key(row) if row is not None else (
            "section",
            hit.document_name,
            (hit.section_title or "").strip().lower(),
        )
        members = groups.get(key)
        if not members:
            if hit.chunk_id not in used:
                out.append(hit)
                used.add(hit.chunk_id)
            continue
        for member in members:
            _append_row(member, hit.score)

    for hit in hits:
        if len(out) >= limit:
            break
        if hit.chunk_id in used:
            continue
        out.append(hit)
        used.add(hit.chunk_id)
    return out[:limit]

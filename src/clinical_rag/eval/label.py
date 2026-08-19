"""Map eval templates to gold chunk ids.

When a template has `filename` + `section_title`, gold is every chunk from that
article section (union any extra anchor hits). Anchor-only matching under-labels
sibling section chunks and caps Precision@k at ~1/k even when ranking is correct.
"""

from __future__ import annotations

from clinical_rag.errors import IngestError
from clinical_rag.eval.runner import EvalQuestion


def _anchor_hits(anchor: str, chunks: list[dict]) -> list[str]:
    gold = [str(c["chunk_id"]) for c in chunks if anchor and anchor in str(c.get("text") or "")]
    if gold:
        return gold
    tokens = [t.lower() for t in anchor.split() if len(t) > 3]
    if not tokens:
        return []
    out: list[str] = []
    for c in chunks:
        text = str(c.get("text") or "").lower()
        if all(t in text for t in tokens):
            out.append(str(c["chunk_id"]))
    return out


def label_from_chunks(templates: list[dict], chunks: list[dict]) -> list[EvalQuestion]:
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        by_file.setdefault(str(c.get("filename") or ""), []).append(c)

    out: list[EvalQuestion] = []
    for row in templates:
        qid = str(row.get("id") or "")
        anchor = str(row.get("anchor") or "").strip()
        filename = str(row.get("filename") or "")
        section = str(row.get("section_title") or "").strip().lower()
        if not anchor:
            raise IngestError(f"{qid or row}: missing anchor")

        gold: list[str] = []
        seen: set[str] = set()

        if filename and section:
            for c in by_file.get(filename, []):
                sec = str(c.get("section_title") or "").strip().lower()
                cid = str(c.get("chunk_id") or "")
                if cid and sec == section and cid not in seen:
                    seen.add(cid)
                    gold.append(cid)

        for cid in _anchor_hits(anchor, chunks):
            if cid not in seen:
                seen.add(cid)
                gold.append(cid)

        if not gold:
            raise IngestError(
                f"{qid}: no chunk contains anchor {anchor!r} "
                f"(file={filename or row.get('filename')})"
            )
        out.append(
            EvalQuestion(
                id=qid,
                query=str(row["query"]),
                relevant_chunk_ids=gold,
                notes=str(row.get("notes") or ""),
            )
        )
    return out

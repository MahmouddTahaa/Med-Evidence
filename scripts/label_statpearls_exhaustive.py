"""Build exhaustive-ish gold labels and audit chunk-id coverage for StatPearls eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clinical_rag.errors import IngestError


def label_exhaustive(
    templates: list[dict],
    chunks: list[dict],
) -> tuple[list[dict], dict]:
    """Label gold as: anchor containment OR (same filename + matching section)."""
    by_file: dict[str, list[dict]] = {}
    indexed_ids: set[str] = set()
    for c in chunks:
        cid = str(c.get("chunk_id") or "")
        if cid:
            indexed_ids.add(cid)
        fn = str(c.get("filename") or "")
        by_file.setdefault(fn, []).append(c)

    labeled: list[dict] = []
    audit_queries: list[dict] = []
    missing_files = 0
    empty_gold = 0

    for row in templates:
        qid = str(row.get("id") or "")
        anchor = str(row.get("anchor") or "").strip()
        filename = str(row.get("filename") or "")
        section = str(row.get("section_title") or "").strip().lower()
        file_chunks = by_file.get(filename, [])
        if not file_chunks:
            missing_files += 1

        gold: list[str] = []
        via_anchor: list[str] = []
        via_section: list[str] = []
        for c in file_chunks:
            cid = str(c.get("chunk_id") or "")
            text = str(c.get("text") or "")
            sec = str(c.get("section_title") or "").strip().lower()
            if anchor and anchor in text:
                via_anchor.append(cid)
            if section and sec == section:
                via_section.append(cid)
        # Union, preserve order: section hits first (broader), then anchor-only
        seen: set[str] = set()
        for cid in via_section + via_anchor:
            if cid and cid not in seen:
                seen.add(cid)
                gold.append(cid)

        orphans = [cid for cid in gold if cid not in indexed_ids]
        gold = [cid for cid in gold if cid in indexed_ids]
        if not gold:
            empty_gold += 1
            raise IngestError(
                f"{qid}: no gold chunks for file={filename!r} section={section!r}"
            )

        labeled.append(
            {
                "id": qid,
                "query": row["query"],
                "relevant_chunk_ids": gold,
                "notes": row.get("notes") or "",
                "anchor": anchor,
                "filename": filename,
                "section_title": row.get("section_title") or "",
                "n_gold": len(gold),
                "n_via_anchor": len(via_anchor),
                "n_via_section": len(via_section),
            }
        )
        audit_queries.append(
            {
                "id": qid,
                "filename": filename,
                "section_title": row.get("section_title") or "",
                "n_gold": len(gold),
                "n_via_anchor": len(set(via_anchor)),
                "n_via_section": len(set(via_section)),
                "orphans_dropped": orphans,
                "all_gold_in_index": True,
            }
        )

    audit = {
        "n_templates": len(templates),
        "n_chunks": len(chunks),
        "n_indexed_ids": len(indexed_ids),
        "missing_source_files": missing_files,
        "empty_gold": empty_gold,
        "queries": audit_queries,
        "exhaustive_policy": "filename + section_title match OR anchor substring containment",
        "note": (
            "Not fully exhaustive of all semantically relevant chunks across the corpus; "
            "gold is exhaustive within the source article section (+ any anchor hits)."
        ),
    }
    return labeled, audit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--templates", type=Path, required=True)
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    args = p.parse_args()

    templates = [
        json.loads(line)
        for line in args.templates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    labeled, audit = label_exhaustive(templates, chunks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in labeled) + "\n",
        encoding="utf-8",
    )
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote {len(labeled)} questions -> {args.out}")
    print(f"Audit -> {args.audit}")
    print(
        "gold sizes:",
        sorted({r["n_gold"] for r in labeled}),
        "mean",
        round(sum(r["n_gold"] for r in labeled) / len(labeled), 2),
    )


if __name__ == "__main__":
    main()

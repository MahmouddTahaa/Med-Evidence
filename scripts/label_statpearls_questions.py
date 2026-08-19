"""Label StatPearls eval templates against a job's chunks.json via anchor containment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clinical_rag.errors import IngestError


def label_from_chunks(
    templates: list[dict],
    chunks: list[dict],
) -> list[dict]:
    out: list[dict] = []
    for row in templates:
        anchor = str(row.get("anchor") or "").strip()
        if not anchor:
            raise IngestError(f"{row.get('id')}: missing anchor")
        gold = [
            str(c["chunk_id"])
            for c in chunks
            if anchor in str(c.get("text") or "")
        ]
        if not gold:
            # Fallback: require all anchor tokens (len>3) to appear
            tokens = [t.lower() for t in anchor.split() if len(t) > 3]
            gold = []
            for c in chunks:
                text = str(c.get("text") or "").lower()
                if tokens and all(t in text for t in tokens):
                    gold.append(str(c["chunk_id"]))
        if not gold:
            raise IngestError(
                f"{row.get('id')}: no chunk contains anchor {anchor!r} "
                f"(file={row.get('filename')})"
            )
        out.append(
            {
                "id": row["id"],
                "query": row["query"],
                "relevant_chunk_ids": gold,
                "notes": row.get("notes") or "",
                "anchor": anchor,
                "filename": row.get("filename") or "",
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--templates", type=Path, required=True)
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    templates = [
        json.loads(line)
        for line in args.templates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    labeled = label_from_chunks(templates, chunks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in labeled) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(labeled)} questions -> {args.out}")


if __name__ == "__main__":
    main()

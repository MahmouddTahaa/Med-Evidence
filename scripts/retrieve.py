"""CLI: retrieve cited chunks from the serving index using the frozen stack.

No generation. For the clinician UI, import RetrievalSession instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clinical_rag.errors import IngestError
from clinical_rag.query import RetrievalSession, citation_blocks


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieve from the frozen serving index")
    p.add_argument("query", help="Clinician-style question")
    p.add_argument("--job-id", default=None, help="Override configs/serving.yaml")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--json", action="store_true", help="Print hits as JSON instead of citation blocks")
    return p.parse_args()


def main() -> None:
    args = _args()
    try:
        session = RetrievalSession.open(job_id=args.job_id)
        hits = session.retrieve(args.query, top_k=args.top_k)
    except IngestError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        print(json.dumps([h.model_dump(mode="json") for h in hits], indent=2))
        return
    print(citation_blocks(hits))


if __name__ == "__main__":
    main()

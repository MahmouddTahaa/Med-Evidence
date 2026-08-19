"""CLI: score a job/collection against labeled questions (Day 2 Precision@K)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clinical_rag.config import get_settings
from clinical_rag.eval.runner import load_questions, run_retrieval_eval
from clinical_rag.errors import IngestError
from clinical_rag.schemas import (
    DEFAULT_RERANK_MODEL,
    KeywordMethod,
    RetrievalConfig,
    RetrievalMode,
)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run retrieval eval for a job combination")
    p.add_argument("--job-id", required=True)
    p.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    p.add_argument("--k", default="3,5,10", help="Comma-separated k values")
    p.add_argument("--eval-set-id", default="default")
    p.add_argument(
        "--mode",
        choices=[m.value for m in RetrievalMode],
        default=None,
        help="dense (semantic) | keyword | hybrid (default: settings / dense)",
    )
    p.add_argument(
        "--keyword-method",
        choices=[m.value for m in KeywordMethod],
        default=None,
    )
    p.add_argument(
        "--semantic-weight",
        type=float,
        default=None,
        help="Hybrid semantic weight; keyword_weight = 1 - semantic_weight",
    )
    p.add_argument("--rrf-k", type=int, default=None)
    p.add_argument("--fetch-k", type=int, default=None)
    p.add_argument("--rerank", action="store_true", help="Hybrid only: cross-encoder after RRF")
    p.add_argument("--rerank-model", default=None)
    p.add_argument(
        "--sibling-fill",
        action="store_true",
        help="After ranking, pack remaining slots with other windows of the rank-1 section",
    )
    p.add_argument(
        "--parent-child",
        action="store_true",
        help="Expand same-parent/section windows into the candidate pool before rerank",
    )
    p.add_argument(
        "--templates",
        type=Path,
        default=None,
        help="If set, auto-label gold from this job's chunks.json (StatPearls templates JSONL)",
    )
    return p.parse_args()


def main() -> None:
    args = _args()
    settings = get_settings()
    report_path = Path(settings.jobs_dir) / args.job_id / "report.json"
    if not report_path.is_file():
        raise SystemExit(f"Job report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]

    base = settings.retrieval
    mode = RetrievalMode(args.mode) if args.mode else base.mode
    keyword_method = (
        KeywordMethod(args.keyword_method) if args.keyword_method else base.keyword_method
    )
    if args.semantic_weight is not None:
        sem_w = float(args.semantic_weight)
        kw_w = 1.0 - sem_w
    else:
        sem_w = base.semantic_weight
        kw_w = base.keyword_weight
    retrieval = RetrievalConfig(
        mode=mode,
        top_k=base.top_k,
        keyword_method=keyword_method,
        semantic_weight=sem_w,
        keyword_weight=kw_w,
        rrf_k=args.rrf_k if args.rrf_k is not None else base.rrf_k,
        fetch_k=args.fetch_k if args.fetch_k is not None else base.fetch_k,
        rerank=bool(args.rerank) or base.rerank,
        rerank_model=args.rerank_model or base.rerank_model or DEFAULT_RERANK_MODEL,
        rerank_top_n=base.rerank_top_n,
        sibling_fill=bool(args.sibling_fill),
        parent_child=bool(args.parent_child),
    )

    try:
        if args.templates is not None:
            from clinical_rag.eval.label import label_from_chunks

            templates_path = args.templates
            if not templates_path.is_file():
                raise IngestError(f"Templates file not found: {templates_path}")
            chunks_path = Path(settings.jobs_dir) / args.job_id / "chunks.json"
            if not chunks_path.is_file():
                raise IngestError(f"chunks.json not found: {chunks_path}")
            templates = [
                json.loads(line)
                for line in templates_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            questions = label_from_chunks(templates, chunks)
        else:
            questions = load_questions(args.questions)
        persist = (
            settings.qdrant.persist_dir
            if report.get("vector_store") == "qdrant"
            else settings.chroma.persist_dir
        )
        result = run_retrieval_eval(
            job_report=report,
            questions=questions,
            persist_dir=str(persist),
            k_values=k_values,
            evals_dir=Path("artifacts/evals"),
            eval_set_id=args.eval_set_id,
            retrieval=retrieval,
            jobs_dir=settings.jobs_dir,
        )
    except IngestError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result.metrics, indent=2))
    print(f"Wrote {result.run_dir}")


if __name__ == "__main__":
    main()

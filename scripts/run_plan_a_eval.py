"""Plan A: index section_aware + bge-small + chroma (full StatPearls), then eval grid.

Retrieval grid:
  - dense (semantic)
  - keyword BM25
  - keyword TF-IDF
  - hybrid RRF weights (0.7/0.3, 0.5/0.5, 0.3/0.7) × BM25, no CE
  - hybrid RRF same weights × BM25 + CE

K = 1,3,5,10. Per-query dumps include retrieved IDs + relevance labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SCRIPTS = Path(__file__).resolve().parent
for _p in (_SRC, _SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=False)

from clinical_rag.config import get_settings, make_doc_id, make_job_id  # noqa: E402
from clinical_rag.eval.runner import load_questions, run_retrieval_eval  # noqa: E402
from clinical_rag.parsing.router import media_type_for  # noqa: E402
from clinical_rag.pipeline.ingest import run_ingest  # noqa: E402
from clinical_rag.schemas import (  # noqa: E402
    ChunkConfig,
    EmbedConfig,
    EmbedProvider,
    IngestJobConfig,
    KeywordMethod,
    LegalFlags,
    ParserConfig,
    RawDocument,
    RetrievalConfig,
    RetrievalMode,
    StrategyId,
    VectorStoreKind,
)
from label_statpearls_exhaustive import label_exhaustive  # noqa: E402

NXML_DIR = Path("data/pharmacology_data/statpearls_NBK430685")
TEMPLATES = Path("data/eval/statpearls_pharmacology_templates.jsonl")
OUT_DIR = Path("artifacts/plan_a")
EVALS_DIR = Path("artifacts/evals")
CORPUS_ID = "statpearls_pharmacology"
K_VALUES = [1, 3, 5, 10]


def retrieval_grid() -> list[RetrievalConfig]:
    configs = [
        RetrievalConfig(mode=RetrievalMode.dense, top_k=10),
        RetrievalConfig(mode=RetrievalMode.keyword, top_k=10, keyword_method=KeywordMethod.bm25),
        RetrievalConfig(mode=RetrievalMode.keyword, top_k=10, keyword_method=KeywordMethod.tfidf),
    ]
    for sem, kw in ((0.7, 0.3), (0.5, 0.5), (0.3, 0.7)):
        for rerank in (False, True):
            configs.append(
                RetrievalConfig(
                    mode=RetrievalMode.hybrid,
                    top_k=10,
                    keyword_method=KeywordMethod.bm25,
                    semantic_weight=sem,
                    keyword_weight=kw,
                    fetch_k=20,
                    rrf_k=60,
                    rerank=rerank,
                    rerank_top_n=20,
                )
            )
    return configs


def collect_files() -> list[RawDocument]:
    legal = LegalFlags(
        open_access_or_reusable=True,
        redistribution_ok_for_indexing=True,
        edition_current=True,
        attribution_documented=True,
    )
    # Ensure template source articles are included (they are under NXML_DIR)
    paths = sorted(NXML_DIR.glob("*.nxml"))
    files: list[RawDocument] = []
    used: dict[str, int] = {}
    for path in paths:
        doc_id = make_doc_id(path.name)
        used[doc_id] = used.get(doc_id, 0) + 1
        if used[doc_id] > 1:
            doc_id = f"{doc_id}-{used[doc_id]}"
        files.append(
            RawDocument(
                doc_id=doc_id,
                filename=path.name,
                media_type=media_type_for(path.name),
                document_name=path.stem.replace("_", " "),
                source_url="https://www.ncbi.nlm.nih.gov/books/NBK430685/",
                path=str(path.resolve()),
                legal=legal,
            )
        )
    return files


def run_ingest_job() -> dict:
    settings = get_settings()
    files = collect_files()
    print(f"Ingesting {len(files)} NXML files…", flush=True)
    config = IngestJobConfig(
        corpus_id=CORPUS_ID,
        job_id=make_job_id(),
        files=files,
        parser=ParserConfig(),
        chunk=ChunkConfig(strategy_id=StrategyId.section_aware),
        embed=EmbedConfig(
            model_id="BAAI/bge-small-en-v1.5",
            provider=EmbedProvider.sentence_transformers,
            device=settings.embed.device,
            batch_size=settings.embed.batch_size,
        ),
        chroma=settings.chroma,
        qdrant=settings.qdrant,
        vector_store=VectorStoreKind.chroma,
        smoke_query=settings.smoke_query,
    )

    def progress(stage: str, msg: str) -> None:
        print(f"  [{stage}] {msg}", flush=True)

    report, _ = run_ingest(config, progress=progress, jobs_dir=settings.jobs_dir)
    return report.model_dump()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-id", default="", help="Reuse an existing job instead of ingesting")
    p.add_argument("--skip-ingest", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    state_path = OUT_DIR / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"evals": {}}

    if args.job_id:
        report_path = Path(settings.jobs_dir) / args.job_id / "report.json"
        job_report = json.loads(report_path.read_text(encoding="utf-8"))
    elif args.skip_ingest and state.get("job_id"):
        report_path = Path(settings.jobs_dir) / state["job_id"] / "report.json"
        job_report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        job_report = run_ingest_job()
        state["job_id"] = job_report["job_id"]
        state["ingest_timestamp"] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(state, indent=2))

    job_id = job_report["job_id"]
    chunks_path = Path(settings.jobs_dir) / job_id / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    templates = [
        json.loads(line)
        for line in TEMPLATES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labeled, audit = label_exhaustive(templates, chunks)
    q_path = OUT_DIR / "questions.jsonl"
    audit_path = OUT_DIR / "gold_audit.json"
    q_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in labeled) + "\n",
        encoding="utf-8",
    )
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Gold audit -> {audit_path}")
    print(
        f"Questions -> {q_path} (gold sizes {[r['n_gold'] for r in labeled]})",
        flush=True,
    )

    # Chunk-id coverage check
    indexed = {str(c["chunk_id"]) for c in chunks}
    for r in labeled:
        missing = [c for c in r["relevant_chunk_ids"] if c not in indexed]
        if missing:
            raise SystemExit(f"{r['id']}: gold IDs not in index: {missing}")
    print("All gold chunk IDs present in indexed chunks.json", flush=True)

    questions = load_questions(q_path)
    for cfg in retrieval_grid():
        key = (
            f"{cfg.mode.value}|{cfg.keyword_method.value}|"
            f"{cfg.semantic_weight}|{cfg.keyword_weight}|rerank={cfg.rerank}"
        )
        if key in state.get("evals", {}):
            print(f"skip {key}")
            continue
        print(f"EVAL {key} …", flush=True)
        result = run_retrieval_eval(
            job_report=job_report,
            questions=questions,
            persist_dir=settings.chroma.persist_dir,
            k_values=K_VALUES,
            evals_dir=EVALS_DIR,
            eval_set_id="statpearls_plan_a",
            retrieval=cfg,
            jobs_dir=settings.jobs_dir,
        )
        # Copy per-query with labels into plan_a folder
        dest = OUT_DIR / "per_query" / f"{result.run_id}.jsonl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = EVALS_DIR / result.run_id / "per_query.jsonl"
        dest.write_bytes(src.read_bytes())
        state.setdefault("evals", {})[key] = {
            "run_id": result.run_id,
            "aggregates": result.metrics.get("aggregates"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        state_path.write_text(json.dumps(state, indent=2))
        agg = result.metrics["aggregates"]
        print(
            f"  → {result.run_id} mrr={agg.get('mrr')} "
            f"P@5={agg.get('precision@5')} R@5={agg.get('recall@5')} "
            f"Hit@5={agg.get('hit@5')} nDCG@5={agg.get('ndcg@5')}",
            flush=True,
        )

    print("Plan A complete.")
    print(f"Leaderboard: {EVALS_DIR / 'leaderboard.csv'}")
    print(f"Per-query dumps: {OUT_DIR / 'per_query'}")


if __name__ == "__main__":
    main()

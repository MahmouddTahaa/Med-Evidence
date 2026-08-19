from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from clinical_rag.adapters.embedders import embedder_for_index
from clinical_rag.config import get_settings, store_kwargs
from clinical_rag.errors import IngestError
from clinical_rag.eval.metrics import (
    aggregate_mean,
    hit_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from clinical_rag.retrieval.pipeline import run_retrieve
from clinical_rag.retrieval.sparse import SparseIndex
from clinical_rag.schemas import (
    ChromaConfig,
    EmbedProvider,
    KeywordMethod,
    RetrievalConfig,
    RetrievalMode,
)


@dataclass
class EvalQuestion:
    id: str
    query: str
    relevant_chunk_ids: list[str]
    notes: str = ""


@dataclass
class EvalRunResult:
    run_id: str
    metrics: dict
    per_query: list[dict] = field(default_factory=list)
    run_dir: Path | None = None


def load_questions(path: Path) -> list[EvalQuestion]:
    if not path.is_file():
        raise IngestError(f"Questions file not found: {path}")
    out: list[EvalQuestion] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IngestError(f"{path}:{line_no}: invalid JSONL ({exc})") from exc
        qid = str(row.get("id") or f"q{line_no}")
        query = str(row.get("query") or "").strip()
        gold = row.get("relevant_chunk_ids") or []
        if not query:
            raise IngestError(f"{path}:{line_no}: missing query")
        if not isinstance(gold, list) or not gold:
            raise IngestError(f"{path}:{line_no}: relevant_chunk_ids must be a non-empty list")
        out.append(
            EvalQuestion(
                id=qid,
                query=query,
                relevant_chunk_ids=[str(x) for x in gold],
                notes=str(row.get("notes") or ""),
            )
        )
    if not out:
        raise IngestError(f"No questions in {path}")
    return out


def run_retrieval_eval(
    *,
    job_report: dict,
    questions: list[EvalQuestion],
    persist_dir: str,
    k_values: list[int],
    evals_dir: Path,
    eval_set_id: str = "default",
    retrieval: RetrievalConfig | None = None,
    jobs_dir: Path | None = None,
    reranker=None,
) -> EvalRunResult:
    k_values = sorted({int(k) for k in k_values if int(k) > 0})
    if not k_values:
        raise IngestError("k_values must include at least one positive integer")

    settings = get_settings()
    retrieval_cfg = retrieval or settings.retrieval
    # Eval top_k must cover the largest metric k
    max_k = max(k_values)
    retrieval_cfg = retrieval_cfg.model_copy(update={"top_k": max(max_k, retrieval_cfg.top_k)})

    model_id = job_report["embed_model_id"]
    device = job_report.get("embed_device") or "auto"
    provider = job_report.get("embed_provider") or EmbedProvider.sentence_transformers.value
    collection = job_report["collection_name"]
    job_id = job_report.get("job_id") or ""
    jobs_root = Path(jobs_dir or settings.jobs_dir)
    chunks_path = jobs_root / job_id / "chunks.json"

    needs_embedder = retrieval_cfg.mode is not RetrievalMode.keyword
    embedder = None
    if needs_embedder:
        embedder = embedder_for_index(
            model_id=model_id,
            provider=provider,
            device=device,
            batch_size=8,
            purpose="query",
        )

    sparse_index = None
    chunks_rows: list[dict] | None = None
    needs_chunks = (
        retrieval_cfg.mode in (RetrievalMode.keyword, RetrievalMode.hybrid)
        or retrieval_cfg.sibling_fill
        or retrieval_cfg.parent_child
    )
    if needs_chunks:
        if not chunks_path.is_file():
            raise IngestError(f"chunks.json not found for job {job_id}: {chunks_path}")
        chunks_rows = json.loads(chunks_path.read_text(encoding="utf-8"))
        if retrieval_cfg.mode in (RetrievalMode.keyword, RetrievalMode.hybrid):
            sparse_index = SparseIndex.from_chunks(chunks_rows, method=retrieval_cfg.keyword_method)

    per_query: list[dict] = []
    latencies_ms: list[float] = []
    metric_bags: dict[str, list[float]] = {f"precision@{k}": [] for k in k_values}
    metric_bags.update({f"recall@{k}": [] for k in k_values})
    metric_bags.update({f"hit@{k}": [] for k in k_values})
    metric_bags.update({f"ndcg@{k}": [] for k in k_values})
    mrr_values: list[float] = []

    stores = store_kwargs(settings)
    vector_store = job_report.get("vector_store", "chroma")
    persist = persist_dir
    if vector_store == "qdrant":
        persist = persist_dir or settings.qdrant.persist_dir
        stores["qdrant"] = settings.qdrant.model_copy(update={"persist_dir": persist})
    else:
        persist = persist_dir or settings.chroma.persist_dir
        stores["chroma"] = ChromaConfig(persist_dir=persist)

    active_reranker = reranker
    if retrieval_cfg.mode is RetrievalMode.hybrid and retrieval_cfg.rerank and active_reranker is None:
        from clinical_rag.retrieval.rerank import CrossEncoderReranker

        active_reranker = CrossEncoderReranker(retrieval_cfg.rerank_model, device="cpu")

    for q in questions:
        t0 = time.perf_counter()
        hits = run_retrieve(
            query=q.query,
            retrieval=retrieval_cfg,
            collection=collection,
            chunks=chunks_rows,
            chunks_path=chunks_path if chunks_path.is_file() else None,
            sparse_index=sparse_index,
            embedder=embedder,
            index_model_id=model_id if needs_embedder else None,
            vector_store=vector_store,
            persist_dir=persist,
            reranker=active_reranker,
            **stores,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        retrieved = [h.chunk_id for h in hits]
        relevant = set(q.relevant_chunk_ids)
        labeled_hits = [
            {
                "rank": i,
                "chunk_id": h.chunk_id,
                "score": h.score,
                "relevant": h.chunk_id in relevant,
                "document_name": h.document_name,
                "section_title": h.section_title,
            }
            for i, h in enumerate(hits, start=1)
        ]
        row = {
            "id": q.id,
            "query": q.query,
            "relevant_chunk_ids": q.relevant_chunk_ids,
            "retrieved_chunk_ids": retrieved,
            "hits": labeled_hits,
            "latency_ms": round(elapsed_ms, 2),
            "scores": {},
        }
        for k in k_values:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            h = hit_at_k(retrieved, relevant, k)
            n = ndcg_at_k(retrieved, relevant, k)
            metric_bags[f"precision@{k}"].append(p)
            metric_bags[f"recall@{k}"].append(r)
            metric_bags[f"hit@{k}"].append(h)
            metric_bags[f"ndcg@{k}"].append(n)
            row["scores"][f"precision@{k}"] = p
            row["scores"][f"recall@{k}"] = r
            row["scores"][f"hit@{k}"] = h
            row["scores"][f"ndcg@{k}"] = n
        mrr = reciprocal_rank(retrieved, relevant)
        mrr_values.append(mrr)
        row["scores"]["mrr"] = mrr
        per_query.append(row)

    run_id = uuid.uuid4().hex[:12]
    combo = dict(
        job_report.get("combo")
        or {
            "parser_engine": job_report.get("parser_engine"),
            "parser_profile": job_report.get("parser_profile"),
            "chunk": {"strategy_id": job_report.get("strategy_id")},
            "embed": {"model_id": model_id},
            "vector_store": job_report.get("vector_store", "chroma"),
        }
    )
    # Query-time retrieval knobs overwrite ingest stub so leaderboard reflects the run
    combo["retrieval"] = retrieval_cfg.to_combo_dict()

    metrics = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job_report.get("job_id"),
        "collection_name": collection,
        "eval_set_id": eval_set_id,
        "k_values": k_values,
        "n_questions": len(questions),
        "combo": combo,
        "aggregates": {
            **{name: round(aggregate_mean(vals), 4) for name, vals in metric_bags.items()},
            "mrr": round(aggregate_mean(mrr_values), 4),
            "latency_ms_p50": round(percentile(latencies_ms, 50), 2),
            "latency_ms_p95": round(percentile(latencies_ms, 95), 2),
        },
    }

    evals_dir = Path(evals_dir)
    run_dir = evals_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (run_dir / "per_query.jsonl").open("w", encoding="utf-8") as fh:
        for row in per_query:
            fh.write(json.dumps(row) + "\n")
    _append_leaderboard(evals_dir / "leaderboard.csv", metrics)
    return EvalRunResult(run_id=run_id, metrics=metrics, per_query=per_query, run_dir=run_dir)


def _leaderboard_row(metrics: dict) -> dict:
    combo = metrics.get("combo") or {}
    chunk = combo.get("chunk") or {}
    embed = combo.get("embed") or {}
    retrieval = combo.get("retrieval") or {}
    aggregates = metrics.get("aggregates") or {}
    return {
        "run_id": metrics.get("run_id"),
        "timestamp": metrics.get("timestamp"),
        "job_id": metrics.get("job_id"),
        "collection_name": metrics.get("collection_name"),
        "eval_set_id": metrics.get("eval_set_id"),
        "n_questions": metrics.get("n_questions"),
        "parser_engine": combo.get("parser_engine"),
        "parser_profile": combo.get("parser_profile"),
        "chunk_strategy": chunk.get("strategy_id"),
        "embed_model_id": embed.get("model_id"),
        "vector_store": combo.get("vector_store"),
        "retrieval_mode": retrieval.get("mode"),
        "keyword_method": retrieval.get("keyword_method"),
        "semantic_weight": retrieval.get("semantic_weight"),
        "rerank": retrieval.get("rerank"),
        "sibling_fill": retrieval.get("sibling_fill"),
        "parent_child": retrieval.get("parent_child"),
        "prefix_section_title": chunk.get("prefix_section_title"),
        "precision@1": aggregates.get("precision@1"),
        "precision@3": aggregates.get("precision@3"),
        "precision@5": aggregates.get("precision@5"),
        "precision@10": aggregates.get("precision@10"),
        "recall@1": aggregates.get("recall@1"),
        "recall@3": aggregates.get("recall@3"),
        "recall@5": aggregates.get("recall@5"),
        "recall@10": aggregates.get("recall@10"),
        "hit@1": aggregates.get("hit@1"),
        "hit@3": aggregates.get("hit@3"),
        "hit@5": aggregates.get("hit@5"),
        "hit@10": aggregates.get("hit@10"),
        "ndcg@1": aggregates.get("ndcg@1"),
        "ndcg@3": aggregates.get("ndcg@3"),
        "ndcg@5": aggregates.get("ndcg@5"),
        "ndcg@10": aggregates.get("ndcg@10"),
        "mrr": aggregates.get("mrr"),
        "latency_ms_p50": aggregates.get("latency_ms_p50"),
        "latency_ms_p95": aggregates.get("latency_ms_p95"),
    }


def _append_leaderboard(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _leaderboard_row(metrics)
    fieldnames = list(row.keys())

    existing: list[dict] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            old_fields = list(reader.fieldnames or [])
            existing = list(reader)
        # Schema expanded (e.g. hybrid knobs) — rewrite so DictReader stays aligned
        if old_fields != fieldnames:
            normalized = [{k: r.get(k) for k in fieldnames} for r in existing]
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(normalized)
                writer.writerow(row)
            return

    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_leaderboard(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Streamlit Arrow/JSON cannot serialize NaN; empty CSV cells → "" → keep as None
    cleaned: list[dict] = []
    for row in rows:
        cleaned.append({k: (None if v == "" or v is None else v) for k, v in row.items()})
    return cleaned

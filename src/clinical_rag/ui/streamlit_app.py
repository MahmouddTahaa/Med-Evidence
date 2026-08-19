"""Operator lab: ingest + retrieval eval for the clinician/student pharma guide.

Clinician query/generation: `clinical_rag.ui.clinician_app` (port 8502).
Product retrieval is `clinical_rag.query.RetrievalSession` on the frozen stack.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from clinical_rag.adapters.embedders import embedder_for_index
from clinical_rag.config import get_settings, make_doc_id, make_job_id, resolve_embed_config, store_kwargs
from clinical_rag.errors import IngestError
from clinical_rag.eval.freeze import freeze_combo
from clinical_rag.stack import (
    WINNING_PATH,
    retrieval_config_from_mapping,
    try_load_frozen_stack,
)
from clinical_rag.eval.grid import retrieval_eval_grid, retrieval_label
from clinical_rag.eval.label import label_from_chunks
from clinical_rag.eval.protocol import (
    OPENAI_EMBED_MODEL_ID,
    STAGE2_EMBEDS,
    STAGE3_CHUNKS,
    TEMPLATES_PATH,
    score_key,
)
from clinical_rag.eval.runner import load_leaderboard, load_questions, run_retrieval_eval
from clinical_rag.parsing.router import media_type_for
from clinical_rag.pipeline.ingest import run_ingest
from clinical_rag.retrieval.pipeline import run_retrieve
from clinical_rag.retrieval.sparse import SparseIndex
from clinical_rag.schemas import (
    ChromaConfig,
    ChunkConfig,
    EmbedConfig,
    EmbedProvider,
    IngestJobConfig,
    KeywordMethod,
    LegalFlags,
    ParserConfig,
    ParserEngine,
    ParserProfile,
    QdrantConfig,
    RawDocument,
    RetrievalConfig,
    RetrievalMode,
    StrategyId,
    VectorStoreKind,
)

EVALS_DIR = Path("artifacts/evals")
LOCK_STATE_PATH = Path("artifacts/lock_winning/state.json")
_STACK = try_load_frozen_stack()
DEFAULT_TEMPLATES = TEMPLATES_PATH
CHUNK_STRATEGIES = [s.value for s in STAGE3_CHUNKS]
EMBED_MODELS = [c.model_id for c in STAGE2_EMBEDS]
# Ranking metrics first so best-at-top is obvious; order matches score_key.
TABLE_COLS = [
    "run_id",
    "mrr",
    "ndcg@5",
    "precision@5",
    "precision@3",
    "precision@1",
    "recall@5",
    "hit@5",
    "latency_ms_p50",
    "chunk_strategy",
    "embed_model_id",
    "vector_store",
    "retrieval_mode",
    "keyword_method",
    "rerank",
    "sibling_fill",
    "parent_child",
    "prefix_section_title",
    "n_questions",
    "eval_set_id",
    "timestamp",
]
_SCORE_COLS = frozenset(
    {
        "mrr",
        "ndcg@5",
        "precision@5",
        "precision@3",
        "precision@1",
        "recall@5",
        "hit@5",
        "latency_ms_p50",
        "n_questions",
    }
)

st.set_page_config(
    page_title="Med-Evidence pharma guide lab",
    layout="wide",
    menu_items={
        "About": (
            "Operator lab for choosing the retrieval stack that will power the "
            "Operator lab for the locked retrieval stack. Clinician query/generation "
            "uses clinical_rag.query.RetrievalSession, not this app."
        )
    },
)

get_settings.cache_clear()
settings = get_settings()
JOBS = Path(settings.jobs_dir)
UPLOADS = Path(settings.uploads_dir)


def _embed_provider(model_id: str) -> EmbedProvider:
    if model_id == OPENAI_EMBED_MODEL_ID:
        return EmbedProvider.openai
    return EmbedProvider.sentence_transformers


def _jobs() -> list[Path]:
    """Ingest jobs with a finished report, newest report first."""
    if not JOBS.exists():
        return []
    jobs = [p for p in JOBS.iterdir() if p.is_dir() and (p / "report.json").is_file()]
    return sorted(jobs, key=lambda p: (p / "report.json").stat().st_mtime, reverse=True)


def _incomplete_jobs() -> list[Path]:
    """Job dirs that never finished (no report.json) — not queryable."""
    if not JOBS.exists():
        return []
    return sorted(
        (p for p in JOBS.iterdir() if p.is_dir() and not (p / "report.json").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _load_report(job_dir: Path) -> dict:
    return json.loads((job_dir / "report.json").read_text(encoding="utf-8"))


def _job_label(job_dir: Path) -> str:
    report = _load_report(job_dir)
    return (
        f"{report.get('job_id')} · {report.get('strategy_id')} · "
        f"{report.get('embed_model_id')} · {report.get('vector_store')} · "
        f"{report.get('chunk_count')} chunks"
    )


def _filter_jobs(jobs: list[Path], needle: str) -> list[Path]:
    q = (needle or "").strip().lower()
    if not q:
        return jobs
    out: list[Path] = []
    for job_dir in jobs:
        label = _job_label(job_dir).lower()
        if q in label or q in job_dir.name.lower():
            out.append(job_dir)
    return out


def _legal() -> LegalFlags:
    st.caption("Legal checklist must be complete before ingest (fail closed).")
    cols = st.columns(2)
    flags = {}
    keys = (
        "open_access_or_reusable",
        "redistribution_ok_for_indexing",
        "edition_current",
        "attribution_documented",
    )
    labels = {
        "open_access_or_reusable": "Open access / reusable for this project",
        "redistribution_ok_for_indexing": "Indexing / vector store allowed",
        "edition_current": "Current edition",
        "attribution_documented": "Attribution recorded (filename / source)",
    }
    for i, key in enumerate(keys):
        with cols[i % 2]:
            flags[key] = st.checkbox(labels[key], key=f"legal-{key}")
    return LegalFlags(**flags)


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _leaderboard_view(rows: list[dict]) -> list[dict]:
    """Best first by protocol score_key: (MRR, nDCG@5, Precision@5). Coerce metrics to float."""
    view: list[dict] = []
    for row in rows:
        out: dict = {}
        for col in TABLE_COLS:
            val = row.get(col)
            if col in _SCORE_COLS and val is not None and val != "":
                out[col] = _to_float(val)
            else:
                out[col] = val
        view.append(out)
    return sorted(view, key=score_key, reverse=True)


def _persist_dir_for(vector_store: str) -> str:
    if vector_store == VectorStoreKind.qdrant.value:
        return str(settings.qdrant.persist_dir)
    return str(settings.chroma.persist_dir)


def page_ingest() -> None:
    st.header("1. Ingest")
    st.write(
        "Batch-parse any supported file, chunk, embed, and write a vector index. "
        "Parser is a **router**: NXML/XML, digital PDF (PyMuPDF), scanned PDF (OCR), TXT, MD. "
        "Use non-pharma files to prove the router; official freeze uses StatPearls only."
    )
    corpus_id = st.text_input("corpus_id", value="statpearls_pharmacology")
    uploaded = st.file_uploader(
        "Upload files (multiple)",
        type=["pdf", "txt", "md", "xml", "nxml", "json"],
        accept_multiple_files=True,
    )
    local_dir = st.text_input(
        "Or index a local directory (all supported files)",
        value="",
        placeholder="data/pharmacology_data/statpearls_NBK430685",
    )
    st.caption(
        "StatPearls dump (~9.6k NXML) is at `data/pharmacology_data/statpearls_NBK430685`. "
        "Paste that path for a full index, or a smaller subfolder for a first run."
    )
    store_options = [VectorStoreKind.chroma.value, VectorStoreKind.qdrant.value]
    default_strategy = (
        _STACK.chunk.strategy_id.value
        if _STACK and _STACK.chunk.strategy_id.value in CHUNK_STRATEGIES
        else CHUNK_STRATEGIES[0]
    )
    default_embed = (
        _STACK.embed.model_id
        if _STACK and _STACK.embed.model_id in EMBED_MODELS
        else EMBED_MODELS[0]
    )
    default_store = (
        _STACK.vector_store.value
        if _STACK and _STACK.vector_store.value in store_options
        else store_options[0]
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        strategy = st.selectbox(
            "Chunk strategy",
            CHUNK_STRATEGIES,
            index=CHUNK_STRATEGIES.index(default_strategy),
        )
    with c2:
        embed_model = st.selectbox(
            "Embedding model",
            EMBED_MODELS,
            index=EMBED_MODELS.index(default_embed),
        )
    with c3:
        store = st.selectbox(
            "Vector store",
            store_options,
            index=store_options.index(default_store),
        )
    st.caption(
        "Defaults follow `configs/winning.yaml` when present (locked product stack, "
        "including `prefix_section_title`). Chunk size: 400 tokens, 0.12 overlap. "
        f"`{OPENAI_EMBED_MODEL_ID}` requires `OPENAI_API_KEY` (fail closed if missing)."
    )
    legal = _legal()

    if st.button("Run ingest", type="primary"):
        UPLOADS.mkdir(parents=True, exist_ok=True)
        files: list[RawDocument] = []
        job_id = make_job_id()
        for item in uploaded or []:
            dest = UPLOADS / item.name
            dest.write_bytes(item.getvalue())
            files.append(
                RawDocument(
                    doc_id=make_doc_id(item.name),
                    filename=item.name,
                    media_type=media_type_for(item.name),
                    document_name=Path(item.name).stem,
                    path=str(dest),
                    legal=legal,
                )
            )
        if local_dir.strip():
            root = Path(local_dir.strip())
            if not root.is_dir():
                st.error(f"Not a directory: {root}")
                return
            for path in sorted(root.rglob("*")):
                if path.suffix.lower() not in {".pdf", ".txt", ".md", ".xml", ".nxml", ".json"}:
                    continue
                files.append(
                    RawDocument(
                        doc_id=make_doc_id(path.name),
                        filename=path.name,
                        media_type=media_type_for(path.name),
                        document_name=path.stem,
                        path=str(path),
                        legal=legal,
                    )
                )
        if not files:
            st.error("No files selected.")
            return
        provider = _embed_provider(embed_model)
        embed_cfg, warnings = resolve_embed_config(
            EmbedConfig(
                model_id=embed_model,
                provider=provider,
                fallback_model_id="BAAI/bge-small-en-v1.5"
                if provider is EmbedProvider.sentence_transformers
                else "",
            )
        )
        if _STACK:
            chunk_cfg = _STACK.chunk.model_copy(update={"strategy_id": StrategyId(strategy)})
            parser_cfg = _STACK.parser()
        else:
            chunk_cfg = ChunkConfig(strategy_id=StrategyId(strategy))
            parser_cfg = ParserConfig(
                engine=ParserEngine.pymupdf, profile=ParserProfile.ocr_fallback
            )
        config = IngestJobConfig(
            corpus_id=corpus_id,
            job_id=job_id,
            files=files,
            parser=parser_cfg,
            chunk=chunk_cfg,
            embed=embed_cfg,
            vector_store=VectorStoreKind(store),
            chroma=ChromaConfig(persist_dir=str(settings.chroma.persist_dir)),
            qdrant=QdrantConfig(persist_dir=str(settings.qdrant.persist_dir)),
        )
        bar = st.progress(0, text="Starting…")

        def progress(stage: str, message: str) -> None:
            bar.progress(min(100, hash(stage) % 90 + 10), text=f"{stage}: {message}")

        try:
            report, chunks = run_ingest(config, progress=progress)
        except IngestError as exc:
            st.error(str(exc))
            return
        bar.progress(100, text="Done")
        for w in warnings + report.warnings:
            st.warning(w)
        st.success(
            f"Job `{report.job_id}` · {len(chunks)} chunks · `{report.collection_name}` · {report.vector_store}"
        )


def _load_eval_questions(job_dir: Path, source: str) -> list:
    if source == "gold jsonl":
        uploaded = st.session_state.get("gold_upload")
        if uploaded is None:
            raise IngestError("Upload a gold JSONL with relevant_chunk_ids.")
        path = Path(settings.uploads_dir) / uploaded.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(uploaded.getvalue())
        return load_questions(path)
    templates_path = Path(st.session_state.get("templates_path") or DEFAULT_TEMPLATES)
    templates = [
        json.loads(line)
        for line in templates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = json.loads((job_dir / "chunks.json").read_text(encoding="utf-8"))
    return label_from_chunks(templates, chunks)


def page_eval() -> None:
    st.header("2. Retrieval evaluation")
    st.write(
        "Each **run** is one retrieval configuration on one ingest job. "
        "Default gold = StatPearls pharmacology templates, auto-labeled from this job's chunks. "
        "Metrics go to `artifacts/evals/leaderboard.csv`."
    )
    st.caption(
        "Official freeze is `scripts/lock_winning_combo.py` on the StatPearls dump "
        "(sequential store → embed → chunk → retrieval). This tab is for ad-hoc jobs only."
    )
    jobs = _jobs()
    if not jobs:
        st.info("Ingest a corpus first.")
        return
    job_dir = st.selectbox("Ingest job", jobs, format_func=_job_label)
    report = _load_report(job_dir)
    persist = _persist_dir_for(str(report.get("vector_store") or "chroma"))

    c1, c2 = st.columns(2)
    with c1:
        gold_mode = st.radio(
            "Gold set",
            ["StatPearls templates (auto-label from this job's chunks)", "gold jsonl"],
            index=0,
        )
        st.session_state["templates_path"] = str(
            st.text_input("Templates JSONL", value=str(DEFAULT_TEMPLATES))
        )
    with c2:
        st.session_state["gold_upload"] = st.file_uploader(
            "Gold JSONL (id, query, relevant_chunk_ids)",
            type=["jsonl", "json"],
            key="gold-file",
        )
        k_values = st.multiselect("k", [1, 3, 5, 10], default=[1, 3, 5])

    run_grid = st.checkbox("Run full retrieval grid (dense, sparse, hybrid, hybrid+rerank × BM25/TF-IDF)", value=True)
    if not run_grid:
        mode = st.selectbox("mode", [m.value for m in RetrievalMode])
        kw = st.selectbox("sparse method", [m.value for m in KeywordMethod])
        rerank = st.checkbox("rerank (hybrid only)", value=False)
        grid = [
            RetrievalConfig(
                mode=RetrievalMode(mode),
                keyword_method=KeywordMethod(kw),
                rerank=rerank,
                top_k=max(k_values or [5]),
                fetch_k=20,
            )
        ]
    else:
        grid = retrieval_eval_grid(top_k=max(k_values or [10]))

    if st.button("Run retrieval eval", type="primary"):
        try:
            questions = _load_eval_questions(
                job_dir,
                "gold jsonl" if gold_mode.startswith("gold") else "templates",
            )
        except (IngestError, OSError, json.JSONDecodeError) as exc:
            st.error(str(exc))
            return
        progress = st.progress(0, text="Evaluating…")
        results = []
        for i, cfg in enumerate(grid, start=1):
            progress.progress(int((i - 1) / len(grid) * 100), text=f"{retrieval_label(cfg)} ({i}/{len(grid)})")
            try:
                result = run_retrieval_eval(
                    job_report=report,
                    questions=questions,
                    persist_dir=str(persist),
                    k_values=k_values or [3, 5],
                    evals_dir=EVALS_DIR,
                    eval_set_id="statpearls_pharmacology",
                    retrieval=cfg,
                    jobs_dir=JOBS,
                )
            except Exception as exc:
                st.error(f"{retrieval_label(cfg)} failed: {exc}")
                continue
            results.append(result)
        progress.progress(100, text="Done")
        st.success(f"{len(results)} eval runs written.")

    st.subheader("Retrieval metrics by configuration")
    st.caption(
        "All runs in `artifacts/evals/leaderboard.csv` (append-only history). "
        "Sorted best-first by protocol score key: MRR → nDCG@5 → Precision@5."
    )
    rows = load_leaderboard(EVALS_DIR / "leaderboard.csv")
    if not rows:
        st.info("No runs yet.")
        return
    view = _leaderboard_view(rows)
    st.dataframe(view, use_container_width=True, hide_index=True)

    best = view[0] if view else None
    if best:
        st.metric(
            "Best run (MRR)",
            f"{_to_float(best.get('mrr')):.4f}",
            help=(
                f"{best.get('run_id')} · nDCG@5={_to_float(best.get('ndcg@5')):.4f} · "
                f"P@5={_to_float(best.get('precision@5')):.4f}"
            ),
        )

    run_ids = [r.get("run_id") for r in view if r.get("run_id")]
    # Prefer the frozen winner in the inspect picker when present
    default_pick = 0
    if WINNING_PATH.is_file() and LOCK_STATE_PATH.is_file():
        try:
            win_run = (json.loads(LOCK_STATE_PATH.read_text(encoding="utf-8")).get("winner") or {}).get(
                "run_id"
            )
        except (OSError, json.JSONDecodeError):
            win_run = None
        if win_run and win_run in run_ids:
            default_pick = run_ids.index(win_run)

    pick = st.selectbox(
        "Inspect / freeze run",
        run_ids,
        index=default_pick if run_ids else 0,
        help="Frozen winner is selected by default when state.json records it.",
    )
    if pick:
        metrics_path = EVALS_DIR / pick / "metrics.json"
        per_query_path = EVALS_DIR / pick / "per_query.jsonl"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.json(metrics.get("aggregates") or {})
            combo = metrics.get("combo") or {}
            if st.button("Freeze this combo → configs/winning.yaml"):
                out = freeze_combo(combo, WINNING_PATH)
                st.success(f"Wrote {out}. Product retrieval reads this via FrozenStack.")
        if per_query_path.is_file():
            with st.expander("Per-query breakdown"):
                pq = [
                    json.loads(line)
                    for line in per_query_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                st.dataframe(
                    [
                        {
                            "id": r.get("id"),
                            "query": r.get("query"),
                            "mrr": (r.get("scores") or {}).get("mrr"),
                            "precision@5": (r.get("scores") or {}).get("precision@5"),
                            "recall@5": (r.get("scores") or {}).get("recall@5"),
                            "hit@5": (r.get("scores") or {}).get("hit@5"),
                            "latency_ms": r.get("latency_ms"),
                        }
                        for r in pq
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


def page_freeze() -> None:
    st.header("3. Frozen stack")
    st.write(
        "Production lock comes from `scripts/lock_winning_combo.py` on the StatPearls eval corpus. "
        "The clinician/student query app is **not built in this slice** — it will load "
        "`configs/winning.yaml` later."
    )
    if WINNING_PATH.is_file():
        st.code(WINNING_PATH.read_text(encoding="utf-8"), language="yaml")
    else:
        st.info("No frozen combo yet. Run the lock script (or freeze a row from eval).")


def _winning_job_id() -> str | None:
    if not LOCK_STATE_PATH.is_file():
        return None
    try:
        state = json.loads(LOCK_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    job_id = state.get("winning_job_id")
    return str(job_id) if job_id else None


def _flag_of(obj, name: str, default: bool = False) -> bool:
    """Read a bool flag from a dict or config object; tolerate missing fields."""
    if isinstance(obj, dict):
        if name not in obj:
            return default
        return bool(obj.get(name))
    return bool(getattr(obj, name, default))


def _retrieval_from_dict(raw: dict | None, fallback: RetrievalConfig | None = None) -> RetrievalConfig:
    return retrieval_config_from_mapping(raw, fallback=fallback or settings.retrieval)


def _retrieval_from_winning() -> RetrievalConfig:
    """Load retrieval knobs from winning.yaml; fall back to settings."""
    if _STACK is not None:
        return _STACK.retrieval.model_copy()
    return _retrieval_from_dict({})


def _eval_run_dirs() -> list[Path]:
    if not EVALS_DIR.exists():
        return []
    return sorted(
        (p for p in EVALS_DIR.iterdir() if p.is_dir() and (p / "metrics.json").is_file()),
        key=lambda p: (p / "metrics.json").stat().st_mtime,
        reverse=True,
    )


def _eval_run_label(run_dir: Path) -> str:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    combo = metrics.get("combo") or {}
    chunk = combo.get("chunk") or {}
    embed = combo.get("embed") or {}
    retrieval = combo.get("retrieval") or {}
    mode = retrieval.get("mode") or "?"
    rerank = retrieval.get("rerank")
    kw = retrieval.get("keyword_method") or ""
    label = mode
    if mode == "hybrid":
        label = f"hybrid/{kw}" + ("+rerank" if rerank else "")
    elif mode == "keyword":
        label = f"sparse/{kw}"
    if retrieval.get("sibling_fill"):
        label = f"{label}+siblings"
    if retrieval.get("parent_child"):
        label = f"{label}+parent_child"
    if chunk.get("prefix_section_title"):
        label = f"{label}+titleprefix"
    agg = metrics.get("aggregates") or {}
    mrr = agg.get("mrr")
    mrr_s = f"{float(mrr):.3f}" if mrr is not None else "?"
    return (
        f"{run_dir.name} · {label} · job={metrics.get('job_id')} · "
        f"{chunk.get('strategy_id')} · {embed.get('model_id')} · MRR={mrr_s}"
    )


def _resolve_job_dir(job_id: str) -> Path | None:
    if not job_id:
        return None
    path = JOBS / job_id
    if path.is_dir() and (path / "report.json").is_file():
        return path
    return None


def page_query() -> None:
    st.header("4. Try a query")
    st.write(
        "Retrieve ranked chunks with scores. "
        "**Ingest job ids** live under `artifacts/jobs/`. "
        "**Leaderboard run ids** (e.g. `7d630b335113`) live under `artifacts/evals/` "
        "and point at a job + retrieval combo. "
        "Product UI should call `RetrievalSession`, not this tab."
    )
    jobs = _jobs()
    incomplete = _incomplete_jobs()
    eval_runs = _eval_run_dirs()
    if not jobs:
        st.info("Ingest a corpus first (or run the lock script).")
        if incomplete:
            st.caption(f"{len(incomplete)} incomplete job dir(s) have no report.json yet.")
        return

    source = st.radio(
        "Index source",
        ["Ingest job", "Eval run (leaderboard)"],
        horizontal=True,
        key="query-source",
        help="Eval run ids from the leaderboard are not ingest job ids.",
    )

    retrieval_override: dict | None = None
    job_dir: Path | None = None

    if source == "Eval run (leaderboard)":
        if not eval_runs:
            st.warning("No eval runs in artifacts/evals/.")
            return
        st.caption(f"{len(eval_runs)} eval run(s) · newest first")
        run_filter = st.text_input(
            "Filter eval runs",
            value="",
            placeholder="e.g. 7d630b335113, hybrid, section_aware…",
            key="query-eval-filter",
        )
        needle = (run_filter or "").strip().lower()
        filtered_runs = eval_runs
        if needle:
            filtered_runs = []
            for run_dir in eval_runs:
                try:
                    label = _eval_run_label(run_dir).lower()
                except (OSError, json.JSONDecodeError):
                    label = run_dir.name.lower()
                if needle in label or needle in run_dir.name.lower():
                    filtered_runs.append(run_dir)
        if not filtered_runs:
            st.warning("No eval runs match that filter.")
            return

        default_run = 0
        # Prefer frozen winner run if present
        winner_run = None
        if LOCK_STATE_PATH.is_file():
            try:
                winner_run = (json.loads(LOCK_STATE_PATH.read_text(encoding="utf-8")).get("winner") or {}).get(
                    "run_id"
                )
            except (OSError, json.JSONDecodeError):
                winner_run = None
        if winner_run:
            for i, run_dir in enumerate(filtered_runs):
                if run_dir.name == str(winner_run):
                    default_run = i
                    break

        run_dir = st.selectbox(
            "Eval run",
            filtered_runs,
            index=min(default_run, len(filtered_runs) - 1),
            format_func=_eval_run_label,
            key="query-eval-run",
        )
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        job_id = str(metrics.get("job_id") or "")
        job_dir = _resolve_job_dir(job_id)
        if job_dir is None:
            st.error(
                f"Eval run `{run_dir.name}` points at job `{job_id}`, "
                f"but `artifacts/jobs/{job_id}/report.json` is missing."
            )
            return
        retrieval_override = (metrics.get("combo") or {}).get("retrieval") or {}
        st.info(
            f"Eval `{run_dir.name}` → ingest job `{job_id}` · "
            f"collection `{metrics.get('collection_name')}`"
        )
    else:
        st.caption(
            f"{len(jobs)} finished ingest job(s)"
            + (f" · {len(incomplete)} incomplete (no report)" if incomplete else "")
            + " · sorted newest first"
        )
        filter_text = st.text_input(
            "Filter jobs",
            value="",
            placeholder="e.g. section_aware, bge-small, cd3337fa4762…",
            key="query-job-filter",
        )
        filtered = _filter_jobs(jobs, filter_text)
        if not filtered:
            st.warning("No ingest jobs match that filter.")
            if incomplete:
                with st.expander(f"Incomplete jobs ({len(incomplete)})"):
                    st.code("\n".join(p.name for p in incomplete))
            return

        win_job = _winning_job_id()
        default_idx = 0
        if win_job:
            for i, jdir in enumerate(filtered):
                if jdir.name == win_job:
                    default_idx = i
                    break

        job_dir = st.selectbox(
            "Ingest job (index)",
            filtered,
            index=min(default_idx, len(filtered) - 1),
            format_func=_job_label,
            key="query-ingest-job",
            help="Leaderboard run ids are under 'Eval run' — not listed here.",
        )
        if incomplete:
            with st.expander(f"Incomplete jobs not listed ({len(incomplete)})"):
                st.caption("These directories have no report.json (ingest did not finish).")
                st.code("\n".join(p.name for p in incomplete))

    assert job_dir is not None
    report = _load_report(job_dir)
    chunks_path = job_dir / "chunks.json"
    persist = _persist_dir_for(str(report.get("vector_store") or "chroma"))

    base = (
        _retrieval_from_dict(retrieval_override)
        if retrieval_override is not None
        else _retrieval_from_winning()
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode_options = [m.value for m in RetrievalMode]
        mode = st.selectbox(
            "mode",
            mode_options,
            index=mode_options.index(base.mode.value),
            key="query-mode",
        )
    with c2:
        top_k = st.number_input(
            "top_k", min_value=1, max_value=50, value=int(base.top_k), key="query-topk"
        )
    with c3:
        kw_options = [m.value for m in KeywordMethod]
        kw = st.selectbox(
            "sparse method",
            kw_options,
            index=kw_options.index(base.keyword_method.value),
            key="query-kw",
        )
    with c4:
        rerank = st.checkbox(
            "rerank (hybrid)",
            value=bool(base.rerank),
            disabled=mode != RetrievalMode.hybrid.value,
            key="query-rerank",
        )

    src_label = "eval run combo" if retrieval_override is not None else (
        "winning.yaml" if WINNING_PATH.is_file() else "settings"
    )
    st.caption(
        f"Index: `{report.get('collection_name')}` · "
        f"{report.get('strategy_id')} · {report.get('embed_model_id')} · "
        f"{report.get('vector_store')} · retrieval defaults from `{src_label}`"
    )

    query = st.text_area(
        "Query",
        placeholder="e.g. What are the contraindications for Ampicillin?",
        height=80,
    )
    if not st.button("Retrieve", type="primary"):
        return
    q = (query or "").strip()
    if not q:
        st.error("Enter a non-empty query.")
        return

    retrieval = RetrievalConfig(
        mode=RetrievalMode(mode),
        top_k=int(top_k),
        keyword_method=KeywordMethod(kw),
        semantic_weight=base.semantic_weight,
        keyword_weight=base.keyword_weight,
        rrf_k=base.rrf_k,
        fetch_k=max(base.fetch_k, int(top_k)),
        rerank=bool(rerank) if mode == RetrievalMode.hybrid.value else False,
        rerank_model=base.rerank_model,
        rerank_top_n=base.rerank_top_n,
        **{
            name: _flag_of(base, name)
            for name in ("sibling_fill", "parent_child")
            if name in RetrievalConfig.model_fields
        },
    )

    try:
        model_id = str(report["embed_model_id"])
        provider = report.get("embed_provider") or EmbedProvider.sentence_transformers.value
        device = report.get("embed_device") or "auto"
        collection = str(report["collection_name"])
        vector_store = report.get("vector_store") or "chroma"
        stores = store_kwargs(settings)
        if vector_store == "qdrant":
            stores["qdrant"] = settings.qdrant.model_copy(update={"persist_dir": persist})
        else:
            stores["chroma"] = ChromaConfig(persist_dir=persist)

        needs_embedder = retrieval.mode is not RetrievalMode.keyword
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
        if retrieval.mode in (RetrievalMode.keyword, RetrievalMode.hybrid):
            if not chunks_path.is_file():
                raise IngestError(f"chunks.json not found: {chunks_path}")
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            sparse_index = SparseIndex.from_chunks(chunks, method=retrieval.keyword_method)

        reranker = None
        if retrieval.mode is RetrievalMode.hybrid and retrieval.rerank:
            from clinical_rag.retrieval.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker(retrieval.rerank_model, device="cpu")

        t0 = time.perf_counter()
        hits = run_retrieve(
            query=q,
            retrieval=retrieval,
            collection=collection,
            chunks_path=chunks_path if chunks_path.is_file() else None,
            sparse_index=sparse_index,
            embedder=embedder,
            index_model_id=model_id if needs_embedder else None,
            vector_store=vector_store,
            persist_dir=persist,
            reranker=reranker,
            **stores,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    except (IngestError, OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(str(exc))
        return

    st.success(f"{len(hits)} hits · {retrieval_label(retrieval)} · {elapsed_ms:.0f} ms")
    if not hits:
        st.info("No chunks returned.")
        return

    st.dataframe(
        [
            {
                "rank": i,
                "score": h.score,
                "document_name": h.document_name,
                "section_title": h.section_title,
                "page_number": h.page_number,
                "chunk_id": h.chunk_id,
                "token_count": h.token_count,
            }
            for i, h in enumerate(hits, start=1)
        ],
        use_container_width=True,
        hide_index=True,
    )
    for i, h in enumerate(hits, start=1):
        title = f"#{i} · score={h.score:.4f} · {h.document_name} · {h.section_title}"
        with st.expander(title):
            meta = f"`{h.chunk_id}`"
            if h.page_number is not None:
                meta += f" · page {h.page_number}"
            st.markdown(meta)
            st.write(h.text)


def main() -> None:
    st.title("Med-Evidence · pharma guide retrieval lab")
    st.caption(
        "Operator lab. Retrieval is frozen in `configs/winning.yaml`. "
        "Clinician query/generation uses `clinical_rag.query.RetrievalSession`."
    )
    tab_ingest, tab_eval, tab_query, tab_freeze = st.tabs(
        ["Ingest", "Retrieval eval", "Try query", "Winning config"]
    )
    with tab_ingest:
        page_ingest()
    with tab_eval:
        page_eval()
    with tab_query:
        page_query()
    with tab_freeze:
        page_freeze()


if __name__ == "__main__":
    main()

"""Sequential freeze: store → embed → chunk → retrieval; write configs/winning.yaml.

Official bakeoff lives in clinical_rag.eval.protocol. Resume-safe under
artifacts/lock_winning/state.json. Index axes use dense-only probes; stage 4
runs the locked 7-config retrieval grid.

The 2026-08-19 product lock additionally sets chunk.prefix_section_title.
Re-running this script overwrites configs/winning.yaml with the bakeoff
winner (prefix off, sibling_fill/parent_child off). Do not run it unless
you intend to reset that product flag.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=False)

from clinical_rag.config import get_settings, make_job_id
from clinical_rag.eval.freeze import freeze_combo
from clinical_rag.eval.grid import retrieval_label
from clinical_rag.eval.label import label_from_chunks
from clinical_rag.eval.protocol import (
    EVAL_SET_ID,
    HOLD_CHUNK,
    HOLD_EMBED_MODEL_ID,
    HOLD_EMBED_PROVIDER,
    K_VALUES,
    STAGE1_STORES,
    STAGE2_EMBEDS,
    STAGE3_CHUNKS,
    EmbedCandidate,
    collect_eval_documents,
    dense_probe_config,
    load_templates,
    rank_by_score_key,
    score_key,
    should_skip_openai_embed,
    stage4_retrieval_configs,
)
from clinical_rag.eval.runner import run_retrieval_eval
from clinical_rag.pipeline.ingest import run_ingest
from clinical_rag.schemas import (
    ChunkConfig,
    EmbedConfig,
    EmbedProvider,
    IngestJobConfig,
    ParserConfig,
    RawDocument,
    RetrievalConfig,
    StrategyId,
    VectorStoreKind,
)

OUT_DIR = Path("artifacts/lock_winning")
EVALS_DIR = Path("artifacts/evals")
WINNING_PATH = Path("configs/winning.yaml")
STATE_PATH = OUT_DIR / "state.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_state(state: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def _persist_dir_for(store: VectorStoreKind) -> str:
    settings = get_settings()
    if store is VectorStoreKind.qdrant:
        return settings.qdrant.persist_dir
    return settings.chroma.persist_dir


def ingest_job(
    *,
    files: list[RawDocument],
    store: VectorStoreKind,
    strategy: StrategyId,
    embed: EmbedCandidate,
) -> dict:
    settings = get_settings()
    print(
        f"INGEST store={store.value} chunk={strategy.value} "
        f"embed={embed.model_id} · {len(files)} files",
        flush=True,
    )
    config = IngestJobConfig(
        corpus_id=EVAL_SET_ID,
        job_id=make_job_id(),
        files=files,
        parser=ParserConfig(),
        chunk=ChunkConfig(strategy_id=strategy),
        embed=EmbedConfig(
            model_id=embed.model_id,
            fallback_model_id=HOLD_EMBED_MODEL_ID,
            provider=embed.provider,
            device="cuda",
            batch_size=32 if embed.provider is EmbedProvider.sentence_transformers else 64,
        ),
        chroma=settings.chroma,
        qdrant=settings.qdrant,
        vector_store=store,
    )

    def progress(stage: str, msg: str) -> None:
        print(f"  [{stage}] {msg}", flush=True)

    report, _ = run_ingest(config, progress=progress, jobs_dir=settings.jobs_dir)
    return report.model_dump()


def questions_for_job(job_id: str) -> list:
    """Re-label gold from this job's chunks (chunk ids change with strategy)."""
    settings = get_settings()
    chunks = json.loads((Path(settings.jobs_dir) / job_id / "chunks.json").read_text(encoding="utf-8"))
    questions = label_from_chunks(load_templates(), chunks)
    sizes = [len(q.relevant_chunk_ids) for q in questions]
    print(
        f"Gold: {len(questions)} queries, "
        f"n_gold min/mean/max={min(sizes)}/{sum(sizes) / len(sizes):.1f}/{max(sizes)}",
        flush=True,
    )
    return questions


def eval_retrieval(
    job_report: dict,
    questions: list,
    cfg: RetrievalConfig,
    *,
    store: VectorStoreKind,
) -> dict:
    settings = get_settings()
    label = retrieval_label(cfg)
    print(f"EVAL {label} on job {job_report['job_id']} …", flush=True)
    result = run_retrieval_eval(
        job_report=job_report,
        questions=questions,
        persist_dir=_persist_dir_for(store),
        k_values=list(K_VALUES),
        evals_dir=EVALS_DIR,
        eval_set_id=EVAL_SET_ID,
        retrieval=cfg,
        jobs_dir=settings.jobs_dir,
    )
    agg = result.metrics["aggregates"]
    print(
        f"  → {result.run_id} {label} mrr={agg.get('mrr')} "
        f"P@1={agg.get('precision@1')} P@3={agg.get('precision@3')} "
        f"P@5={agg.get('precision@5')} R@5={agg.get('recall@5')} "
        f"Hit@5={agg.get('hit@5')} nDCG@5={agg.get('ndcg@5')} "
        f"p50={agg.get('latency_ms_p50')}ms",
        flush=True,
    )
    return {
        "run_id": result.run_id,
        "label": label,
        "aggregates": agg,
        "combo": result.metrics.get("combo"),
    }


def _ensure_dense_slot(
    *,
    state: dict,
    stage_key: str,
    slot_key: str,
    files: list[RawDocument],
    store: VectorStoreKind,
    strategy: StrategyId,
    embed: EmbedCandidate,
) -> dict:
    stage = state.setdefault(stage_key, {})
    slot = stage.setdefault(slot_key, {})
    settings = get_settings()
    dense_cfg = dense_probe_config()

    if not slot.get("job_id"):
        report = ingest_job(files=files, store=store, strategy=strategy, embed=embed)
        slot["job_id"] = report["job_id"]
        slot["chunk_count"] = report["chunk_count"]
        slot["collection_name"] = report["collection_name"]
        slot["store"] = store.value
        slot["strategy"] = strategy.value
        slot["embed_model_id"] = embed.model_id
        slot["embed_provider"] = embed.provider.value
        _save_state(state)
    else:
        print(f"skip ingest {stage_key}/{slot_key} (job {slot['job_id']})", flush=True)

    if not slot.get("dense_run_id"):
        report_path = Path(settings.jobs_dir) / slot["job_id"] / "report.json"
        job_report = json.loads(report_path.read_text(encoding="utf-8"))
        questions = questions_for_job(slot["job_id"])
        result = eval_retrieval(job_report, questions, dense_cfg, store=store)
        slot["dense_run_id"] = result["run_id"]
        slot["dense_aggregates"] = result["aggregates"]
        _save_state(state)

    return slot


def _pick_winner(slots: dict, *, label: str) -> tuple[str, dict]:
    ranked = rank_by_score_key(
        list(slots.items()),
        aggregates_of=lambda kv: (kv[1] or {}).get("dense_aggregates") or {},
    )
    if not ranked:
        raise SystemExit(f"No completed slots for {label}")
    key, slot = ranked[0]
    agg = slot.get("dense_aggregates") or {}
    print(
        f"{label} WINNER: {key} mrr={agg.get('mrr')} nDCG@5={agg.get('ndcg@5')} "
        f"P@5={agg.get('precision@5')}",
        flush=True,
    )
    return key, slot


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to this process (need NVIDIA driver + CUDA torch). "
            "Refusing to silently fall back to CPU."
        )
    print(
        f"CUDA: {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB) "
        f"torch={torch.__version__}",
        flush=True,
    )

    state = _load_state()
    files = collect_eval_documents()
    n_gold = len(load_templates())
    if "n_files" not in state:
        state["n_files"] = len(files)
        state["n_gold"] = n_gold
        state["n_distractors"] = len(files) - n_gold
        state["eval_set_id"] = EVAL_SET_ID
        state["score_key"] = ["mrr", "ndcg@5", "precision@5"]
        state["created_at"] = _now()
        _save_state(state)
        print(
            f"Corpus: {len(files)} NXML ({n_gold} gold + {len(files) - n_gold} distractors)",
            flush=True,
        )
    else:
        print(f"Resume corpus: {len(files)} NXML", flush=True)

    hold_embed = EmbedCandidate(HOLD_EMBED_PROVIDER, HOLD_EMBED_MODEL_ID)

    # --- Stage 1: store ---
    for store in STAGE1_STORES:
        _ensure_dense_slot(
            state=state,
            stage_key="stage1_store",
            slot_key=store.value,
            files=files,
            store=store,
            strategy=HOLD_CHUNK,
            embed=hold_embed,
        )
    winning_store_key, _ = _pick_winner(state["stage1_store"], label="STAGE1 STORE")
    winning_store = VectorStoreKind(winning_store_key)
    state["winning_store"] = winning_store_key
    _save_state(state)

    # --- Stage 2: embed ---
    for candidate in STAGE2_EMBEDS:
        skip = should_skip_openai_embed(candidate)
        if skip:
            print(skip, flush=True)
            state.setdefault("stage2_embed", {}).setdefault(candidate.key, {})["skipped"] = skip
            _save_state(state)
            continue
        try:
            _ensure_dense_slot(
                state=state,
                stage_key="stage2_embed",
                slot_key=candidate.key,
                files=files,
                store=winning_store,
                strategy=HOLD_CHUNK,
                embed=candidate,
            )
        except Exception as exc:
            # OpenAI (and only OpenAI) may be unavailable despite a key (quota/billing).
            # Local ST candidates must still fail closed.
            if candidate.provider is not EmbedProvider.openai:
                raise
            reason = (
                f"skip embed {candidate.model_id}: OpenAI embed failed "
                f"({type(exc).__name__}: {exc})"
            )
            print(reason, flush=True)
            slot = state.setdefault("stage2_embed", {}).setdefault(candidate.key, {})
            slot.clear()
            slot["skipped"] = reason
            _save_state(state)
    embed_slots = {
        k: v
        for k, v in state.get("stage2_embed", {}).items()
        if v.get("dense_aggregates") and not v.get("skipped")
    }
    winning_embed_key, winning_embed_slot = _pick_winner(embed_slots, label="STAGE2 EMBED")
    winning_embed = EmbedCandidate(
        EmbedProvider(winning_embed_slot["embed_provider"]),
        winning_embed_slot["embed_model_id"],
    )
    state["winning_embed"] = {
        "model_id": winning_embed.model_id,
        "provider": winning_embed.provider.value,
    }
    _save_state(state)

    # --- Stage 3: chunk ---
    for strategy in STAGE3_CHUNKS:
        _ensure_dense_slot(
            state=state,
            stage_key="stage3_chunk",
            slot_key=strategy.value,
            files=files,
            store=winning_store,
            strategy=strategy,
            embed=winning_embed,
        )
    winning_chunk_key, winning_chunk_slot = _pick_winner(
        state["stage3_chunk"], label="STAGE3 CHUNK"
    )
    state["winning_chunk"] = winning_chunk_key
    state["winning_job_id"] = winning_chunk_slot["job_id"]
    _save_state(state)

    # --- Stage 4: retrieval grid on winning index ---
    settings = get_settings()
    report_path = Path(settings.jobs_dir) / winning_chunk_slot["job_id"] / "report.json"
    job_report = json.loads(report_path.read_text(encoding="utf-8"))
    questions = questions_for_job(winning_chunk_slot["job_id"])

    retrieval_results: list[dict] = []
    for cfg in stage4_retrieval_configs():
        label = retrieval_label(cfg)
        slot = state.setdefault("stage4_retrieval", {}).setdefault(label, {})
        if slot.get("run_id"):
            print(f"skip eval {label} (run {slot['run_id']})", flush=True)
            retrieval_results.append(slot)
            continue
        result = eval_retrieval(job_report, questions, cfg, store=winning_store)
        slot.update(result)
        _save_state(state)
        retrieval_results.append(slot)

    ranked = rank_by_score_key(
        retrieval_results,
        aggregates_of=lambda r: r.get("aggregates") or {},
    )
    winner = ranked[0]
    print(
        f"STAGE4 RETRIEVAL WINNER: {winner['label']} "
        f"mrr={winner['aggregates'].get('mrr')} "
        f"nDCG@5={winner['aggregates'].get('ndcg@5')} "
        f"P@5={winner['aggregates'].get('precision@5')}",
        flush=True,
    )
    combo = winner.get("combo") or {}
    out = freeze_combo(combo, WINNING_PATH)
    state["winner"] = {
        "label": winner["label"],
        "run_id": winner["run_id"],
        "aggregates": winner["aggregates"],
        "store": winning_store_key,
        "embed": state["winning_embed"],
        "chunk": winning_chunk_key,
        "job_id": winning_chunk_slot["job_id"],
        "score_key": list(score_key(winner.get("aggregates"))),
        "frozen_at": _now(),
        "path": str(out),
    }
    _save_state(state)
    print(f"Froze {out}", flush=True)
    print(out.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()

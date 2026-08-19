"""Official StatPearls sequential freeze protocol.

Owns corpus paths, fixed sample, stage candidate lists, and the predeclared
score key. The lock script and operator UI call this; do not cartesian-sweep.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from clinical_rag.config import make_doc_id
from clinical_rag.eval.grid import retrieval_eval_grid
from clinical_rag.parsing.router import media_type_for
from clinical_rag.schemas import (
    EmbedProvider,
    LegalFlags,
    RawDocument,
    RetrievalConfig,
    RetrievalMode,
    StrategyId,
    VectorStoreKind,
)

# --- Fixed eval corpus (do not change mid-bakeoff) ---
TEMPLATES_PATH = Path("data/eval/statpearls_pharmacology_templates.jsonl")
NXML_DIR = Path("data/pharmacology_data/statpearls_NBK430685")
EVAL_SET_ID = "statpearls_pharmacology"
N_DISTRACTORS = 100
SAMPLE_SEED = 42
K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
# Resume-safe sequential freeze state (written by scripts/lock_winning_combo.py)
LOCK_STATE_PATH = Path("artifacts/lock_winning/state.json")
_STAGE_SLOTS: tuple[str, ...] = (
    "stage1_store",
    "stage2_embed",
    "stage3_chunk",
    "stage4_retrieval",
)

# Held while searching earlier axes
HOLD_CHUNK = StrategyId.section_aware
HOLD_EMBED_MODEL_ID = "BAAI/bge-small-en-v1.5"
HOLD_EMBED_PROVIDER = EmbedProvider.sentence_transformers

OPENAI_EMBED_MODEL_ID = "text-embedding-3-small"


@dataclass(frozen=True)
class EmbedCandidate:
    provider: EmbedProvider
    model_id: str

    @property
    def key(self) -> str:
        return self.model_id


# Stage 1: vary store; hold chunk=section_aware, embed=bge-small
STAGE1_STORES: tuple[VectorStoreKind, ...] = (
    VectorStoreKind.chroma,
    VectorStoreKind.qdrant,
)

# Stage 2: vary embed; hold winning store, chunk=section_aware
STAGE2_EMBEDS: tuple[EmbedCandidate, ...] = (
    EmbedCandidate(EmbedProvider.sentence_transformers, "sentence-transformers/all-MiniLM-L6-v2"),
    EmbedCandidate(EmbedProvider.sentence_transformers, "sentence-transformers/all-mpnet-base-v2"),
    EmbedCandidate(EmbedProvider.sentence_transformers, HOLD_EMBED_MODEL_ID),
    EmbedCandidate(EmbedProvider.openai, OPENAI_EMBED_MODEL_ID),
)

# Stage 3: all seven chunk strategies; hold winning store + embed
STAGE3_CHUNKS: tuple[StrategyId, ...] = (
    StrategyId.section_aware,
    StrategyId.fixed,
    StrategyId.hierarchical,
    StrategyId.langchain_recursive,
    StrategyId.langchain_token,
    StrategyId.langchain_markdown,
    StrategyId.semantic,
)


def score_key(aggregates: dict | None) -> tuple[float, float, float]:
    """Predeclared winner rule: (MRR, nDCG@5, Precision@5) descending."""
    agg = aggregates or {}
    return (
        float(agg.get("mrr") or 0.0),
        float(agg.get("ndcg@5") or 0.0),
        float(agg.get("precision@5") or 0.0),
    )


def rank_by_score_key(
    items: list,
    *,
    aggregates_of,
    reverse: bool = True,
) -> list:
    """Sort items by score_key(aggregates_of(item))."""
    return sorted(items, key=lambda item: score_key(aggregates_of(item)), reverse=reverse)


def dense_probe_config(*, top_k: int | None = None) -> RetrievalConfig:
    """Index-axis probe: dense only so hybrid/rerank do not confound."""
    return RetrievalConfig(mode=RetrievalMode.dense, top_k=top_k or max(K_VALUES))


def stage4_retrieval_configs(*, top_k: int | None = None) -> list[RetrievalConfig]:
    return retrieval_eval_grid(top_k=top_k or max(K_VALUES))


def openai_api_key_present() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def should_skip_openai_embed(candidate: EmbedCandidate) -> str | None:
    """Return a printed skip reason, or None if the candidate should run."""
    if candidate.provider is not EmbedProvider.openai:
        return None
    if openai_api_key_present():
        return None
    return (
        f"skip embed {candidate.model_id}: OPENAI_API_KEY not set "
        "(UI still fail-closed if this model is selected)"
    )


def load_templates(path: Path | None = None) -> list[dict]:
    templates_path = path or TEMPLATES_PATH
    return [
        json.loads(line)
        for line in templates_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def collect_eval_documents(
    *,
    nxml_dir: Path | None = None,
    templates: list[dict] | None = None,
    n_distractors: int = N_DISTRACTORS,
    sample_seed: int = SAMPLE_SEED,
) -> list[RawDocument]:
    """20 gold NXML from templates + seeded distractors from the same dump."""
    root = nxml_dir or NXML_DIR
    rows = templates if templates is not None else load_templates()
    gold_names = [str(row["filename"]) for row in rows]
    legal = LegalFlags(
        open_access_or_reusable=True,
        redistribution_ok_for_indexing=True,
        edition_current=True,
        attribution_documented=True,
    )
    gold_paths = [root / name for name in gold_names]
    missing = [str(p) for p in gold_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Gold NXML missing: {missing}")

    gold_set = set(gold_names)
    others = [p for p in sorted(root.glob("*.nxml")) if p.name not in gold_set]
    rng = random.Random(sample_seed)
    distractors = rng.sample(others, k=min(n_distractors, len(others)))
    paths = gold_paths + distractors

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


def stage1_defaults() -> dict:
    return {
        "chunk": HOLD_CHUNK,
        "embed_provider": HOLD_EMBED_PROVIDER,
        "embed_model_id": HOLD_EMBED_MODEL_ID,
        "probe": "dense",
    }


def stage2_defaults(*, store: VectorStoreKind) -> dict:
    return {
        "store": store,
        "chunk": HOLD_CHUNK,
        "probe": "dense",
    }


def stage3_defaults(*, store: VectorStoreKind, embed: EmbedCandidate) -> dict:
    return {
        "store": store,
        "embed_provider": embed.provider,
        "embed_model_id": embed.model_id,
        "probe": "dense",
    }


def freeze_run_ids(state_path: Path | str | None = None) -> set[str]:
    """Run ids recorded by the current sequential freeze (excludes skipped slots)."""
    path = Path(state_path) if state_path is not None else LOCK_STATE_PATH
    if not path.is_file():
        return set()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(state, dict):
        return set()

    ids: set[str] = set()
    for stage in _STAGE_SLOTS:
        slots = state.get(stage) or {}
        if not isinstance(slots, dict):
            continue
        for slot in slots.values():
            if not isinstance(slot, dict) or slot.get("skipped"):
                continue
            rid = slot.get("dense_run_id") or slot.get("run_id")
            if rid:
                ids.add(str(rid))
    winner = state.get("winner") or {}
    if isinstance(winner, dict) and winner.get("run_id"):
        ids.add(str(winner["run_id"]))
    return ids

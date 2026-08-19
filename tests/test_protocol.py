"""Focused tests for the sequential eval protocol (no CUDA freeze)."""

import json

from clinical_rag.eval.grid import retrieval_eval_grid
from clinical_rag.eval.protocol import (
    EVAL_SET_ID,
    HOLD_CHUNK,
    HOLD_EMBED_MODEL_ID,
    K_VALUES,
    N_DISTRACTORS,
    SAMPLE_SEED,
    STAGE1_STORES,
    STAGE2_EMBEDS,
    STAGE3_CHUNKS,
    EmbedCandidate,
    dense_probe_config,
    freeze_run_ids,
    rank_by_score_key,
    score_key,
    should_skip_openai_embed,
    stage1_defaults,
    stage2_defaults,
    stage3_defaults,
    stage4_retrieval_configs,
)
from clinical_rag.schemas import EmbedProvider, RetrievalMode, StrategyId, VectorStoreKind


def test_score_key_order():
    low = {"mrr": 0.1, "ndcg@5": 0.9, "precision@5": 0.9}
    mid = {"mrr": 0.5, "ndcg@5": 0.1, "precision@5": 0.9}
    high = {"mrr": 0.5, "ndcg@5": 0.8, "precision@5": 0.1}
    best = {"mrr": 0.5, "ndcg@5": 0.8, "precision@5": 0.9}
    assert score_key(best) > score_key(high) > score_key(mid) > score_key(low)


def test_score_key_missing_defaults_zero():
    assert score_key(None) == (0.0, 0.0, 0.0)
    assert score_key({}) == (0.0, 0.0, 0.0)


def test_rank_by_score_key():
    rows = [
        {"name": "a", "aggregates": {"mrr": 0.2, "ndcg@5": 0.9, "precision@5": 0.9}},
        {"name": "b", "aggregates": {"mrr": 0.8, "ndcg@5": 0.1, "precision@5": 0.1}},
        {"name": "c", "aggregates": {"mrr": 0.8, "ndcg@5": 0.5, "precision@5": 0.1}},
    ]
    ranked = rank_by_score_key(rows, aggregates_of=lambda r: r["aggregates"])
    assert [r["name"] for r in ranked] == ["c", "b", "a"]


def test_stage1_stores():
    assert STAGE1_STORES == (VectorStoreKind.chroma, VectorStoreKind.qdrant)
    defaults = stage1_defaults()
    assert defaults["chunk"] is HOLD_CHUNK is StrategyId.section_aware
    assert defaults["embed_model_id"] == HOLD_EMBED_MODEL_ID
    assert defaults["probe"] == "dense"


def test_stage2_embeds():
    assert len(STAGE2_EMBEDS) == 4
    ids = [c.model_id for c in STAGE2_EMBEDS]
    assert "sentence-transformers/all-MiniLM-L6-v2" in ids
    assert "sentence-transformers/all-mpnet-base-v2" in ids
    assert HOLD_EMBED_MODEL_ID in ids
    assert "text-embedding-3-small" in ids
    openai = next(c for c in STAGE2_EMBEDS if c.provider is EmbedProvider.openai)
    assert openai.model_id == "text-embedding-3-small"
    defaults = stage2_defaults(store=VectorStoreKind.chroma)
    assert defaults["chunk"] is StrategyId.section_aware
    assert defaults["store"] is VectorStoreKind.chroma


def test_stage3_chunks_seven_strategies():
    assert len(STAGE3_CHUNKS) == 7
    assert set(STAGE3_CHUNKS) == {
        StrategyId.section_aware,
        StrategyId.fixed,
        StrategyId.hierarchical,
        StrategyId.langchain_recursive,
        StrategyId.langchain_token,
        StrategyId.langchain_markdown,
        StrategyId.semantic,
    }
    emb = EmbedCandidate(EmbedProvider.sentence_transformers, HOLD_EMBED_MODEL_ID)
    defaults = stage3_defaults(store=VectorStoreKind.qdrant, embed=emb)
    assert defaults["embed_model_id"] == HOLD_EMBED_MODEL_ID
    assert defaults["probe"] == "dense"


def test_stage4_matches_retrieval_eval_grid():
    grid = stage4_retrieval_configs()
    assert len(grid) == 7
    assert [c.mode for c in grid] == [c.mode for c in retrieval_eval_grid(top_k=max(K_VALUES))]
    assert all(not c.sibling_fill for c in grid)
    assert all(not c.parent_child for c in grid)


def test_dense_probe_and_k_values():
    cfg = dense_probe_config()
    assert cfg.mode is RetrievalMode.dense
    assert cfg.top_k == 10
    assert K_VALUES == (1, 3, 5, 10)
    assert EVAL_SET_ID == "statpearls_pharmacology"
    assert N_DISTRACTORS == 100
    assert SAMPLE_SEED == 42


def test_skip_openai_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cand = EmbedCandidate(EmbedProvider.openai, "text-embedding-3-small")
    reason = should_skip_openai_embed(cand)
    assert reason is not None
    assert "OPENAI_API_KEY" in reason

    st = EmbedCandidate(EmbedProvider.sentence_transformers, HOLD_EMBED_MODEL_ID)
    assert should_skip_openai_embed(st) is None

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert should_skip_openai_embed(cand) is None


def test_freeze_run_ids_from_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "stage1_store": {"chroma": {"dense_run_id": "aaa"}},
                "stage2_embed": {
                    "bge": {"dense_run_id": "bbb"},
                    "openai": {"skipped": True, "run_id": "should-ignore"},
                },
                "stage3_chunk": {"section_aware": {"dense_run_id": "ccc"}},
                "stage4_retrieval": {"hybrid/bm25+rerank": {"run_id": "ddd"}},
                "winner": {"run_id": "ddd"},
            }
        ),
        encoding="utf-8",
    )
    assert freeze_run_ids(path) == {"aaa", "bbb", "ccc", "ddd"}
    assert freeze_run_ids(tmp_path / "missing.json") == set()

from pathlib import Path

import pytest
import yaml

from clinical_rag.errors import IngestError
from clinical_rag.schemas import (
    RetrievalConfig,
    RetrievalMode,
    StrategyId,
    VectorStoreKind,
)
from clinical_rag.stack import (
    FrozenStack,
    load_frozen_stack,
    load_serving_pointer,
    retrieval_config_from_mapping,
    try_load_frozen_stack,
)
from tests.helpers import legal_ok, raw_doc


def _stack_payload() -> dict:
    return {
        "parser_engine": "pymupdf",
        "parser_profile": "ocr_fallback",
        "chunk": {
            "strategy_id": "section_aware",
            "target_tokens": 400,
            "overlap_ratio": 0.12,
            "parent_tokens": 800,
            "child_tokens": 350,
            "max_tokens": 520,
            "prefix_section_title": True,
        },
        "embed": {
            "provider": "sentence_transformers",
            "model_id": "BAAI/bge-small-en-v1.5",
            "device": "cuda",
        },
        "vector_store": "chroma",
        "retrieval": {
            "mode": "hybrid",
            "top_k": 10,
            "keyword_method": "bm25",
            "semantic_weight": 0.7,
            "keyword_weight": 0.3,
            "rrf_k": 60,
            "fetch_k": 20,
            "rerank": True,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_top_n": 20,
            "sibling_fill": False,
            "parent_child": False,
        },
    }


def test_load_frozen_stack_roundtrip(tmp_path: Path):
    path = tmp_path / "winning.yaml"
    path.write_text(yaml.safe_dump(_stack_payload(), sort_keys=False), encoding="utf-8")
    stack = load_frozen_stack(path)
    assert stack.chunk.strategy_id is StrategyId.section_aware
    assert stack.chunk.prefix_section_title is True
    assert stack.vector_store is VectorStoreKind.chroma
    assert stack.retrieval.mode is RetrievalMode.hybrid
    assert stack.retrieval.rerank is True
    assert stack.retrieval.sibling_fill is False
    assert stack.embed.model_id == "BAAI/bge-small-en-v1.5"


def test_try_load_frozen_stack_missing(tmp_path: Path):
    assert try_load_frozen_stack(tmp_path / "nope.yaml") is None


def test_load_frozen_stack_missing_fails_closed(tmp_path: Path):
    with pytest.raises(IngestError, match="No frozen stack"):
        load_frozen_stack(tmp_path / "nope.yaml")


def test_serving_pointer_roundtrip(tmp_path: Path):
    path = tmp_path / "serving.yaml"
    path.write_text("job_id: ae69f99b47b7\n", encoding="utf-8")
    pointer = load_serving_pointer(path)
    assert pointer.job_id == "ae69f99b47b7"


def test_serving_pointer_missing_fails_closed(tmp_path: Path):
    with pytest.raises(IngestError, match="No serving pointer"):
        load_serving_pointer(tmp_path / "serving.yaml")


def test_ingest_config_from_stack(tmp_path: Path):
    stack = FrozenStack.model_validate(_stack_payload())
    path = tmp_path / "doc.md"
    path.write_text("# A\n\nhello\n", encoding="utf-8")
    raw = raw_doc(path)
    raw.legal = legal_ok()
    cfg = stack.ingest_config(corpus_id="pharma", job_id="job1", files=[raw])
    assert cfg.chunk.prefix_section_title is True
    assert cfg.chunk.strategy_id is StrategyId.section_aware
    assert cfg.embed.model_id == "BAAI/bge-small-en-v1.5"
    assert cfg.retrieval.mode is RetrievalMode.hybrid


def test_retrieval_config_from_mapping_overlay():
    cfg = retrieval_config_from_mapping(
        {"mode": "hybrid", "rerank": True, "semantic_weight": 0.5, "keyword_weight": 0.5},
        fallback=RetrievalConfig(),
    )
    assert cfg.mode is RetrievalMode.hybrid
    assert cfg.rerank is True
    assert cfg.semantic_weight == 0.5
    assert cfg.sibling_fill is False


def test_workspace_winning_yaml_matches_product_lock():
    from clinical_rag.stack import WINNING_PATH

    if not WINNING_PATH.is_file():
        pytest.skip("configs/winning.yaml not present")
    stack = load_frozen_stack(WINNING_PATH)
    assert stack.chunk.prefix_section_title is True
    assert stack.chunk.strategy_id is StrategyId.section_aware
    assert stack.retrieval.mode is RetrievalMode.hybrid
    assert stack.retrieval.rerank is True
    assert stack.retrieval.sibling_fill is False
    assert stack.retrieval.parent_child is False

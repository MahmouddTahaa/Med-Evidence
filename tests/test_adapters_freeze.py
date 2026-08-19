from pathlib import Path

import pytest

from clinical_rag.adapters.embedders import build_embedder
from clinical_rag.adapters.stores import build_store
from clinical_rag.errors import IngestError
from clinical_rag.eval.freeze import freeze_combo, load_selected
from clinical_rag.indexing.qdrant_store import QdrantStore
from clinical_rag.schemas import (
    ChromaConfig,
    EmbedConfig,
    EmbedProvider,
    QdrantConfig,
    VectorStoreKind,
)


def test_openai_embed_fails_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("clinical_rag.adapters.embedders._load_dotenv_keys", lambda: None)
    with pytest.raises(IngestError, match="OPENAI_API_KEY"):
        build_embedder(EmbedConfig(provider=EmbedProvider.openai, model_id="text-embedding-3-small"))


def test_qdrant_store_builds(tmp_path: Path):
    store = build_store(
        VectorStoreKind.qdrant,
        qdrant=QdrantConfig(persist_dir=str(tmp_path / "qdrant")),
    )
    assert isinstance(store, QdrantStore)


def test_chroma_store_builds():
    store = build_store(VectorStoreKind.chroma, chroma=ChromaConfig(persist_dir="artifacts/indexes/chroma"))
    assert store is not None


def test_unsupported_store_fails_closed():
    with pytest.raises(IngestError, match="unsupported store"):
        build_store("faiss")  # type: ignore[arg-type]


def test_vector_store_env(monkeypatch):
    from clinical_rag.config import Settings

    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    settings = Settings()
    assert settings.vector_store is VectorStoreKind.qdrant


def test_freeze_and_load(tmp_path: Path):
    path = tmp_path / "selected.yaml"
    combo = {
        "parser_engine": "pymupdf",
        "parser_profile": "ocr_fallback",
        "chunk": {"strategy_id": "section_aware", "target_tokens": 400, "overlap_ratio": 0.12},
        "embed": {"model_id": "BAAI/bge-small-en-v1.5"},
        "vector_store": "chroma",
        "retrieval": {
            "mode": "dense",
            "top_k": 5,
            "keyword_method": "bm25",
            "semantic_weight": 0.70,
            "keyword_weight": 0.30,
            "rrf_k": 60,
            "fetch_k": 20,
            "rerank": False,
        },
    }
    freeze_combo(combo, path)
    loaded = load_selected(path)
    assert loaded["parser_engine"] == "pymupdf"
    assert loaded["retrieval"]["mode"] == "dense"
    assert loaded["retrieval"]["keyword_method"] == "bm25"
    assert loaded["retrieval"]["semantic_weight"] == 0.70


def test_freeze_roundtrips_title_prefix(tmp_path: Path):
    path = tmp_path / "selected.yaml"
    combo = {
        "parser_engine": "pymupdf",
        "chunk": {
            "strategy_id": "section_aware",
            "target_tokens": 400,
            "overlap_ratio": 0.12,
            "prefix_section_title": True,
        },
        "embed": {"model_id": "BAAI/bge-small-en-v1.5"},
        "vector_store": "chroma",
        "retrieval": {
            "mode": "hybrid",
            "top_k": 10,
            "keyword_method": "bm25",
            "semantic_weight": 0.70,
            "keyword_weight": 0.30,
            "rerank": True,
            "sibling_fill": False,
            "parent_child": False,
        },
    }
    freeze_combo(combo, path)
    loaded = load_selected(path)
    assert loaded["chunk"]["prefix_section_title"] is True
    assert loaded["retrieval"]["sibling_fill"] is False
    assert loaded["retrieval"]["parent_child"] is False

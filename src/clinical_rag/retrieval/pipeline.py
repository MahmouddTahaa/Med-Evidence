from __future__ import annotations

from pathlib import Path
from typing import Any

from clinical_rag.adapters.stores import VectorStore, build_store
from clinical_rag.errors import IngestError
from clinical_rag.indexing.embed import Embedder
from clinical_rag.pipeline.smoke_query import run_smoke_query
from clinical_rag.retrieval.parents import expand_parent_children
from clinical_rag.retrieval.rerank import CrossEncoderReranker, Reranker
from clinical_rag.retrieval.rrf import weighted_rrf
from clinical_rag.retrieval.siblings import fill_section_siblings
from clinical_rag.retrieval.sparse import SparseIndex
from clinical_rag.schemas import (
    ChromaConfig,
    KeywordMethod,
    QdrantConfig,
    RetrievalConfig,
    RetrievalMode,
    SmokeHit,
    VectorStoreKind,
)


def load_chunks_json(path: Path | str) -> list[dict[str, Any]]:
    import json

    p = Path(path)
    if not p.is_file():
        raise IngestError(f"chunks.json not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise IngestError(f"{p}: expected a JSON array of chunks")
    return data


def _effective_mode(mode: RetrievalMode) -> RetrievalMode:
    # dense is the semantic alias used in existing job reports
    if mode is RetrievalMode.dense:
        return RetrievalMode.dense
    return mode


def run_retrieve(
    *,
    query: str,
    retrieval: RetrievalConfig,
    collection: str,
    chunks: list[dict[str, Any]] | None = None,
    chunks_path: Path | str | None = None,
    sparse_index: SparseIndex | None = None,
    embedder: Embedder | None = None,
    index_model_id: str | None = None,
    vector_store: VectorStoreKind | str = VectorStoreKind.chroma,
    persist_dir: str | None = None,
    chroma: ChromaConfig | None = None,
    qdrant: QdrantConfig | None = None,
    store: VectorStore | None = None,
    reranker: Reranker | None = None,
) -> list[SmokeHit]:
    """Query-time retrieval: semantic, keyword (BM25|TF-IDF), or hybrid (weighted RRF + optional CE)."""
    q = (query or "").strip()
    if not q:
        raise IngestError("Query must be non-empty")

    mode = _effective_mode(retrieval.mode)
    top_k = retrieval.top_k
    fetch_k = max(retrieval.fetch_k, top_k)
    need_pool = retrieval.parent_child or (
        mode is RetrievalMode.hybrid
    )
    pool_n = fetch_k if need_pool else top_k

    def dense_hits(n: int) -> list[SmokeHit]:
        if embedder is None:
            raise IngestError("Semantic/hybrid retrieval requires an embedder")
        return run_smoke_query(
            collection=collection,
            embedder=embedder,
            query=q,
            top_k=n,
            index_model_id=index_model_id,
            vector_store=vector_store,
            persist_dir=persist_dir,
            chroma=chroma,
            qdrant=qdrant,
            store=store,
        )

    def keyword_hits(n: int) -> list[SmokeHit]:
        idx = sparse_index
        if idx is None:
            rows = chunks
            if rows is None:
                if chunks_path is None:
                    raise IngestError(
                        "Keyword/hybrid retrieval requires chunks, chunks_path, or sparse_index"
                    )
                rows = load_chunks_json(chunks_path)
            idx = SparseIndex.from_chunks(rows, method=retrieval.keyword_method)
        elif idx.method is not retrieval.keyword_method:
            # Rebuild if cached index used a different method
            rows = chunks
            if rows is None:
                if chunks_path is None:
                    raise IngestError(
                        "Sparse index method mismatch; pass chunks or chunks_path to rebuild"
                    )
                rows = load_chunks_json(chunks_path)
            idx = SparseIndex.from_chunks(rows, method=retrieval.keyword_method)
        return idx.query(q, n)

    if mode is RetrievalMode.keyword:
        hits = keyword_hits(pool_n)
    elif mode is RetrievalMode.dense:
        hits = dense_hits(pool_n)
    else:
        sem = dense_hits(fetch_k)
        kw = keyword_hits(fetch_k)
        hits = weighted_rrf(
            [sem, kw],
            [retrieval.semantic_weight, retrieval.keyword_weight],
            k=retrieval.rrf_k,
        )

    rows = chunks
    if retrieval.parent_child or retrieval.sibling_fill:
        if rows is None:
            if chunks_path is None:
                raise IngestError(
                    "parent_child/sibling_fill requires chunks or chunks_path"
                )
            rows = load_chunks_json(chunks_path)

    if retrieval.parent_child:
        cap = max(retrieval.rerank_top_n, top_k, 40)
        hits = expand_parent_children(hits, rows or [], limit=cap)

    if mode is RetrievalMode.hybrid and retrieval.rerank:
        n_cand = max(retrieval.rerank_top_n, top_k)
        if retrieval.parent_child:
            n_cand = max(n_cand, len(hits))
        candidates = hits[:n_cand]
        engine = reranker or CrossEncoderReranker(retrieval.rerank_model)
        hits = engine.rerank(q, candidates)[:top_k]
    else:
        hits = hits[:top_k]

    if retrieval.sibling_fill:
        hits = fill_section_siblings(hits, rows or [], top_k=top_k)
    return hits

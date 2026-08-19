from __future__ import annotations

from clinical_rag.adapters.stores import VectorStore, build_store
from clinical_rag.errors import IngestError
from clinical_rag.indexing.embed import Embedder, assert_query_embedder_matches_index
from clinical_rag.schemas import (
    ChromaConfig,
    QdrantConfig,
    SmokeHit,
    VectorStoreKind,
)


def run_smoke_query(
    *,
    collection: str,
    embedder: Embedder,
    query: str,
    top_k: int,
    index_model_id: str | None = None,
    vector_store: VectorStoreKind | str = VectorStoreKind.chroma,
    persist_dir: str | None = None,
    chroma: ChromaConfig | None = None,
    qdrant: QdrantConfig | None = None,
    store: VectorStore | None = None,
) -> list[SmokeHit]:
    kind = (
        vector_store
        if isinstance(vector_store, VectorStoreKind)
        else VectorStoreKind(str(vector_store))
    )
    chroma_cfg = chroma or ChromaConfig(persist_dir=persist_dir or "artifacts/indexes/chroma")
    qdrant_cfg = qdrant or QdrantConfig(persist_dir=persist_dir or "artifacts/indexes/qdrant")
    if persist_dir and chroma is None and kind is VectorStoreKind.chroma:
        chroma_cfg = ChromaConfig(persist_dir=persist_dir)
    if persist_dir and qdrant is None and kind is VectorStoreKind.qdrant:
        qdrant_cfg = QdrantConfig(persist_dir=persist_dir)
    resolved_store = store or build_store(
        kind,
        chroma=chroma_cfg,
        qdrant=qdrant_cfg,
    )
    resolved_model = index_model_id or resolved_store.index_embed_model_id(collection)
    if not resolved_model:
        raise IngestError(f"Collection {collection!r} has no embed_model_id metadata")
    assert_query_embedder_matches_index(embedder, resolved_model)
    vectors = embedder.encode([query])
    return resolved_store.query(collection, vectors[0], top_k)

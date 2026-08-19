from __future__ import annotations

from typing import Protocol

from clinical_rag.errors import IngestError
from clinical_rag.indexing.chroma_store import ChromaStore
from clinical_rag.indexing.qdrant_store import QdrantStore
from clinical_rag.schemas import (
    ChromaConfig,
    Chunk,
    QdrantConfig,
    SmokeHit,
    VectorStoreKind,
)


class VectorStore(Protocol):
    def replace(self, name: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def index_embed_model_id(self, name: str) -> str | None: ...

    def query(self, name: str, embedding: list[float], top_k: int) -> list[SmokeHit]: ...


def build_store(
    kind: VectorStoreKind,
    *,
    chroma: ChromaConfig | None = None,
    qdrant: QdrantConfig | None = None,
) -> VectorStore:
    if kind is VectorStoreKind.chroma:
        return ChromaStore((chroma or ChromaConfig()).persist_dir)
    if kind is VectorStoreKind.qdrant:
        return QdrantStore((qdrant or QdrantConfig()).persist_dir)
    raise IngestError(f"unsupported store: {kind}")


__all__ = [
    "ChromaStore",
    "Chunk",
    "QdrantStore",
    "SmokeHit",
    "VectorStore",
    "build_store",
]

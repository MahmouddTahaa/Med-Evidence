from __future__ import annotations

import uuid
from pathlib import Path

from clinical_rag.errors import IngestError
from clinical_rag.indexing.payload import chunk_payload, hit_from_meta
from clinical_rag.schemas import Chunk, SmokeHit

# Stable namespace so point ids are deterministic across replace/reopen.
_CHUNK_ID_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_CHUNK_ID_NS, chunk_id))


class QdrantStore:
    """On-disk Qdrant store (``QdrantClient(path=...)``). Cosine distance."""

    UPSERT_BATCH = 256

    def __init__(self, persist_dir: str | Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _client_obj(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(path=str(self.persist_dir))
        return self._client

    def replace(self, name: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if not embeddings:
            raise IngestError("Qdrant replace requires at least one embedding")

        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = self._client_obj()
        if client.collection_exists(name):
            client.delete_collection(name)
        dim = len(embeddings[0])
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        batch = self.UPSERT_BATCH
        for start in range(0, len(chunks), batch):
            end = start + batch
            points = [
                PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vec,
                    payload=chunk_payload(chunk, include_text=True),
                )
                for chunk, vec in zip(chunks[start:end], embeddings[start:end], strict=True)
            ]
            client.upsert(collection_name=name, points=points)

    def index_embed_model_id(self, name: str) -> str | None:
        client = self._client_obj()
        if not client.collection_exists(name):
            return None
        records, _offset = client.scroll(
            collection_name=name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        model = payload.get("embed_model_id")
        return str(model) if model else None

    def query(self, name: str, embedding: list[float], top_k: int) -> list[SmokeHit]:
        client = self._client_obj()
        if not client.collection_exists(name):
            raise IngestError(f"Qdrant collection {name!r} not found at {self.persist_dir}")
        count = client.count(collection_name=name, exact=True).count
        if count == 0:
            return []
        n = min(top_k, count)
        response = client.query_points(
            collection_name=name,
            query=embedding,
            limit=n,
            with_payload=True,
        )
        hits: list[SmokeHit] = []
        for point in response.points:
            payload = dict(point.payload or {})
            hits.append(
                hit_from_meta(
                    payload,
                    text=str(payload.get("text") or ""),
                    score=float(point.score if point.score is not None else 0.0),
                )
            )
        return hits

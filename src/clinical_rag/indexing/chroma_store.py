from __future__ import annotations

from pathlib import Path

from clinical_rag.indexing.payload import chunk_payload, hit_from_meta
from clinical_rag.schemas import Chunk, SmokeHit


class ChromaStore:
    # Chroma raises if a single upsert exceeds its max batch (~5461 in current builds).
    UPSERT_BATCH = 4000

    def __init__(self, persist_dir: str | Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _client_obj(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def replace(self, name: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        client = self._client_obj()
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col = client.get_or_create_collection(
            name=name,
            metadata={
                "hnsw:space": "cosine",
                "embed_model_id": chunks[0].embed_model_id if chunks else "",
            },
        )
        batch = self.UPSERT_BATCH
        for start in range(0, len(chunks), batch):
            end = start + batch
            part = chunks[start:end]
            col.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[start:end],
                documents=[c.text for c in part],
                metadatas=[chunk_payload(c, include_text=False) for c in part],
            )

    def index_embed_model_id(self, name: str) -> str | None:
        col = self._client_obj().get_collection(name)
        collection_meta = col.metadata or {}
        model_id = collection_meta.get("embed_model_id")
        if model_id:
            return str(model_id)
        count = min(col.count(), 1)
        if count == 0:
            return None
        raw = col.get(limit=count, include=["metadatas"])
        metas = raw.get("metadatas") or []
        if metas and metas[0]:
            chunk_model = metas[0].get("embed_model_id")
            if chunk_model:
                return str(chunk_model)
        return None

    def query(self, name: str, embedding: list[float], top_k: int) -> list[SmokeHit]:
        col = self._client_obj().get_collection(name)
        count = col.count()
        n = min(top_k, max(count, 1))
        if count == 0:
            return []
        result = col.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SmokeHit] = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for text, meta, dist in zip(docs, metas, dists, strict=False):
            hits.append(
                hit_from_meta(meta or {}, text=text or "", score=1.0 - float(dist))
            )
        return hits

    def list_chunks(self, name: str, limit: int = 500) -> list[dict]:
        col = self._client_obj().get_collection(name)
        count = min(col.count(), limit)
        if count == 0:
            return []
        raw = col.get(limit=count, include=["documents", "metadatas"])
        rows = []
        for cid, text, meta in zip(raw.get("ids") or [], raw.get("documents") or [], raw.get("metadatas") or []):
            row = dict(meta or {})
            row["chunk_id"] = cid
            row["text"] = text
            rows.append(row)
        return rows

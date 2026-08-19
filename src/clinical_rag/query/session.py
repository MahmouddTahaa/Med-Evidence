"""Reusable retrieval session for the clinician query / generation path.

Loads the frozen stack once, binds it to one ingest job, and reuses the
embedder / sparse index / reranker. UI and CLI call `retrieve`; they do not
assemble store kwargs themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clinical_rag.adapters.embedders import embedder_for_index
from clinical_rag.adapters.stores import VectorStore, build_store
from clinical_rag.config import get_settings
from clinical_rag.errors import IngestError
from clinical_rag.indexing.embed import Embedder
from clinical_rag.retrieval.pipeline import run_retrieve
from clinical_rag.retrieval.rerank import CrossEncoderReranker, Reranker
from clinical_rag.retrieval.sparse import SparseIndex
from clinical_rag.schemas import (
    ChromaConfig,
    QdrantConfig,
    RetrievalMode,
    SmokeHit,
    VectorStoreKind,
)
from clinical_rag.stack import (
    SERVING_PATH,
    WINNING_PATH,
    FrozenStack,
    load_frozen_stack,
    load_serving_pointer,
)


def _persist_dir(report: dict[str, Any], stack: FrozenStack) -> str:
    combo = report.get("combo") or {}
    stored = (combo.get("store") or {}).get("persist_dir")
    if stored:
        return str(stored)
    settings = get_settings()
    if stack.vector_store is VectorStoreKind.qdrant:
        return str(settings.qdrant.persist_dir)
    return str(settings.chroma.persist_dir)


def _load_job(job_id: str, jobs_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    job_dir = Path(jobs_dir) / job_id
    report_path = job_dir / "report.json"
    chunks_path = job_dir / "chunks.json"
    if not report_path.is_file():
        raise IngestError(f"Ingest job report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise IngestError(f"{report_path}: expected a mapping")
    chunks: list[dict[str, Any]] = []
    if chunks_path.is_file():
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise IngestError(f"{chunks_path}: expected a JSON array")
        chunks = payload
    return report, chunks, chunks_path


class RetrievalSession:
    """Frozen hybrid retrieval bound to one collection."""

    def __init__(
        self,
        *,
        stack: FrozenStack,
        collection: str,
        persist_dir: str,
        chunks: list[dict[str, Any]],
        chunks_path: Path | None,
        index_model_id: str,
        embed_provider: str,
        embed_device: str,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        store: VectorStore | None = None,
        job_id: str = "",
    ) -> None:
        self.stack = stack
        self.collection = collection
        self.persist_dir = persist_dir
        self.job_id = job_id
        self._chunks = chunks
        self._chunks_path = chunks_path
        self._index_model_id = index_model_id
        self._embedder = embedder
        self._reranker = reranker
        self._store = store
        self._embed_provider = embed_provider
        self._embed_device = embed_device
        self._sparse: SparseIndex | None = None

        kind = stack.vector_store
        if store is None:
            chroma = ChromaConfig(persist_dir=persist_dir)
            qdrant = QdrantConfig(persist_dir=persist_dir)
            self._store = build_store(kind, chroma=chroma, qdrant=qdrant)

        retrieval = stack.retrieval
        needs_sparse = retrieval.mode in (RetrievalMode.keyword, RetrievalMode.hybrid)
        if needs_sparse:
            if not chunks:
                raise IngestError(
                    "Keyword/hybrid retrieval requires chunks.json on the ingest job"
                )
            self._sparse = SparseIndex.from_chunks(chunks, method=retrieval.keyword_method)

        if embedder is None and retrieval.mode is not RetrievalMode.keyword:
            self._embedder = embedder_for_index(
                model_id=index_model_id,
                provider=embed_provider,
                device=embed_device,
                batch_size=8,
                purpose="query",
            )

        if (
            reranker is None
            and retrieval.mode is RetrievalMode.hybrid
            and retrieval.rerank
        ):
            # CPU: sharing a 4 GiB GPU with the query embedder OOMs (locked freeze).
            self._reranker = CrossEncoderReranker(retrieval.rerank_model, device="cpu")

    @classmethod
    def open(
        cls,
        *,
        job_id: str | None = None,
        stack: FrozenStack | None = None,
        winning_path: Path | str = WINNING_PATH,
        serving_path: Path | str = SERVING_PATH,
        jobs_dir: Path | str | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        store: VectorStore | None = None,
    ) -> RetrievalSession:
        frozen = stack or load_frozen_stack(winning_path)
        if job_id:
            jobs_root = Path(jobs_dir or "artifacts/jobs")
            resolved_job = job_id
        else:
            pointer = load_serving_pointer(serving_path)
            jobs_root = Path(jobs_dir or pointer.jobs_dir)
            resolved_job = pointer.job_id
        report, chunks, chunks_path = _load_job(resolved_job, jobs_root)
        collection = str(report.get("collection_name") or "")
        if not collection:
            raise IngestError(f"Job {resolved_job}: report missing collection_name")
        model_id = str(report.get("embed_model_id") or frozen.embed.model_id)
        return cls(
            stack=frozen,
            collection=collection,
            persist_dir=_persist_dir(report, frozen),
            chunks=chunks,
            chunks_path=chunks_path if chunks_path.is_file() else None,
            index_model_id=model_id,
            embed_provider=str(
                report.get("embed_provider") or frozen.embed.provider.value
            ),
            embed_device=str(report.get("embed_device") or frozen.embed.device),
            embedder=embedder,
            reranker=reranker,
            store=store,
            job_id=resolved_job,
        )

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[SmokeHit]:
        retrieval = self.stack.retrieval
        if top_k is not None:
            retrieval = retrieval.model_copy(update={"top_k": top_k})
        return run_retrieve(
            query=query,
            retrieval=retrieval,
            collection=self.collection,
            chunks=self._chunks or None,
            chunks_path=self._chunks_path,
            sparse_index=self._sparse,
            embedder=self._embedder,
            index_model_id=self._index_model_id,
            vector_store=self.stack.vector_store,
            persist_dir=self.persist_dir,
            store=self._store,
            reranker=self._reranker,
        )

from __future__ import annotations

import math
from typing import Protocol

from clinical_rag.config import detect_device
from clinical_rag.errors import IngestError
from clinical_rag.schemas import DEFAULT_RERANK_MODEL, SmokeHit


class Reranker(Protocol):
    model_id: str

    def rerank(self, query: str, hits: list[SmokeHit]) -> list[SmokeHit]: ...


def sigmoid(x: float) -> float:
    """Map unbounded CE logits to (0, 1) for display / comparable scores."""
    # Numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder wrapper (fail closed on load errors).

    Defaults to CPU: sharing a 4 GiB GPU with the query embedder reliably OOMs.
    Reported scores are sigmoid(logit) in (0, 1); ranking uses the raw logit order.
    """

    def __init__(self, model_id: str = DEFAULT_RERANK_MODEL, *, device: str = "cpu"):
        self.model_id = model_id or DEFAULT_RERANK_MODEL
        # "auto" still allowed for operators with spare VRAM
        self.device = detect_device(device) if device == "auto" else device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise IngestError(
                "Cross-encoder rerank requires sentence-transformers. "
                "Install with: uv sync"
            ) from exc
        try:
            self._model = CrossEncoder(self.model_id, device=self.device)
        except Exception as exc:
            raise IngestError(
                f"Failed to load cross-encoder {self.model_id!r}: {exc}"
            ) from exc

    def rerank(self, query: str, hits: list[SmokeHit]) -> list[SmokeHit]:
        if not hits:
            return []
        self._load()
        pairs = [(query, h.text) for h in hits]
        try:
            scores = self._model.predict(pairs, batch_size=8, show_progress_bar=False)
        except Exception as exc:
            raise IngestError(f"Cross-encoder rerank failed: {exc}") from exc
        ranked = sorted(
            zip(hits, scores, strict=True),
            key=lambda pair: (-float(pair[1]), pair[0].chunk_id),
        )
        out: list[SmokeHit] = []
        for hit, score in ranked:
            out.append(
                SmokeHit(
                    score=round(sigmoid(float(score)), 6),
                    text=hit.text,
                    document_name=hit.document_name,
                    section_title=hit.section_title,
                    page_number=hit.page_number,
                    chunk_id=hit.chunk_id,
                    extraction_method=hit.extraction_method,
                    source_url=hit.source_url,
                    token_count=hit.token_count,
                )
            )
        return out

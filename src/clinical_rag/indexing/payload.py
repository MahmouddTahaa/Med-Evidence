"""Shared citation payload + SmokeHit helpers for Chroma and Qdrant."""

from __future__ import annotations

from clinical_rag.schemas import Chunk, SmokeHit


def chunk_payload(chunk: Chunk, *, include_text: bool = True) -> dict:
    """Flat citation metadata for vector-store payloads.

    Chroma keeps text in ``documents`` (pass ``include_text=False``).
    Qdrant stores text inside the payload (default).
    """
    payload: dict = {
        "chunk_id": chunk.chunk_id,
        "document_name": chunk.document_name,
        "section_title": chunk.section_title,
        "page_number": chunk.page_number if chunk.page_number is not None else -1,
        "source_url": chunk.source_url or "",
        "strategy_id": chunk.strategy_id.value,
        "corpus_id": chunk.corpus_id,
        "job_id": chunk.job_id,
        "extraction_method": chunk.extraction_method.value,
        "token_count": chunk.token_count,
        "embed_model_id": chunk.embed_model_id,
        "filename": chunk.filename,
        "doc_id": chunk.doc_id,
        "parent_chunk_id": chunk.parent_chunk_id or "",
    }
    if include_text:
        payload["text"] = chunk.text
    return payload


def hit_from_meta(meta: dict, *, text: str, score: float) -> SmokeHit:
    page = meta.get("page_number")
    return SmokeHit(
        score=round(float(score), 4),
        text=text or "",
        document_name=str(meta.get("document_name") or ""),
        section_title=str(meta.get("section_title") or ""),
        page_number=None if page in (None, -1, "-1") else int(page),
        chunk_id=str(meta.get("chunk_id") or ""),
        extraction_method=str(meta.get("extraction_method") or ""),
        source_url=str(meta.get("source_url") or ""),
        token_count=int(meta.get("token_count") or 0),
    )

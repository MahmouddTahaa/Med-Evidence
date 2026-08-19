from __future__ import annotations

from clinical_rag.errors import IngestError
from clinical_rag.schemas import (
    Chunk,
    ExtractionMethod,
    ParsedDocument,
    ParsedPage,
    RawDocument,
    StrategyId,
    TextBlock,
)


def is_prechunked_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    chunks = payload.get("chunks")
    return isinstance(chunks, list)


def load_prechunked(payload: dict, raw: RawDocument) -> tuple[list[Chunk], list[str]]:
    document_name = str(payload.get("document_name") or raw.document_name or "").strip()
    if not document_name:
        raise IngestError("Pre-chunked JSON missing document_name")
    source_url = str(payload.get("source_url") or raw.source_url or "")
    items = payload.get("chunks")
    if not isinstance(items, list) or not items:
        raise IngestError("Pre-chunked JSON has empty or missing chunks[]")

    out: list[Chunk] = []
    warnings: list[str] = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise IngestError(f"Pre-chunked item {i} is not an object")
        text = item.get("text")
        chunk_id = item.get("chunk_id")
        if not text or not str(text).strip():
            raise IngestError(f"Pre-chunked item {i} missing text")
        if not chunk_id:
            raise IngestError(f"Pre-chunked item {i} missing chunk_id")
        section = item.get("section_title")
        if not section or not str(section).strip():
            warnings.append(f"{chunk_id}: missing section_title, using (unknown)")
            section = "(unknown)"
        page = item.get("page_number")
        if page is None:
            warnings.append(f"{chunk_id}: missing page_number")
        out.append(
            Chunk(
                chunk_id=str(chunk_id),
                text=str(text),
                document_name=document_name,
                section_title=str(section),
                page_number=int(page) if page is not None else None,
                source_url=source_url,
                strategy_id=StrategyId.passthrough,
                corpus_id="",
                job_id="",
                extraction_method=ExtractionMethod.na,
                token_count=0,
                embed_model_id="",
                filename=raw.filename,
                doc_id=raw.doc_id,
            )
        )
    return out, warnings

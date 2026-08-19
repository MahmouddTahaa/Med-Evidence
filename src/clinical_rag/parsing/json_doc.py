from __future__ import annotations

from clinical_rag.errors import IngestError
from clinical_rag.schemas import (
    ExtractionMethod,
    ParsedDocument,
    ParsedPage,
    RawDocument,
    TextBlock,
)


def parse_raw_json(payload: dict, raw: RawDocument) -> ParsedDocument:
    document_name = str(payload.get("document_name") or raw.document_name or "").strip()
    source_url = str(payload.get("source_url") or raw.source_url or "")
    pages: list[ParsedPage] = []

    if isinstance(payload.get("pages"), list) and payload["pages"]:
        for i, item in enumerate(payload["pages"], 1):
            if not isinstance(item, dict):
                raise IngestError(f"pages[{i}] is not an object")
            text = str(item.get("text") or "")
            page_number = int(item.get("page_number") or i)
            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=text,
                    extraction_method=ExtractionMethod.na,
                    blocks=[TextBlock(kind="paragraph", text=text)] if text.strip() else [],
                )
            )
    elif isinstance(payload.get("sections"), list) and payload["sections"]:
        for i, item in enumerate(payload["sections"], 1):
            if not isinstance(item, dict):
                raise IngestError(f"sections[{i}] is not an object")
            title = str(item.get("title") or item.get("section_title") or "(unknown)")
            text = str(item.get("text") or "")
            page_number = int(item.get("page_number") or i)
            blocks = [TextBlock(kind="heading", text=title, heading_level=1)]
            if text.strip():
                blocks.append(TextBlock(kind="paragraph", text=text))
            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=f"{title}\n\n{text}".strip(),
                    extraction_method=ExtractionMethod.na,
                    blocks=blocks,
                )
            )
    else:
        raise IngestError(
            "JSON is neither pre-chunked (chunks[]) nor a raw document (pages[] or sections[])"
        )

    return ParsedDocument(
        doc_id=raw.doc_id,
        document_name=document_name or raw.document_name,
        source_url=source_url,
        media_type=raw.media_type,
        filename=raw.filename,
        pages=pages,
    )

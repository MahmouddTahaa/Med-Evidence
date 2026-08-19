from __future__ import annotations

import statistics
from pathlib import Path

from clinical_rag.errors import IngestError
from clinical_rag.parsing.cleanup import strip_repeating_headers_footers
from clinical_rag.parsing.pdf_ocr_tesseract import ocr_page, tesseract_available
from clinical_rag.parsing.quality import needs_ocr
from clinical_rag.schemas import (
    ExtractionMethod,
    ParsedDocument,
    ParsedPage,
    ParserConfig,
    ParserProfile,
    RawDocument,
    TextBlock,
)


def _extract_blocks(page) -> tuple[str, list[TextBlock]]:
    data = page.get_text("dict")
    rows: list[tuple[str, float]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            line_text = "".join(s.get("text", "") for s in spans).strip()
            if not line_text:
                continue
            size = max((float(s.get("size") or 0) for s in spans), default=0.0)
            rows.append((line_text, size))
    if not rows:
        text = page.get_text("text") or ""
        blocks = [TextBlock(kind="paragraph", text=text)] if text.strip() else []
        return text, blocks

    sizes = [s for _, s in rows]
    median = statistics.median(sizes) if sizes else 0.0
    blocks: list[TextBlock] = []
    for text, size in rows:
        heading = median > 0 and size >= median * 1.2 and len(text) < 120
        blocks.append(
            TextBlock(
                kind="heading" if heading else "paragraph",
                text=text,
                heading_level=1 if heading else None,
            )
        )
    joined = "\n".join(t for t, _ in rows)
    return joined, blocks


def parse_pdf(raw: RawDocument, parser: ParserConfig, cache_dir: Path | None) -> ParsedDocument:
    import pymupdf

    doc = pymupdf.open(raw.path)
    pages: list[ParsedPage] = []
    warnings: list[str] = []
    try:
        if doc.page_count == 0:
            raise IngestError(f"{raw.filename}: PDF has no pages")
        for i, page in enumerate(doc, 1):
            text, blocks = _extract_blocks(page)
            method = ExtractionMethod.text
            page_warnings: list[str] = []
            weak = needs_ocr(text, parser.min_chars, parser.min_alnum_ratio)
            want_ocr = parser.profile is ParserProfile.ocr_all or (
                parser.profile is ParserProfile.ocr_fallback and weak
            )
            if want_ocr:
                cache_path = None
                if cache_dir is not None:
                    cache_path = cache_dir / f"{raw.doc_id}_p{i}.txt"
                try:
                    ocr_text = ocr_page(
                        page, lang=parser.ocr_lang, dpi=parser.ocr_dpi, cache_path=cache_path
                    )
                except IngestError as exc:
                    page_warnings.append(str(exc))
                    warnings.append(f"{raw.filename} p{i}: {exc}")
                    if weak:
                        page_warnings.append("weak digital text and OCR unavailable")
                else:
                    ocr_text = ocr_text.strip()
                    if parser.profile is ParserProfile.ocr_all:
                        text, blocks = ocr_text, (
                            [TextBlock(kind="paragraph", text=ocr_text)] if ocr_text else []
                        )
                        method = ExtractionMethod.ocr
                    elif weak:
                        if ocr_text:
                            method = ExtractionMethod.hybrid if text.strip() else ExtractionMethod.ocr
                            text, blocks = ocr_text, [TextBlock(kind="paragraph", text=ocr_text)]
                        else:
                            page_warnings.append("OCR returned empty text")
            elif weak:
                page_warnings.append("low-text page; OCR skipped (text_only profile)")
                warnings.append(f"{raw.filename} p{i}: low-text page, OCR skipped")

            pages.append(
                ParsedPage(
                    page_number=i,
                    text=text,
                    extraction_method=method,
                    blocks=blocks,
                    warnings=page_warnings,
                )
            )
    finally:
        doc.close()

    if parser.profile is not ParserProfile.text_only and not tesseract_available():
        if any(needs_ocr(p.text, parser.min_chars, parser.min_alnum_ratio) for p in pages):
            warnings.append("tesseract not found; weak pages kept as digital text")

    parsed = ParsedDocument(
        doc_id=raw.doc_id,
        document_name=raw.document_name,
        source_url=raw.source_url,
        media_type=raw.media_type,
        filename=raw.filename,
        pages=pages,
        warnings=warnings,
    )
    return strip_repeating_headers_footers(parsed)

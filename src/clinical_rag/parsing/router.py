from __future__ import annotations

import json
from pathlib import Path

from clinical_rag.errors import IngestError
from clinical_rag.parsing.base import ParseOutcome
from clinical_rag.parsing.json_doc import parse_raw_json
from clinical_rag.parsing.json_prechunked import is_prechunked_payload, load_prechunked
from clinical_rag.parsing.pdf_pymupdf import parse_pdf
from clinical_rag.parsing.text_plain import parse_plain
from clinical_rag.parsing.xml_doc import parse_xml
from clinical_rag.schemas import MediaType, ParserConfig, RawDocument

_SUFFIX = {
    ".pdf": MediaType.pdf,
    ".md": MediaType.md,
    ".txt": MediaType.txt,
    ".json": MediaType.json,
    ".xml": MediaType.xml,
    ".nxml": MediaType.nxml,
}


def media_type_for(filename: str) -> MediaType:
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUFFIX:
        raise IngestError(f"Unsupported file type: {suffix or filename}")
    return _SUFFIX[suffix]


def parse_document(raw: RawDocument, parser: ParserConfig, cache_dir: Path | None) -> ParseOutcome:
    suffix = Path(raw.path).suffix.lower()
    if suffix == ".pdf":
        parsed = parse_pdf(raw, parser, cache_dir)
        return ParseOutcome(parsed=parsed, warnings=list(parsed.warnings))
    if suffix in {".md", ".txt"}:
        parsed = parse_plain(raw)
        return ParseOutcome(parsed=parsed)
    if suffix in {".xml", ".nxml"}:
        parsed = parse_xml(raw)
        return ParseOutcome(parsed=parsed)
    if suffix == ".json":
        try:
            payload = json.loads(Path(raw.path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IngestError(f"{raw.filename}: invalid JSON ({exc})") from exc
        if is_prechunked_payload(payload):
            chunks, warnings = load_prechunked(payload, raw)
            return ParseOutcome(prechunked=chunks, warnings=warnings)
        parsed = parse_raw_json(payload, raw)
        return ParseOutcome(parsed=parsed)
    raise IngestError(f"Unsupported file type: {suffix}")

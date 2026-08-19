from __future__ import annotations

from pathlib import Path

from clinical_rag.errors import IngestError
from clinical_rag.parsing.base import ParseOutcome
from clinical_rag.parsing.cleanup import strip_repeating_headers_footers
from clinical_rag.parsing.pdf_pymupdf import parse_pdf
from clinical_rag.parsing.router import parse_document as parse_document_default
from clinical_rag.schemas import ParserConfig, ParserEngine, RawDocument


def parse_with_engine(
    raw: RawDocument,
    parser: ParserConfig,
    cache_dir: Path | None,
) -> ParseOutcome:
    suffix = Path(raw.path).suffix.lower()
    # Non-PDF / non-engine-specific types always use the default router (md/txt/json/xml).
    if suffix != ".pdf" or parser.engine is ParserEngine.pymupdf:
        outcome = parse_document_default(raw, parser, cache_dir)
        if outcome.parsed is not None and suffix == ".pdf":
            outcome.parsed = strip_repeating_headers_footers(outcome.parsed)
        return outcome

    raise IngestError(f"unsupported parser engine: {parser.engine}")


# Keep pymupdf reference for type checkers / unused-import silence when routing PDFs.
_ = parse_pdf

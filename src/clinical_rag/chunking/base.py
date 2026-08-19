from __future__ import annotations

from dataclasses import dataclass

from clinical_rag.errors import IngestError
from clinical_rag.schemas import (
    Chunk,
    ExtractionMethod,
    IngestJobConfig,
    ParsedDocument,
    ParsedPage,
    RawDocument,
    StrategyId,
    TextBlock,
)

UNKNOWN = "(unknown)"


def apply_section_title_prefix(text: str, title: str) -> str:
    """Prepend section_title when the window does not already start with it."""
    heading = (title or "").strip()
    body = (text or "").lstrip()
    if not heading or heading == UNKNOWN or not body:
        return text
    if body.lower().startswith(heading.lower()):
        return text
    return f"{heading}\n{text}"
# ~4 characters per token. Offline stand-in for tiktoken; good enough to pack 300–500 token windows.
# LangChain recursive chunk_size uses the same ratio (target_tokens * CHARS_PER_TOKEN).
CHARS_PER_TOKEN = 4
_CHARS_PER_TOKEN = CHARS_PER_TOKEN


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def encode_tokens(text: str) -> list[str]:
    if not text:
        return []
    return [text[i : i + _CHARS_PER_TOKEN] for i in range(0, len(text), _CHARS_PER_TOKEN)]


def decode_tokens(tokens: list[str]) -> str:
    return "".join(tokens)


def window_ranges(n: int, size: int, overlap_ratio: float) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    size = max(1, size)
    step = max(1, int(size * (1 - overlap_ratio)))
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = min(n, i + size)
        out.append((i, j))
        if j == n:
            break
        i += step
    return out


def window_texts(text: str, size: int, overlap_ratio: float) -> list[str]:
    tokens = encode_tokens(text)
    if not tokens:
        return []
    if len(tokens) <= size:
        return [text]
    return [decode_tokens(tokens[i:j]) for i, j in window_ranges(len(tokens), size, overlap_ratio)]


@dataclass
class DraftChunk:
    text: str
    section_title: str
    page_number: int | None
    extraction_method: ExtractionMethod
    chunk_id: str | None = None
    parent_chunk_id: str | None = None


@dataclass(frozen=True)
class SpanCitation:
    page_number: int | None
    section_title: str
    extraction_method: ExtractionMethod


@dataclass
class FlattenedDocument:
    """ParsedDocument flattened to a single string with citation spans by char offset."""

    text: str
    spans: list[tuple[int, int, SpanCitation]]  # half-open [start, end)

    def citation_at(self, offset: int) -> SpanCitation:
        if not self.spans:
            raise IngestError("FlattenedDocument has no citation spans")
        if offset < 0:
            offset = 0
        for start, end, citation in self.spans:
            if start <= offset < end:
                return citation
        return self.spans[-1][2]


def flatten_parsed(parsed: ParsedDocument) -> FlattenedDocument:
    """Join page texts and map char offsets → (page_number, section_title, extraction_method)."""
    if not parsed.pages:
        raise IngestError("ParsedDocument has no pages to flatten")
    parts: list[str] = []
    spans: list[tuple[int, int, SpanCitation]] = []
    pos = 0
    for page in parsed.pages:
        piece = page.text if page.text.endswith("\n") else page.text + "\n"
        if not piece:
            continue
        start = pos
        end = pos + len(piece)
        spans.append(
            (
                start,
                end,
                SpanCitation(
                    page_number=page.page_number,
                    section_title=first_heading(page),
                    extraction_method=page.extraction_method,
                ),
            )
        )
        parts.append(piece)
        pos = end
    text = "".join(parts)
    if not text.strip():
        raise IngestError("ParsedDocument flatten produced empty text")
    if not spans:
        raise IngestError("ParsedDocument flatten produced no citation spans")
    return FlattenedDocument(text=text, spans=spans)


def drafts_from_char_splits(
    flat: FlattenedDocument,
    pieces: list[tuple[str, int]],
    *,
    section_titles: list[str | None] | None = None,
) -> list[DraftChunk]:
    """Map (chunk_text, start_offset) pairs to DraftChunks via the span map. Fail closed if empty."""
    drafts: list[DraftChunk] = []
    for i, (raw_text, start) in enumerate(pieces):
        text = raw_text.strip()
        if not text:
            continue
        citation = flat.citation_at(start)
        if section_titles is not None and i < len(section_titles) and section_titles[i]:
            title = section_titles[i] or UNKNOWN
        else:
            title = citation.section_title or UNKNOWN
        drafts.append(
            DraftChunk(
                text=text,
                section_title=title,
                page_number=citation.page_number,
                extraction_method=citation.extraction_method,
            )
        )
    if not drafts:
        raise IngestError("Chunk split produced no non-empty chunks")
    return drafts


def fallback_blocks(page: ParsedPage) -> list[TextBlock]:
    if page.blocks:
        return page.blocks
    if page.text.strip():
        return [TextBlock(kind="paragraph", text=page.text)]
    return []


def first_heading(page: ParsedPage) -> str:
    for block in fallback_blocks(page):
        if block.kind == "heading" and block.text.strip():
            return block.text.strip()
    return UNKNOWN


def stamp_chunk(
    draft: DraftChunk,
    *,
    raw: RawDocument,
    config: IngestJobConfig,
    strategy_id: StrategyId,
    index: int,
) -> Chunk:
    chunk_id = draft.chunk_id or f"{raw.doc_id}-{strategy_id.value}-{index:04d}"
    title = draft.section_title or UNKNOWN
    text = draft.text
    if config.chunk.prefix_section_title:
        text = apply_section_title_prefix(text, title)
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        document_name=raw.document_name or draft.text[:80],
        section_title=title,
        page_number=draft.page_number,
        source_url=raw.source_url,
        strategy_id=strategy_id,
        corpus_id=config.corpus_id,
        job_id=config.job_id,
        extraction_method=draft.extraction_method,
        token_count=count_tokens(text),
        embed_model_id=config.embed.model_id,
        parent_chunk_id=draft.parent_chunk_id,
        filename=raw.filename,
        doc_id=raw.doc_id,
    )


def stamp_prechunked(chunk: Chunk, config: IngestJobConfig) -> Chunk:
    data = chunk.model_dump()
    data["corpus_id"] = config.corpus_id
    data["job_id"] = config.job_id
    data["embed_model_id"] = config.embed.model_id
    data["strategy_id"] = StrategyId.passthrough
    data["token_count"] = count_tokens(chunk.text)
    return Chunk.model_validate(data)


def token_stream(
    parsed: ParsedDocument,
) -> tuple[list[str], list[int], list[ExtractionMethod], list[str]]:
    tokens: list[str] = []
    pages: list[int] = []
    methods: list[ExtractionMethod] = []
    titles: list[str] = []
    for page in parsed.pages:
        piece = page.text if page.text.endswith("\n") else page.text + "\n"
        encoded = encode_tokens(piece)
        title = first_heading(page)
        tokens.extend(encoded)
        pages.extend([page.page_number] * len(encoded))
        methods.extend([page.extraction_method] * len(encoded))
        titles.extend([title] * len(encoded))
    return tokens, pages, methods, titles

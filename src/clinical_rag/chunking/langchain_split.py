from __future__ import annotations

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from clinical_rag.chunking.base import (
    CHARS_PER_TOKEN,
    DraftChunk,
    count_tokens,
    drafts_from_char_splits,
    flatten_parsed,
    window_texts,
)
from clinical_rag.errors import IngestError
from clinical_rag.schemas import ChunkConfig, ParsedDocument

_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_MD_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def _char_window(cfg: ChunkConfig) -> tuple[int, int]:
    chunk_size = max(1, cfg.target_tokens * CHARS_PER_TOKEN)
    overlap = max(0, int(chunk_size * cfg.overlap_ratio))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    return chunk_size, overlap


def _token_window(cfg: ChunkConfig) -> tuple[int, int]:
    chunk_size = max(1, cfg.target_tokens)
    overlap = max(0, int(chunk_size * cfg.overlap_ratio))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    return chunk_size, overlap


def _recursive_splitter(cfg: ChunkConfig) -> RecursiveCharacterTextSplitter:
    chunk_size, overlap = _char_window(cfg)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=_RECURSIVE_SEPARATORS,
        length_function=len,
        add_start_index=True,
    )


def _section_title_from_md_meta(meta: dict) -> str | None:
    for key in ("h4", "h3", "h2", "h1"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _md_section_start(flat_text: str, meta: dict, search_from: int) -> int:
    """Locate section in original text via header line (splitter rewrites body whitespace)."""
    for level, key in ((4, "h4"), (3, "h3"), (2, "h2"), (1, "h1")):
        title = meta.get(key)
        if not isinstance(title, str) or not title.strip():
            continue
        marker = f"{'#' * level} {title.strip()}"
        idx = flat_text.find(marker, search_from)
        if idx >= 0:
            return idx
        idx = flat_text.find(marker)
        if idx >= 0:
            return idx
    return search_from


def chunk_langchain_recursive(parsed: ParsedDocument, cfg: ChunkConfig) -> list[DraftChunk]:
    flat = flatten_parsed(parsed)
    docs = _recursive_splitter(cfg).create_documents([flat.text])
    pieces = [(d.page_content, int(d.metadata.get("start_index", 0))) for d in docs]
    return drafts_from_char_splits(flat, pieces)


def chunk_langchain_token(parsed: ParsedDocument, cfg: ChunkConfig) -> list[DraftChunk]:
    flat = flatten_parsed(parsed)
    chunk_size, overlap = _token_window(cfg)
    splitter = TokenTextSplitter(
        encoding_name="gpt2",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
    )
    docs = splitter.create_documents([flat.text])
    pieces = [(d.page_content, int(d.metadata.get("start_index", 0))) for d in docs]
    return drafts_from_char_splits(flat, pieces)


def chunk_langchain_markdown(parsed: ParsedDocument, cfg: ChunkConfig) -> list[DraftChunk]:
    flat = flatten_parsed(parsed)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_MD_HEADERS,
        strip_headers=False,
    )
    sections = header_splitter.split_text(flat.text)
    if not sections:
        raise IngestError("MarkdownHeaderTextSplitter produced no sections")

    packer = _recursive_splitter(cfg)
    char_size, _ = _char_window(cfg)
    pieces: list[tuple[str, int]] = []
    titles: list[str | None] = []
    search_from = 0

    for section in sections:
        content = section.page_content
        if not content.strip():
            continue
        title = _section_title_from_md_meta(section.metadata)
        idx = _md_section_start(flat.text, section.metadata, search_from)
        search_from = idx + 1

        needs_pack = count_tokens(content) > cfg.max_tokens or len(content) > char_size
        if not needs_pack:
            pieces.append((content, idx))
            titles.append(title)
            continue

        if len(content) > char_size:
            for sub in packer.create_documents([content]):
                local_start = int(sub.metadata.get("start_index", 0))
                pieces.append((sub.page_content, idx + local_start))
                titles.append(title)
        else:
            for part in window_texts(content, cfg.target_tokens, cfg.overlap_ratio):
                pieces.append((part, idx))
                titles.append(title)

    return drafts_from_char_splits(flat, pieces, section_titles=titles)

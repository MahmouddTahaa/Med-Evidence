from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np

from clinical_rag.chunking.base import (
    DraftChunk,
    count_tokens,
    drafts_from_char_splits,
    flatten_parsed,
    window_texts,
)
from clinical_rag.errors import IngestError
from clinical_rag.schemas import ChunkConfig, ParsedDocument

# Fixed breakpoint embedder — must not use the ingest job embedder (stage-2 confound).
SEMANTIC_BREAKPOINT_MODEL = "all-MiniLM-L6-v2"
# Locked percentile default (LangChain SemanticChunker default).
BREAKPOINT_PERCENTILE = 95.0

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")

EncodeFn = Callable[[list[str]], list[list[float]]]

# Tests monkeypatch this to avoid downloading MiniLM in CI.
_encode_override: EncodeFn | None = None
_model_cache: Any = None


def set_semantic_encode_override(fn: EncodeFn | None) -> None:
    """Test hook: replace MiniLM encoding without downloading weights."""
    global _encode_override
    _encode_override = fn


def _load_minilm() -> Any:
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer

        _model_cache = SentenceTransformer(SEMANTIC_BREAKPOINT_MODEL)
    return _model_cache


def _default_encode(sentences: list[str]) -> list[list[float]]:
    if _encode_override is not None:
        return _encode_override(sentences)
    model = _load_minilm()
    vectors = model.encode(sentences, normalize_embeddings=True)
    return [list(map(float, row)) for row in vectors]


def _split_sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """Return (sentence, start, end) spans covering `text` without rewriting whitespace."""
    if not text.strip():
        return []
    spans: list[tuple[str, int, int]] = []
    last = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        start, end = last, match.start()
        piece = text[start:end]
        if piece.strip():
            spans.append((piece.strip(), start, end))
        last = match.end()
    tail = text[last:]
    if tail.strip():
        spans.append((tail.strip(), last, len(text)))
    if not spans:
        spans.append((text.strip(), 0, len(text)))
    return spans


def _cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.array([], dtype=np.float64)
    sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    return 1.0 - sims


def _breakpoint_indices(distances: np.ndarray, percentile: float) -> list[int]:
    if distances.size == 0:
        return []
    threshold = float(np.percentile(distances, percentile))
    return [i for i, d in enumerate(distances) if d > threshold]


def _group_spans(
    spans: list[tuple[str, int, int]],
    breaks: list[int],
    full: str,
) -> list[tuple[str, int]]:
    if not spans:
        return []
    break_set = set(breaks)
    groups: list[list[tuple[str, int, int]]] = [[]]
    for i, span in enumerate(spans):
        groups[-1].append(span)
        if i in break_set:
            groups.append([])
    pieces: list[tuple[str, int]] = []
    for group in groups:
        if not group:
            continue
        start = group[0][1]
        end = group[-1][2]
        chunk = full[start:end].strip()
        if chunk:
            pieces.append((chunk, start))
    return pieces


def _local_semantic_pieces(text: str, encode: EncodeFn, percentile: float) -> list[tuple[str, int]]:
    spans = _split_sentence_spans(text)
    if len(spans) <= 1:
        return [(text.strip(), 0)] if text.strip() else []
    sentences = [s[0] for s in spans]
    vectors = encode(sentences)
    if len(vectors) != len(sentences):
        raise IngestError("Semantic encoder returned the wrong number of vectors")
    emb = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    emb = emb / norms
    breaks = _breakpoint_indices(_cosine_distances(emb), percentile)
    return _group_spans(spans, breaks, text)


def _try_langchain_semantic_chunker(text: str, encode: EncodeFn) -> list[str] | None:
    """Prefer langchain_text_splitters.SemanticChunker when present; else None."""
    try:
        from langchain_text_splitters import SemanticChunker  # type: ignore[attr-defined]
    except ImportError:
        return None

    class _EmbedAdapter:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return encode(texts)

        def embed_query(self, query: str) -> list[float]:
            return encode([query])[0]

    splitter = SemanticChunker(
        _EmbedAdapter(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=BREAKPOINT_PERCENTILE,
    )
    return [c for c in splitter.split_text(text) if c and c.strip()]


def _locate_rewritten_chunks(full: str, chunks: list[str]) -> list[tuple[str, int]]:
    """Best-effort locate for LC SemanticChunker output that may rewrite whitespace."""
    pieces: list[tuple[str, int]] = []
    search_from = 0
    for chunk in chunks:
        idx = full.find(chunk, search_from)
        if idx < 0:
            idx = full.find(chunk)
        if idx < 0:
            probe = re.sub(r"\s+", " ", chunk[: min(80, len(chunk))]).strip()
            collapsed = re.sub(r"\s+", " ", full)
            # Fall back to search_from citation if exact locate fails.
            idx = search_from if probe and probe[:40] in collapsed else search_from
        pieces.append((chunk, idx))
        search_from = idx + max(1, len(chunk) // 4)
    return pieces


def chunk_semantic(
    parsed: ParsedDocument,
    cfg: ChunkConfig,
    *,
    encode: EncodeFn | None = None,
) -> list[DraftChunk]:
    """Embedding-breakpoint semantic chunking (not RecursiveCharacter)."""
    flat = flatten_parsed(parsed)
    encode_fn = encode or _default_encode

    lc_parts = _try_langchain_semantic_chunker(flat.text, encode_fn)
    if lc_parts is not None:
        base_pieces = _locate_rewritten_chunks(flat.text, lc_parts)
    else:
        base_pieces = _local_semantic_pieces(flat.text, encode_fn, BREAKPOINT_PERCENTILE)
    if not base_pieces:
        raise IngestError("Semantic chunking produced no chunks")

    pieces: list[tuple[str, int]] = []
    for part, start in base_pieces:
        if count_tokens(part) > cfg.max_tokens:
            for window in window_texts(part, cfg.target_tokens, cfg.overlap_ratio):
                if window.strip():
                    pieces.append((window, start))
        else:
            pieces.append((part, start))

    return drafts_from_char_splits(flat, pieces)

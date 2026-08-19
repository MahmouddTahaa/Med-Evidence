from __future__ import annotations

import hashlib
from pathlib import Path

from clinical_rag.parsing.router import media_type_for
from clinical_rag.schemas import LegalFlags, RawDocument


class HashEmbedder:
    """Deterministic stand-in so tests do not download bge-m3."""

    model_id = "hash-test"
    device = "cpu"
    warnings: list[str] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [(b / 127.5) - 1.0 for b in digest]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


def legal_ok() -> LegalFlags:
    return LegalFlags(
        open_access_or_reusable=True,
        redistribution_ok_for_indexing=True,
        edition_current=True,
        attribution_documented=True,
    )


def raw_doc(path: Path, *, doc_id: str = "doc", media=None) -> RawDocument:
    return RawDocument(
        doc_id=doc_id,
        filename=path.name,
        media_type=media or media_type_for(path.name),
        document_name=path.stem,
        source_url="https://example.org/demo",
        path=str(path),
        legal=legal_ok(),
    )

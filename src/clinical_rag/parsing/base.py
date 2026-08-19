from __future__ import annotations

from dataclasses import dataclass, field

from clinical_rag.schemas import Chunk, ParsedDocument


@dataclass
class ParseOutcome:
    parsed: ParsedDocument | None = None
    prechunked: list[Chunk] | None = None
    warnings: list[str] = field(default_factory=list)

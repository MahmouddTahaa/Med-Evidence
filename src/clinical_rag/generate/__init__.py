"""Grounded generation for the clinician query path.

Retrieval stays in `clinical_rag.query`. This package owns classify → retrieve
path → grounded answer + extractive fallback.
"""

from clinical_rag.generate.engine import GroundedEngine
from clinical_rag.generate.models import (
    Citation,
    Confidence,
    EvidenceQuote,
    InputClass,
    Outcome,
    TurnRequest,
    TurnResult,
)

__all__ = [
    "Citation",
    "Confidence",
    "EvidenceQuote",
    "GroundedEngine",
    "InputClass",
    "Outcome",
    "TurnRequest",
    "TurnResult",
]

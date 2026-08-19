"""Pydantic types for one clinician turn (generation only)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from clinical_rag.schemas import SmokeHit


class InputClass(str, Enum):
    allowed = "Allowed"
    needs_caution = "NeedsCaution"
    refuse = "Refuse"


class Outcome(str, Enum):
    success = "success"
    complex = "complex"
    refusal = "refusal"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    insufficient = "insufficient"


class Citation(BaseModel):
    document_name: str
    section_title: str
    page_number: int | None = None
    chunk_id: str
    hit_index: int | None = None


class EvidenceQuote(BaseModel):
    text: str
    chunk_id: str
    hit_index: int | None = None


class ChatMessage(BaseModel):
    role: str  # user | assistant
    content: str


class TurnRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)


class TurnResult(BaseModel):
    recommendation: str | None = None
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence = Confidence.insufficient
    outcome: Outcome = Outcome.complex
    disclaimer: str = ""
    hits: list[SmokeHit] = Field(default_factory=list)
    provider_used: str = ""
    retrieval_query: str = ""
    input_class: InputClass = InputClass.allowed
    insufficient_evidence: bool = False
    out_of_corpus: bool = False
    # Pre-bind citation accounting for continuous citation_accuracy in the UI.
    citations_proposed: int = 0
    citations_bound: int = 0
    citations_auto_filled: bool = False


DISCLAIMER = (
    "Educational use only — not a diagnosis, prescription, or emergency advice. "
    "Verify against primary sources and clinical judgment. If this is an emergency, "
    "call local emergency services."
)

INSUFFICIENT_EVIDENCE_COPY = (
    "Retrieved sources are too weak or incomplete to support a grounded recommendation. "
    "The passages below may touch the topic but do not answer it with enough evidence. "
    "Try rephrasing with a specific drug and section (e.g. contraindications, monitoring), "
    "or consult a clinician and the primary reference."
)

OUT_OF_CORPUS_COPY = (
    "This looks like a guideline, screening, vaccination-schedule, or risk-score question "
    "outside the indexed StatPearls pharmacology corpus. Passages below may be loosely "
    "related but are not a substitute for the primary guideline. Ask a drug-specific "
    "pharmacology question (mechanism, contraindications, interactions, monitoring, dosing)."
)

REFUSAL_COPY = (
    "I cannot help with that request. Med-Evidence is a pharmacology copilot for "
    "clinicians and students only — not for personal patient care or self-treatment. "
    "If you are a patient, talk to your clinician. "
    "If you are a clinician or student, rephrase as an educational drug question "
    "(mechanism, contraindications, interactions, monitoring) grounded in the sources."
)

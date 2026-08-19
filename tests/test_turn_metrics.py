"""Tests for citation accuracy and session averaging."""

from __future__ import annotations

from clinical_rag.generate.engine import case_c_result, result_from_tagged
from clinical_rag.generate.models import InputClass
from clinical_rag.generate.turn_metrics import (
    citation_accuracy,
    include_in_session_average,
)
from clinical_rag.schemas import SmokeHit


def _hits(n: int = 5) -> list[SmokeHit]:
    return [
        SmokeHit(
            score=0.9 - i * 0.05,
            text=f"Passage about drug fact {i}.",
            document_name=f"doc{i}",
            section_title="Adverse Effects",
            page_number=1,
            chunk_id=f"c{i}",
        )
        for i in range(n)
    ]


def test_citation_accuracy_fraction_on_hallucinated_id() -> None:
    hits = _hits(1)
    tagged = """<<<RECOMMENDATION>>>
Avoid in allergy.
<<<EVIDENCE>>>
- "Passage about drug fact 0." [1]
<<<CITATIONS>>>
[1] chunk_id=c0
[2] chunk_id=hallucinated
<<<CONFIDENCE>>>
high
"""
    result = result_from_tagged(
        tagged,
        hits,
        input_class=InputClass.allowed,
        retrieval_query="q",
        provider_used="stub",
        prompt_k=5,
    )
    assert result is not None
    assert citation_accuracy(result) == 0.5
    assert result.citations_proposed == 2
    assert result.citations_bound == 1


def test_citation_accuracy_dedupes_duplicate_proposals() -> None:
    hits = _hits(1)
    tagged = """<<<RECOMMENDATION>>>
Avoid in allergy.
<<<EVIDENCE>>>
- "Passage about drug fact 0." [1]
<<<CITATIONS>>>
[1] chunk_id=c0
[1] chunk_id=c0
<<<CONFIDENCE>>>
high
"""
    result = result_from_tagged(
        tagged,
        hits,
        input_class=InputClass.allowed,
        retrieval_query="q",
        provider_used="stub",
        prompt_k=5,
    )
    assert result is not None
    assert result.citations_proposed == 1
    assert result.citations_bound == 1
    assert citation_accuracy(result) == 1.0


def test_citation_accuracy_na_on_case_c_autofill() -> None:
    result = case_c_result(
        _hits(5),
        input_class=InputClass.allowed,
        retrieval_query="q",
        prompt_k=5,
    )
    assert result.citations_auto_filled is True
    assert citation_accuracy(result) is None


def test_session_average_excludes_case_c_and_refusal() -> None:
    ok = {
        "outcome": "success",
        "insufficient_evidence": False,
        "faithfulness": 0.9,
        "citation_accuracy": 1.0,
    }
    case_c = {
        "outcome": "complex",
        "insufficient_evidence": True,
        "exclude_from_session": True,
        "citation_accuracy": None,
    }
    refusal = {
        "outcome": "refusal",
        "exclude_from_session": True,
        "citation_accuracy": None,
    }
    assert include_in_session_average(ok) is True
    assert include_in_session_average(case_c) is False
    assert include_in_session_average(refusal) is False

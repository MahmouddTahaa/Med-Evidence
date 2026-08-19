"""Claim split + entailment scoring (stubbed NLI via lexical scorer)."""

from __future__ import annotations

from clinical_rag.generate.claim_metrics import (
    LexicalEntailmentScorer,
    score_claims,
    split_claims,
)
from clinical_rag.generate.engine import result_from_tagged
from clinical_rag.generate.models import InputClass
from clinical_rag.generate.turn_metrics import citation_accuracy
from clinical_rag.schemas import SmokeHit


def test_split_claims_filters_short_and_disclaimer() -> None:
    text = (
        "Ampicillin is contraindicated in patients with penicillin allergy. "
        "Educational use only — not a diagnosis. "
        "Short. "
        "Monitor renal function during high-dose therapy in older adults."
    )
    claims = split_claims(text)
    assert len(claims) == 2
    assert "penicillin allergy" in claims[0]
    assert "renal function" in claims[1]


def test_lexical_faithfulness_is_continuous() -> None:
    premises = [
        "Ampicillin is contraindicated in patients with a history of penicillin allergy."
    ]
    rec = (
        "Ampicillin is contraindicated in penicillin allergy. "
        "Patients should take ampicillin with grapefruit juice daily."
    )
    report = score_claims(rec, premises, LexicalEntailmentScorer())
    assert report.faithfulness is not None
    assert 0.0 < report.faithfulness < 1.0
    assert len(report.claims) == 2
    assert report.claims[0].entailment_prob > report.claims[1].entailment_prob


def test_citation_accuracy_uses_proposed_before_bind() -> None:
    hits = [
        SmokeHit(
            score=0.9,
            text="Ampicillin is contraindicated in penicillin allergy.",
            document_name="Ampicillin",
            section_title="Contraindications",
            page_number=1,
            chunk_id="c1",
        )
    ]
    tagged = """<<<RECOMMENDATION>>>
Ampicillin should be avoided in penicillin allergy.
<<<EVIDENCE>>>
- "Ampicillin is contraindicated in penicillin allergy." [1]
<<<CITATIONS>>>
[1] chunk_id=c1
[2] chunk_id=hallucinated-id
<<<CONFIDENCE>>>
high
"""
    result = result_from_tagged(
        tagged,
        hits,
        input_class=InputClass.allowed,
        retrieval_query="ampicillin contraindications",
        provider_used="stub",
        prompt_k=5,
    )
    assert result is not None
    assert result.citations_proposed == 2
    assert result.citations_bound == 1
    assert result.citations_auto_filled is False
    assert result.citations_bound / result.citations_proposed == 0.5


def test_bracket_n_wins_for_answer_but_strict_accuracy_penalizes_bad_id() -> None:
    hits = [
        SmokeHit(
            score=0.9,
            text="Ampicillin is contraindicated in penicillin allergy.",
            document_name="Ampicillin",
            section_title="Contraindications",
            page_number=1,
            chunk_id="real-chunk",
        )
    ]
    tagged = """<<<RECOMMENDATION>>>
Avoid ampicillin in penicillin allergy.
<<<EVIDENCE>>>
- "Ampicillin is contraindicated in penicillin allergy." [1]
<<<CITATIONS>>>
[1] chunk_id=also-wrong
<<<CONFIDENCE>>>
medium
"""
    result = result_from_tagged(
        tagged,
        hits,
        input_class=InputClass.allowed,
        retrieval_query="ampicillin",
        provider_used="stub",
        prompt_k=5,
    )
    assert result is not None
    # Answer still cites the real chunk via [n].
    assert result.citations_auto_filled is False
    assert any(c.chunk_id == "real-chunk" for c in result.citations)
    # Accuracy scores the literal bad chunk_id=.
    assert result.citations_proposed == 1
    assert result.citations_bound == 0
    assert citation_accuracy(result) == 0.0

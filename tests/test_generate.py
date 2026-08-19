"""Stubbed tests for grounded generation — no live LLM APIs."""

from __future__ import annotations

from clinical_rag.config import GenerationSettings
from clinical_rag.generate.engine import (
    GroundedEngine,
    bind_citations,
    parse_tagged_answer,
    result_from_tagged,
)
from clinical_rag.generate.guardrails import classify_input, keyword_classify
from clinical_rag.generate.models import (
    ChatMessage,
    Confidence,
    InputClass,
    Outcome,
    TurnRequest,
)
from clinical_rag.generate.prompts import grounded_answer_messages
from clinical_rag.generate.retrieve_path import fuse_by_max_score, is_weak
from clinical_rag.schemas import SmokeHit


def _hit(
    chunk_id: str,
    score: float,
    *,
    text: str = "Ampicillin is contraindicated in penicillin allergy.",
    document_name: str = "Ampicillin",
    section_title: str = "Contraindications",
    page_number: int | None = 2,
) -> SmokeHit:
    return SmokeHit(
        score=score,
        text=text,
        document_name=document_name,
        section_title=section_title,
        page_number=page_number,
        chunk_id=chunk_id,
    )


class StubSession:
    def __init__(self, hits_by_query: dict[str, list[SmokeHit]] | None = None, default: list[SmokeHit] | None = None):
        self.hits_by_query = hits_by_query or {}
        self.default = default or []
        self.calls: list[str] = []

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[SmokeHit]:
        self.calls.append(query)
        hits = self.hits_by_query.get(query, self.default)
        if top_k is not None:
            return hits[:top_k]
        return list(hits)


class StubLlm:
    name = "stub"

    def __init__(
        self,
        *,
        classify: str | None = None,
        rewrite: str | None = None,
        expand: str | None = None,
        answer: str | None = None,
        fail: bool = False,
    ) -> None:
        self.classify = classify
        self.rewrite = rewrite
        self.expand = expand
        self.answer = answer
        self.fail = fail
        self.complete_calls = 0

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str:
        self.complete_calls += 1
        if self.fail:
            raise RuntimeError("stub LLM down")
        blob = " ".join(str(m.get("content", "")) for m in messages).lower()
        if "classify" in blob or '"class"' in blob or "needscaution" in blob.lower():
            return self.classify or '{"class":"Allowed"}'
        if "standalone" in blob or "rewrite" in blob:
            return self.rewrite or '{"query":"ampicillin contraindications"}'
        if "variants" in blob or "diverse" in blob:
            return self.expand or '{"queries":["q1","q2","q3"]}'
        return self.answer or (
            "<<<RECOMMENDATION>>>\nAvoid in penicillin allergy.\n"
            "<<<EVIDENCE>>>\n- \"contraindicated in penicillin allergy\" [1]\n"
            "<<<CITATIONS>>>\n[1] chunk_id=c1\n"
            "<<<CONFIDENCE>>>\nhigh\n"
        )

    def stream(self, messages: list[dict]):
        text = self.complete(messages)
        # Yield in chunks so streaming path is exercised.
        mid = max(1, len(text) // 3)
        yield text[:mid]
        yield text[mid:]


# --- Guardrails ---


def test_keyword_refuse_self_harm():
    assert keyword_classify("I want to kill myself") is InputClass.refuse


def test_keyword_refuse_unrelated():
    assert keyword_classify("What's the weather in Boston?") is InputClass.refuse


def test_keyword_caution_pregnancy():
    assert keyword_classify("Is ampicillin safe in pregnancy?") is InputClass.needs_caution


def test_keyword_refuse_patient_voice():
    assert (
        keyword_classify("I'm a patient — should I take ampicillin for my sore throat?")
        is InputClass.refuse
    )
    assert (
        keyword_classify("My doctor prescribed metoprolol; can I stop it if I feel better?")
        is InputClass.refuse
    )
    assert (
        keyword_classify("I have fever and pain — what should I take?")
        is InputClass.refuse
    )


def test_keyword_allowed_unclear_returns_none():
    assert keyword_classify("What are ampicillin contraindications?") is None


def test_classify_llm_failure_needs_caution():
    llm = StubLlm(fail=True)
    assert classify_input("Tell me about beta blockers", llm) is InputClass.needs_caution


def test_classify_allowed_without_llm():
    assert classify_input("What are ampicillin contraindications?") is InputClass.allowed


# --- Retrieve path ---


def test_is_weak_empty_and_threshold():
    assert is_weak([], threshold=0.35) is True
    assert is_weak([_hit("c1", 0.2)], threshold=0.35) is True
    assert is_weak([_hit("c1", 0.9)], threshold=0.35) is False


def test_fuse_by_max_score_keeps_best():
    a = [_hit("c1", 0.4), _hit("c2", 0.8)]
    b = [_hit("c1", 0.9), _hit("c3", 0.5)]
    fused = fuse_by_max_score([a, b], top_k=10)
    by_id = {h.chunk_id: h.score for h in fused}
    assert by_id["c1"] == 0.9
    assert by_id["c2"] == 0.8
    assert by_id["c3"] == 0.5
    assert fused[0].chunk_id == "c1"


# --- Prompts ---


def test_grounded_prompt_includes_citation_headers():
    hits = [_hit("c1", 0.9)]
    msgs = grounded_answer_messages("contraindications?", hits)
    user = msgs[-1]["content"]
    assert "Ampicillin" in user
    assert "Contraindications" in user
    assert "page 2" in user
    assert "c1" in user
    assert "[1]" in user


# --- Engine ---


def test_engine_hard_refuse_skips_retrieve():
    from clinical_rag.generate.models import REFUSAL_COPY

    session = StubSession(default=[_hit("c1", 0.9)])
    engine = GroundedEngine(session, StubLlm(), settings=GenerationSettings())
    result = engine.turn(TurnRequest(message="I want to kill myself"))
    assert result.outcome is Outcome.refusal
    assert session.calls == []
    assert result.hits == []
    assert result.recommendation == REFUSAL_COPY
    assert "clinician" in (result.recommendation or "").lower()


def test_engine_success_path():
    session = StubSession(default=[_hit("c1", 0.91), _hit("c2", 0.8)])
    engine = GroundedEngine(session, StubLlm(), settings=GenerationSettings())
    result = engine.turn(TurnRequest(message="What are ampicillin contraindications?"))
    assert result.outcome is Outcome.success
    assert result.recommendation
    assert any(c.chunk_id == "c1" for c in result.citations)
    assert result.confidence is Confidence.high
    assert result.disclaimer


def test_engine_case_c_on_weak_hits():
    from clinical_rag.generate.models import INSUFFICIENT_EVIDENCE_COPY

    session = StubSession(default=[_hit("c1", 0.1)])
    llm = StubLlm(
        rewrite='{"query":"weak rewrite"}',
        expand='{"queries":["a","b"]}',
    )
    # All retrieves stay weak
    session.hits_by_query = {
        "weak rewrite": [_hit("c1", 0.1)],
        "a": [_hit("c1", 0.12)],
        "b": [_hit("c2", 0.11)],
    }
    engine = GroundedEngine(session, llm, settings=GenerationSettings(weak_score=0.35))
    result = engine.turn(TurnRequest(message="obscure dosing trivia xyz"))
    assert result.insufficient_evidence is True
    assert result.out_of_corpus is False
    assert result.outcome is Outcome.complex
    assert result.confidence is Confidence.insufficient
    assert result.hits  # Case C still shows chunks
    assert result.recommendation == INSUFFICIENT_EVIDENCE_COPY
    assert "rephras" in (result.recommendation or "").lower()


def test_out_of_corpus_patterns():
    from clinical_rag.generate.guardrails import is_out_of_corpus

    assert is_out_of_corpus(
        "What is the USPSTF breast cancer screening interval for average-risk adults?"
    )
    assert is_out_of_corpus(
        "What is the recommended A1C target for most non-pregnant adults with type 2 diabetes?"
    )
    assert is_out_of_corpus(
        "What is the CHADS2 score threshold for anticoagulation in atrial fibrillation?"
    )
    assert is_out_of_corpus(
        "What is first-line pharmacologic treatment for essential hypertension per ACC/AHA?"
    )
    assert not is_out_of_corpus("What are the contraindications for Ampicillin?")
    assert not is_out_of_corpus("How should patients on Amitriptyline be monitored?")


def test_engine_out_of_corpus_forces_case_c_even_with_strong_hits():
    from clinical_rag.generate.models import OUT_OF_CORPUS_COPY

    # Strong retrieve scores would normally generate — out-of-corpus must still Case C.
    session = StubSession(default=[_hit("c1", 0.99)])
    llm = StubLlm(classify='{"class":"Allowed"}')
    engine = GroundedEngine(session, llm, settings=GenerationSettings(weak_score=0.35))
    result = engine.turn(
        TurnRequest(
            message="What is the USPSTF breast cancer screening interval for average-risk adults?"
        )
    )
    assert result.out_of_corpus is True
    assert result.insufficient_evidence is True
    assert result.recommendation == OUT_OF_CORPUS_COPY
    assert result.confidence is Confidence.insufficient
    assert llm.complete_calls == 1  # classify only — no grounded generate


def test_grounded_prompt_includes_uncertainty_tone_rule():
    from clinical_rag.generate.prompts import TAGGED_ANSWER_INSTRUCTIONS

    assert "Match recommendation tone to CONFIDENCE" in TAGGED_ANSWER_INSTRUCTIONS
    assert "Never imply this replaces clinical judgment" in TAGGED_ANSWER_INSTRUCTIONS


def test_engine_cascade_exhaust_extractive():
    session = StubSession(default=[_hit("c1", 0.95)])
    engine = GroundedEngine(session, StubLlm(fail=True), settings=GenerationSettings())
    # classify_input on unclear+fail → NeedsCaution; retrieve strong; generate fails → extractive
    result = engine.turn(TurnRequest(message="Ampicillin mechanism of action?"))
    assert result.provider_used == "extractive"
    assert result.recommendation is None
    assert result.citations
    assert result.hits


def test_bind_citations_drops_unknown_chunk_id():
    hits = [_hit("c1", 0.9)]
    bound = bind_citations(
        [{"chunk_id": "c1"}, {"chunk_id": "not-in-hits"}, {"chunk_id": "c1"}],
        hits,
    )
    assert len(bound) == 1
    assert bound[0].chunk_id == "c1"


def test_result_from_tagged_missing_citations_auto_binds_hits():
    """Strong retrieve + malformed cites must not become Case C (weak-source copy)."""
    hits = [_hit("c1", 0.9)]
    text = (
        "<<<RECOMMENDATION>>>\nAvoid in penicillin allergy.\n"
        "<<<EVIDENCE>>>\n- made up [9]\n"
        "<<<CITATIONS>>>\n[9] chunk_id=missing\n"
        "<<<CONFIDENCE>>>\nhigh\n"
    )
    result = result_from_tagged(
        text,
        hits,
        input_class=InputClass.allowed,
        retrieval_query="q",
        provider_used="stub",
        prompt_k=5,
    )
    assert result is not None
    assert result.insufficient_evidence is False
    assert result.outcome is Outcome.success
    assert result.recommendation == "Avoid in penicillin allergy."
    assert any(c.chunk_id == "c1" for c in result.citations)
    assert result.confidence is Confidence.low  # lowered after auto-bind


def test_result_from_tagged_recovers_citations_from_evidence_indexes():
    hits = [_hit("c1", 0.9), _hit("c2", 0.8)]
    text = (
        "<<<RECOMMENDATION>>>\nSodium channel blockade.\n"
        "<<<EVIDENCE>>>\n- \"decreases nerve membrane permeability to sodium\" [1]\n"
        "<<<CITATIONS>>>\n\n"
        "<<<CONFIDENCE>>>\nmedium\n"
    )
    result = result_from_tagged(
        text,
        hits,
        input_class=InputClass.allowed,
        retrieval_query="chloroprocaine moa",
        provider_used="stub",
        prompt_k=5,
    )
    assert result is not None
    assert result.insufficient_evidence is False
    assert result.citations[0].chunk_id == "c1"


def test_parse_tagged_answer_sections():
    text = (
        "<<<RECOMMENDATION>>>\nDo not use.\n"
        "<<<EVIDENCE>>>\n- quote [1]\n"
        "<<<CITATIONS>>>\n[1] chunk_id=c1\n"
        "<<<CONFIDENCE>>>\nmedium\n"
    )
    sections = parse_tagged_answer(text)
    assert "Do not use" in sections["recommendation"]
    assert "c1" in sections["citations"]
    assert "medium" in sections["confidence"].lower()


def test_engine_with_history_memory():
    session = StubSession(default=[_hit("c1", 0.9)])
    engine = GroundedEngine(session, StubLlm(), settings=GenerationSettings(memory_turns=6))
    result = engine.turn(
        TurnRequest(
            message="What about contraindications?",
            history=[
                ChatMessage(role="user", content="Tell me about Ampicillin"),
                ChatMessage(role="assistant", content="It is a penicillin antibiotic."),
            ],
        )
    )
    assert result.outcome is Outcome.success

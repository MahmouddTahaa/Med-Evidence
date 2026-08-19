"""GroundedEngine: classify → retrieve path → generate / Case C / extractive."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from clinical_rag.adapters.llms import CascadeClient, LlmClient
from clinical_rag.config import GenerationSettings, get_settings
from clinical_rag.errors import QueryError
from clinical_rag.generate.guardrails import classify_input, is_out_of_corpus
from clinical_rag.generate.models import (
    DISCLAIMER,
    INSUFFICIENT_EVIDENCE_COPY,
    OUT_OF_CORPUS_COPY,
    REFUSAL_COPY,
    ChatMessage,
    Citation,
    Confidence,
    EvidenceQuote,
    InputClass,
    Outcome,
    TurnRequest,
    TurnResult,
)
from clinical_rag.generate.prompts import grounded_answer_messages
from clinical_rag.generate.retrieve_path import Retriever, retrieve_with_weak_path
from clinical_rag.schemas import SmokeHit

_REC_OPEN = "<<<RECOMMENDATION>>>"
_EV_OPEN = "<<<EVIDENCE>>>"
_CIT_OPEN = "<<<CITATIONS>>>"
_CONF_OPEN = "<<<CONFIDENCE>>>"


class StreamEventKind(str, Enum):
    status = "status"
    recommendation_delta = "recommendation_delta"
    result = "result"


@dataclass
class StreamEvent:
    kind: StreamEventKind
    text: str = ""
    result: TurnResult | None = None


def _hit_index_by_chunk(hits: list[SmokeHit]) -> dict[str, int]:
    return {h.chunk_id: i for i, h in enumerate(hits)}


def _citations_from_hits(hits: list[SmokeHit]) -> list[Citation]:
    return [
        Citation(
            document_name=h.document_name,
            section_title=h.section_title,
            page_number=h.page_number,
            chunk_id=h.chunk_id,
            hit_index=i,
        )
        for i, h in enumerate(hits)
    ]


def bind_citations(
    raw_citations: list[dict[str, Any]] | list[Citation],
    hits: list[SmokeHit],
) -> list[Citation]:
    """Keep only citations whose chunk_id appears in this turn's hits."""
    by_id = _hit_index_by_chunk(hits)
    out: list[Citation] = []
    seen: set[str] = set()
    for item in raw_citations:
        if isinstance(item, Citation):
            chunk_id = item.chunk_id
            hit_index = item.hit_index
        else:
            chunk_id = str(item.get("chunk_id") or "").strip()
            hit_index = item.get("hit_index")
            if hit_index is None and item.get("n") is not None:
                try:
                    n = int(item["n"])
                    if 1 <= n <= len(hits):
                        chunk_id = chunk_id or hits[n - 1].chunk_id
                        hit_index = n - 1
                except (TypeError, ValueError):
                    pass
        if not chunk_id or chunk_id not in by_id or chunk_id in seen:
            continue
        hit = hits[by_id[chunk_id]]
        out.append(
            Citation(
                document_name=hit.document_name,
                section_title=hit.section_title,
                page_number=hit.page_number,
                chunk_id=chunk_id,
                hit_index=by_id[chunk_id],
            )
        )
        seen.add(chunk_id)
    return out


def _parse_confidence(raw: str) -> Confidence:
    key = raw.strip().lower().split()[0] if raw.strip() else ""
    for c in Confidence:
        if c.value == key:
            return c
    return Confidence.low


def parse_tagged_answer(text: str) -> dict[str, str]:
    """Split tagged model output into section bodies."""
    sections: dict[str, str] = {
        "recommendation": "",
        "evidence": "",
        "citations": "",
        "confidence": "",
    }
    if not text:
        return sections
    pattern = re.compile(
        r"<<<(RECOMMENDATION|EVIDENCE|CITATIONS|CONFIDENCE)>>>\s*",
        re.I,
    )
    parts = pattern.split(text)
    # parts: [preamble, TAG, body, TAG, body, ...]
    i = 1
    while i + 1 < len(parts):
        tag = parts[i].upper()
        body = parts[i + 1]
        # Trim until next tag leftover handled by split
        key = {
            "RECOMMENDATION": "recommendation",
            "EVIDENCE": "evidence",
            "CITATIONS": "citations",
            "CONFIDENCE": "confidence",
        }.get(tag)
        if key:
            sections[key] = body.strip()
        i += 2
    if not sections["recommendation"] and _REC_OPEN not in text.upper().replace(" ", ""):
        # Soft fallback: entire text as recommendation if no tags
        if "<<<" not in text:
            sections["recommendation"] = text.strip()
    return sections


def _parse_evidence_lines(body: str, hits: list[SmokeHit]) -> list[EvidenceQuote]:
    by_id = _hit_index_by_chunk(hits)
    quotes: list[EvidenceQuote] = []
    for line in body.splitlines():
        line = line.strip().lstrip("-• ").strip()
        if not line:
            continue
        chunk_id = ""
        hit_index = None
        m = re.search(r"\[(\d+)\]", line)
        if m:
            n = int(m.group(1))
            if 1 <= n <= len(hits):
                hit_index = n - 1
                chunk_id = hits[hit_index].chunk_id
        cm = re.search(r"chunk_id\s*=\s*([^\s\]]+)", line, re.I)
        if cm:
            cid = cm.group(1).strip()
            # Prefer an explicit id only when it matches a hit; otherwise keep [n].
            if cid in by_id:
                chunk_id = cid
                hit_index = by_id[cid]
            elif not chunk_id:
                chunk_id = cid
        quote = re.sub(r"^\s*\"|\"\s*$", "", line)
        quote = re.sub(r"\s*\[\d+\]\s*$", "", quote).strip().strip('"')
        if chunk_id and chunk_id in by_id:
            quotes.append(
                EvidenceQuote(text=quote, chunk_id=chunk_id, hit_index=by_id[chunk_id])
            )
    return quotes


def _parse_citation_raw(
    body: str, hits: list[SmokeHit], *, strict: bool = False
) -> list[dict[str, Any]]:
    """Collect model citation attempts.

    ``strict=False`` (product bind): ``[n]`` wins over a bad ``chunk_id=`` so the
    answer still cites the right window.

    ``strict=True`` (citation accuracy): an explicit ``chunk_id=`` that is not in
    this turn's hits counts as a failed proposal even when ``[n]`` is valid.
    """
    by_id = _hit_index_by_chunk(hits)
    raw: list[dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        entry: dict[str, Any] = {}
        m = re.search(r"\[(\d+)\]", line)
        if m:
            entry["n"] = int(m.group(1))
        cm = re.search(r"chunk_id\s*=\s*([^\s\]]+)", line, re.I)
        explicit = cm.group(1).strip() if cm else ""
        if explicit:
            if strict:
                # Score the literal id the model wrote.
                entry["chunk_id"] = explicit
            elif explicit in by_id:
                entry["chunk_id"] = explicit
            elif m and 1 <= int(m.group(1)) <= len(hits):
                entry["chunk_id"] = hits[int(m.group(1)) - 1].chunk_id
            else:
                entry["chunk_id"] = explicit
        elif m and 1 <= int(m.group(1)) <= len(hits):
            entry["chunk_id"] = hits[int(m.group(1)) - 1].chunk_id
        if entry.get("chunk_id") or entry.get("n"):
            raw.append(entry)
    return raw


def _parse_citation_lines(body: str, hits: list[SmokeHit]) -> list[Citation]:
    return bind_citations(_parse_citation_raw(body, hits), hits)


def _unique_citation_attempts(
    raw: list[dict[str, Any]], hits: list[SmokeHit]
) -> list[dict[str, Any]]:
    """Dedupe proposed citation lines by resolved chunk_id / [n] key."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        chunk_id = str(item.get("chunk_id") or "").strip()
        n = item.get("n")
        if not chunk_id and n is not None:
            try:
                ni = int(n)
                if 1 <= ni <= len(hits):
                    chunk_id = hits[ni - 1].chunk_id
            except (TypeError, ValueError):
                pass
        key = chunk_id or (f"n:{n}" if n is not None else "")
        if not key or key in seen:
            continue
        seen.add(key)
        entry = dict(item)
        if chunk_id:
            entry["chunk_id"] = chunk_id
        unique.append(entry)
    return unique


def result_from_tagged(
    text: str,
    hits: list[SmokeHit],
    *,
    input_class: InputClass,
    retrieval_query: str,
    provider_used: str,
    prompt_k: int,
) -> TurnResult | None:
    """Parse tagged output. None = unparseable.

    If the model returns a recommendation but citations fail to parse/bind,
    auto-bind citations from the retrieved window. Case C (insufficient /
    weak retrieval) is only produced by the weak-score gate, not here.
    """
    has_tags = any(
        tag in text.upper() for tag in (_REC_OPEN, _EV_OPEN, _CIT_OPEN, _CONF_OPEN)
    )
    sections = parse_tagged_answer(text)
    window = hits[:prompt_k]
    # Lenient parse for the answer panel ([n] can rescue a bad chunk_id=).
    raw_cites = _unique_citation_attempts(
        _parse_citation_raw(sections["citations"], window, strict=False), window
    )
    citations = bind_citations(raw_cites, window)
    # Strict parse for citation_accuracy (literal chunk_id= must be in hits).
    raw_for_acc = _unique_citation_attempts(
        _parse_citation_raw(sections["citations"], window, strict=True), window
    )
    citations_proposed = len(raw_for_acc)
    citations_bound = len(bind_citations(raw_for_acc, window))
    citations_auto_filled = False

    evidence = _parse_evidence_lines(sections["evidence"], window)
    recommendation = sections["recommendation"].strip() or None
    confidence = _parse_confidence(sections["confidence"])

    # Evidence [n] markers can recover citations when the CITATIONS block is empty.
    if not citations and evidence:
        evidence_raw = _unique_citation_attempts(
            [{"chunk_id": q.chunk_id} for q in evidence if q.chunk_id],
            window,
        )
        citations = bind_citations(evidence_raw, window)
        if citations_proposed == 0:
            citations_proposed = len(evidence_raw)
            citations_bound = len(citations)

    if recommendation and not citations and window:
        citations = _citations_from_hits(window)
        citations_auto_filled = True
        citations_proposed = 0
        citations_bound = 0
        if confidence in (Confidence.high, Confidence.medium):
            confidence = Confidence.low
        if not evidence:
            evidence = [
                EvidenceQuote(
                    text=(h.text[:280] + ("…" if len(h.text) > 280 else "")),
                    chunk_id=h.chunk_id,
                    hit_index=i,
                )
                for i, h in enumerate(window)
            ]

    if not has_tags and not recommendation:
        return None

    if not recommendation and not citations:
        return None

    return TurnResult(
        recommendation=recommendation,
        evidence=evidence,
        citations=citations,
        confidence=confidence if citations else Confidence.insufficient,
        outcome=Outcome.success if citations else Outcome.complex,
        disclaimer=DISCLAIMER,
        hits=window,
        provider_used=provider_used,
        retrieval_query=retrieval_query,
        input_class=input_class,
        insufficient_evidence=False,
        citations_proposed=citations_proposed,
        citations_bound=citations_bound,
        citations_auto_filled=citations_auto_filled,
    )


def extractive_result(
    hits: list[SmokeHit],
    *,
    input_class: InputClass,
    retrieval_query: str,
    prompt_k: int,
    reason: str = "extractive",
) -> TurnResult:
    window = hits[:prompt_k]
    evidence = [
        EvidenceQuote(
            text=(h.text[:280] + ("…" if len(h.text) > 280 else "")),
            chunk_id=h.chunk_id,
            hit_index=i,
        )
        for i, h in enumerate(window)
    ]
    cites = _citations_from_hits(window)
    return TurnResult(
        recommendation=None,
        evidence=evidence,
        citations=cites,
        confidence=Confidence.insufficient,
        outcome=Outcome.complex,
        disclaimer=DISCLAIMER,
        hits=window,
        provider_used=reason,
        retrieval_query=retrieval_query,
        input_class=input_class,
        insufficient_evidence=True,
        citations_proposed=0,
        citations_bound=len(cites),
        citations_auto_filled=True,
    )


def case_c_result(
    hits: list[SmokeHit],
    *,
    input_class: InputClass,
    retrieval_query: str,
    prompt_k: int,
    out_of_corpus: bool = False,
) -> TurnResult:
    window = hits[:prompt_k]
    cites = _citations_from_hits(window)
    return TurnResult(
        recommendation=OUT_OF_CORPUS_COPY if out_of_corpus else INSUFFICIENT_EVIDENCE_COPY,
        evidence=[
            EvidenceQuote(
                text=(h.text[:280] + ("…" if len(h.text) > 280 else "")),
                chunk_id=h.chunk_id,
                hit_index=i,
            )
            for i, h in enumerate(window)
        ],
        citations=cites,
        confidence=Confidence.insufficient,
        outcome=Outcome.complex,
        disclaimer=DISCLAIMER,
        hits=window,
        provider_used="",
        retrieval_query=retrieval_query,
        input_class=input_class,
        insufficient_evidence=True,
        out_of_corpus=out_of_corpus,
        citations_proposed=0,
        citations_bound=len(cites),
        citations_auto_filled=True,
    )


def refusal_result(input_class: InputClass = InputClass.refuse) -> TurnResult:
    return TurnResult(
        recommendation=REFUSAL_COPY,
        evidence=[],
        citations=[],
        confidence=Confidence.insufficient,
        outcome=Outcome.refusal,
        disclaimer=DISCLAIMER,
        hits=[],
        provider_used="",
        retrieval_query="",
        input_class=input_class,
        insufficient_evidence=False,
        citations_proposed=0,
        citations_bound=0,
        citations_auto_filled=False,
    )


def _next_section_index(text: str) -> int:
    upper = text.upper()
    positions = [
        upper.find(tag)
        for tag in (_EV_OPEN, _CIT_OPEN, _CONF_OPEN)
        if upper.find(tag) >= 0
    ]
    return min(positions) if positions else -1


class GroundedEngine:
    """One-turn orchestration for the clinician product path."""

    def __init__(
        self,
        session: Retriever,
        llm: LlmClient | CascadeClient | None,
        *,
        settings: GenerationSettings | None = None,
    ) -> None:
        self.session = session
        self.llm = llm
        self.settings = settings or get_settings().generation

    def turn(self, request: TurnRequest) -> TurnResult:
        events = list(self.stream_turn(request))
        for ev in reversed(events):
            if ev.kind is StreamEventKind.result and ev.result is not None:
                return ev.result
        raise QueryError("GroundedEngine produced no TurnResult")

    def stream_turn(self, request: TurnRequest) -> Iterator[StreamEvent]:
        cfg = self.settings
        history = list(request.history or [])
        if cfg.memory_turns > 0:
            history = history[-(cfg.memory_turns * 2) :]

        yield StreamEvent(StreamEventKind.status, "classifying")
        input_class = classify_input(request.message, self.llm)
        if input_class is InputClass.refuse:
            yield StreamEvent(StreamEventKind.result, result=refusal_result(input_class))
            return

        yield StreamEvent(StreamEventKind.status, "retrieving")
        hits, retrieval_query, still_weak = retrieve_with_weak_path(
            request.message,
            session=self.session,
            llm=self.llm,
            history=history,
            settings=cfg,
        )

        # Guideline / screening / risk-score questions are not this corpus's job.
        # Still show retrieved hits, but never generate a grounded recommendation.
        out_of_corpus = is_out_of_corpus(request.message)
        if still_weak or out_of_corpus:
            yield StreamEvent(
                StreamEventKind.status,
                "out_of_corpus" if out_of_corpus else "insufficient_evidence",
            )
            result = case_c_result(
                hits,
                input_class=input_class,
                retrieval_query=retrieval_query,
                prompt_k=cfg.prompt_k,
                out_of_corpus=out_of_corpus,
            )
            if result.recommendation:
                yield StreamEvent(
                    StreamEventKind.recommendation_delta, text=result.recommendation
                )
            yield StreamEvent(StreamEventKind.result, result=result)
            return

        yield StreamEvent(StreamEventKind.status, "generating")
        messages = grounded_answer_messages(
            request.message,
            hits,
            history=history,
            input_class=input_class,
            memory_turns=cfg.memory_turns,
            prompt_k=cfg.prompt_k,
        )

        if self.llm is None:
            result = extractive_result(
                hits,
                input_class=input_class,
                retrieval_query=retrieval_query,
                prompt_k=cfg.prompt_k,
                reason="extractive",
            )
            yield StreamEvent(StreamEventKind.result, result=result)
            return

        provider = getattr(self.llm, "name", "") or ""
        full_text = ""
        try:
            # Prefer streaming for UI; fall back to complete.
            if hasattr(self.llm, "stream"):
                rec_buf = ""
                full_parts: list[str] = []
                rec_done = False
                for piece in self.llm.stream(messages):
                    full_parts.append(piece)
                    full_text = "".join(full_parts)
                    if not rec_done:
                        upper = full_text.upper()
                        if _REC_OPEN in upper:
                            idx = upper.find(_REC_OPEN)
                            after = full_text[idx + len(_REC_OPEN) :]
                            next_idx = _next_section_index(after)
                            if next_idx >= 0:
                                new_rec = after[:next_idx]
                                delta = new_rec[len(rec_buf) :]
                                if delta:
                                    yield StreamEvent(
                                        StreamEventKind.recommendation_delta, text=delta
                                    )
                                rec_buf = new_rec
                                rec_done = True
                            else:
                                delta = after[len(rec_buf) :]
                                if delta:
                                    yield StreamEvent(
                                        StreamEventKind.recommendation_delta, text=delta
                                    )
                                    rec_buf = after
                if getattr(self.llm, "last_provider", None):
                    provider = self.llm.last_provider  # type: ignore[attr-defined]
                elif getattr(self.llm, "name", None):
                    provider = str(self.llm.name)
            else:
                full_text = self.llm.complete(messages)
                if getattr(self.llm, "last_provider", None):
                    provider = self.llm.last_provider  # type: ignore[attr-defined]
        except Exception:
            result = extractive_result(
                hits,
                input_class=input_class,
                retrieval_query=retrieval_query,
                prompt_k=cfg.prompt_k,
                reason="extractive",
            )
            yield StreamEvent(StreamEventKind.result, result=result)
            return

        parsed = result_from_tagged(
            full_text,
            hits,
            input_class=input_class,
            retrieval_query=retrieval_query,
            provider_used=provider,
            prompt_k=cfg.prompt_k,
        )
        if parsed is None:
            # Parse failure: keep streamed recommendation only if hits can bind.
            sections = parse_tagged_answer(full_text)
            rec = sections["recommendation"].strip()
            window = hits[: cfg.prompt_k]
            if rec and window:
                parsed = TurnResult(
                    recommendation=rec,
                    evidence=[
                        EvidenceQuote(
                            text=(h.text[:200] + ("…" if len(h.text) > 200 else "")),
                            chunk_id=h.chunk_id,
                            hit_index=i,
                        )
                        for i, h in enumerate(window[:3])
                    ],
                    citations=_citations_from_hits(window),
                    confidence=Confidence.low,
                    outcome=Outcome.success,
                    disclaimer=DISCLAIMER,
                    hits=window,
                    provider_used=provider,
                    retrieval_query=retrieval_query,
                    input_class=input_class,
                    citations_proposed=0,
                    citations_bound=len(window),
                    citations_auto_filled=True,
                )
            else:
                parsed = extractive_result(
                    hits,
                    input_class=input_class,
                    retrieval_query=retrieval_query,
                    prompt_k=cfg.prompt_k,
                    reason="extractive",
                )

        yield StreamEvent(StreamEventKind.result, result=parsed)

"""Weak-score gate, rewrite, multi-query expand, max-score fuse (no HyDE / no RRF)."""

from __future__ import annotations

from typing import Protocol

from clinical_rag.adapters.llms import LlmClient, parse_json_object
from clinical_rag.config import GenerationSettings
from clinical_rag.generate.models import ChatMessage
from clinical_rag.generate.prompts import expand_messages, rewrite_messages
from clinical_rag.schemas import SmokeHit


class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int | None = None) -> list[SmokeHit]: ...


def is_weak(
    hits: list[SmokeHit],
    *,
    threshold: float,
) -> bool:
    if not hits:
        return True
    return float(hits[0].score) < threshold


def fuse_by_max_score(
    hit_lists: list[list[SmokeHit]],
    *,
    top_k: int = 10,
) -> list[SmokeHit]:
    """Dedupe by chunk_id; keep the hit with the highest CE/rerank score."""
    best: dict[str, SmokeHit] = {}
    for hits in hit_lists:
        for hit in hits:
            prev = best.get(hit.chunk_id)
            if prev is None or hit.score > prev.score:
                best[hit.chunk_id] = hit
    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
    return ranked[:top_k]


def rewrite_standalone(
    message: str,
    history: list[ChatMessage],
    llm: LlmClient,
) -> str:
    raw = llm.complete(rewrite_messages(message, history), json_mode=True)
    data = parse_json_object(raw)
    q = str(data.get("query") or data.get("question") or "").strip()
    return q or message


def expand_queries(
    message: str,
    llm: LlmClient,
    *,
    n: int = 3,
) -> list[str]:
    raw = llm.complete(expand_messages(message, n=n), json_mode=True)
    data = parse_json_object(raw)
    variants = data.get("queries") or data.get("variants") or []
    out: list[str] = []
    if isinstance(variants, list):
        for item in variants:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
    if not out:
        out = [message]
    return out[:n]


def retrieve_with_weak_path(
    message: str,
    *,
    session: Retriever,
    llm: LlmClient | None,
    history: list[ChatMessage] | None = None,
    settings: GenerationSettings | None = None,
) -> tuple[list[SmokeHit], str, bool]:
    """Raw retrieve → optional rewrite → optional multi-query fuse.

    Returns (hits, retrieval_query_used, still_weak).
    """
    cfg = settings or GenerationSettings()
    history = history or []
    threshold = cfg.weak_score

    hits = session.retrieve(message)
    query_used = message
    if not is_weak(hits, threshold=threshold):
        return hits, query_used, False

    if llm is None:
        return hits, query_used, True

    try:
        rewritten = rewrite_standalone(message, history, llm)
    except Exception:
        rewritten = message
    if rewritten.strip() and rewritten.strip() != message.strip():
        hits = session.retrieve(rewritten)
        query_used = rewritten
        if not is_weak(hits, threshold=threshold):
            return hits, query_used, False

    try:
        variants = expand_queries(query_used, llm, n=cfg.multi_query_n)
    except Exception:
        return hits, query_used, True

    lists: list[list[SmokeHit]] = []
    for q in variants:
        try:
            lists.append(session.retrieve(q))
        except Exception:
            continue
    if not lists:
        return hits, query_used, True

    fused = fuse_by_max_score(lists, top_k=10)
    query_used = " | ".join(variants)
    still_weak = is_weak(fused, threshold=threshold)
    return fused, query_used, still_weak

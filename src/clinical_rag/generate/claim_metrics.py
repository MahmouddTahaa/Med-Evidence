"""Claim-level support metrics for grounded generation.

Faithfulness = mean over recommendation claims of max support score against
retrieved hit texts. Continuous in [0, 1]. Not a labeled bakeoff.

Primary scorer: the same MiniLM cross-encoder used for retrieval rerank
(already cached on the product machine). Optional MNLI models can be tried
first when available; lexical overlap is the last-resort fallback.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from clinical_rag.schemas import DEFAULT_RERANK_MODEL

# Optional true-NLI models (tried first; may be absent from cache / HF).
NLI_MODEL_CANDIDATES = (
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-deberta-v3-base",
)
DEFAULT_SUPPORT_MODEL = DEFAULT_RERANK_MODEL  # ms-marco-MiniLM-L-6-v2
# nli-deberta label order when Softmax is applied
_CONTRADICTION, _ENTAILMENT, _NEUTRAL = 0, 1, 2
DEFAULT_HARD_THRESHOLD = 0.5
_MIN_CLAIM_CHARS = 18


class EntailmentScorer(Protocol):
    def entailment_probs(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[float]:
        """For each hypothesis, return max P(support | premise) over premises."""
        ...


@dataclass
class ClaimScore:
    claim: str
    entailment_prob: float
    entailed: bool


@dataclass
class ClaimEntailmentReport:
    claims: list[ClaimScore] = field(default_factory=list)
    faithfulness: float | None = None
    entailment_rate: float | None = None
    scorer_name: str = ""
    note: str = ""


def split_claims(text: str) -> list[str]:
    """Split recommendation prose into claim-like units."""
    raw = (text or "").strip()
    if not raw:
        return []
    # Sentence / semicolon / newline boundaries.
    parts = re.split(r"(?<=[.!?])\s+|\n+|;\s+", raw)
    claims: list[str] = []
    for part in parts:
        claim = re.sub(r"\s+", " ", part).strip().strip("-•* ")
        if len(claim) < _MIN_CLAIM_CHARS:
            continue
        lower = claim.lower()
        if lower.startswith("educational use only"):
            continue
        if "not a diagnosis" in lower or "call local emergency" in lower:
            continue
        if lower.startswith("retrieved sources are too weak"):
            continue
        claims.append(claim)
    return claims


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class LexicalEntailmentScorer:
    """Token-overlap fallback when no cross-encoder is available."""

    name = "lexical"

    def entailment_probs(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[float]:
        prem_toks = [_tokens(p) for p in premises if p.strip()]
        if not prem_toks:
            return [0.0] * len(hypotheses)
        out: list[float] = []
        for hyp in hypotheses:
            ht = _tokens(hyp)
            if not ht:
                out.append(0.0)
                continue
            best = 0.0
            for pt in prem_toks:
                best = max(best, len(ht & pt) / len(ht))
            out.append(float(best))
        return out


class CrossEncoderNliScorer:
    """True NLI CrossEncoder (3-way Softmax)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        hard_threshold: float = DEFAULT_HARD_THRESHOLD,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_id = model_id
        self.hard_threshold = hard_threshold
        self.name = model_id
        self._model = CrossEncoder(model_id, device=device)
        self._mode = "nli"

    def entailment_probs(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[float]:
        return _max_pair_scores(
            self._model,
            premises,
            hypotheses,
            mode="nli",
        )


class CrossEncoderSupportScorer:
    """MS MARCO CE relevance → sigmoid support in (0, 1). Uses product rerank model."""

    def __init__(
        self,
        model_id: str = DEFAULT_SUPPORT_MODEL,
        *,
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_id = model_id
        self.name = f"{model_id} (claim-support)"
        self._model = CrossEncoder(model_id, device=device)

    def entailment_probs(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[float]:
        return _max_pair_scores(
            self._model,
            premises,
            hypotheses,
            mode="support",
        )


def _max_pair_scores(
    model: object,
    premises: list[str],
    hypotheses: list[str],
    *,
    mode: str,
) -> list[float]:
    clean_premises = [p.strip() for p in premises if p and p.strip()]
    if not clean_premises or not hypotheses:
        return [0.0] * len(hypotheses)

    pairs: list[tuple[str, str]] = []
    owners: list[int] = []
    for hi, hyp in enumerate(hypotheses):
        h = (hyp or "").strip()
        if not h:
            continue
        for prem in clean_premises:
            # CE convention: (query/claim, passage) for MS MARCO; (premise, hyp) for NLI
            if mode == "nli":
                pairs.append((prem, h))
            else:
                pairs.append((h, prem))
            owners.append(hi)

    if not pairs:
        return [0.0] * len(hypotheses)

    scores = model.predict(pairs, apply_softmax=(mode == "nli"), show_progress_bar=False)
    arr = np.asarray(scores, dtype=np.float64)
    if mode == "nli":
        if arr.ndim == 1:
            ent = np.clip(arr, 0.0, 1.0)
        else:
            ent = arr[:, _ENTAILMENT]
    else:
        # Raw logits → sigmoid
        ent = np.vectorize(_sigmoid, otypes=[float])(arr.reshape(-1))

    best = [0.0] * len(hypotheses)
    for owner, p in zip(owners, ent, strict=True):
        best[owner] = max(best[owner], float(p))
    return best


def build_entailment_scorer(
    *,
    prefer_nli: bool = False,
    device: str = "cpu",
) -> EntailmentScorer:
    """Default: product MiniLM CE claim-support (cached, calibrated for passage support).

    Optional MNLI models are stricter logical entailment and under-score paraphrased
    multi-source recommendations; enable with prefer_nli=True when available.
    """
    if prefer_nli:
        for mid in NLI_MODEL_CANDIDATES:
            try:
                return CrossEncoderNliScorer(mid, device=device)
            except Exception:
                continue
    try:
        return CrossEncoderSupportScorer(DEFAULT_SUPPORT_MODEL, device=device)
    except Exception:
        return LexicalEntailmentScorer()


def _prepare_premises(premises: list[str], *, max_each: int = 1200, max_combined: int = 3500) -> list[str]:
    """Per-hit windows plus a combined context block (RAG-style grounding)."""
    cleaned = [(p or "").strip() for p in premises if (p or "").strip()]
    if not cleaned:
        return []
    truncated = [p[:max_each] for p in cleaned]
    combined = "\n\n".join(truncated)
    if len(combined) > max_combined:
        combined = combined[:max_combined]
    # Combined first so max() can pick set-level support for summary claims.
    out = [combined]
    for p in truncated:
        if p not in out:
            out.append(p)
    return out


def score_claims(
    recommendation: str,
    premises: list[str],
    scorer: EntailmentScorer,
    *,
    hard_threshold: float = DEFAULT_HARD_THRESHOLD,
) -> ClaimEntailmentReport:
    claims = split_claims(recommendation)
    if not claims:
        return ClaimEntailmentReport(
            note="No claim-sized sentences in recommendation.",
            scorer_name=getattr(scorer, "name", scorer.__class__.__name__),
        )
    prepared = _prepare_premises(premises)
    if not prepared:
        return ClaimEntailmentReport(
            claims=[
                ClaimScore(claim=c, entailment_prob=0.0, entailed=False) for c in claims
            ],
            faithfulness=0.0,
            entailment_rate=0.0,
            scorer_name=getattr(scorer, "name", scorer.__class__.__name__),
            note="No retrieved premises — all claims unscored as unsupported.",
        )

    probs = scorer.entailment_probs(prepared, claims)
    scored = [
        ClaimScore(
            claim=c,
            entailment_prob=float(p),
            entailed=float(p) >= hard_threshold,
        )
        for c, p in zip(claims, probs, strict=True)
    ]
    faithfulness = sum(s.entailment_prob for s in scored) / len(scored)
    rate = sum(1 for s in scored if s.entailed) / len(scored)
    return ClaimEntailmentReport(
        claims=scored,
        faithfulness=faithfulness,
        entailment_rate=rate,
        scorer_name=getattr(scorer, "name", scorer.__class__.__name__),
        note=f"{len(scored)} claims · hard rate @ {hard_threshold:.2f}",
    )

def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}

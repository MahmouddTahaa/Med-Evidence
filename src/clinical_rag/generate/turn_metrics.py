"""Per-turn generation metric helpers (UI + tests).

Faithfulness stays in claim_metrics (claim support vs passages).
This module owns citation accuracy and session-average filters.
"""

from __future__ import annotations

from clinical_rag.generate.models import Outcome, TurnResult


def citation_accuracy(result: TurnResult) -> float | None:
    """Share of *model-proposed* citation attempts that bind to this turn's hits.

    - ``citations_proposed`` = unique CITATIONS lines (or evidence-recovered ids) before bind
    - ``citations_bound`` = survivors after ``bind_citations``
    - Auto-filled Case C / extractive (engine invented cites) → ``None`` (N/A)
    - Refusal / no proposals → ``None``
    """
    fields = getattr(type(result), "model_fields", {})
    if "citations_proposed" not in fields:
        hit_ids = {h.chunk_id for h in result.hits}
        cites = list(result.citations)
        if not cites:
            return None
        return sum(1 for c in cites if c.chunk_id in hit_ids) / len(cites)

    if result.outcome is Outcome.refusal:
        return None
    if bool(result.citations_auto_filled):
        return None

    proposed = int(result.citations_proposed or 0)
    bound = int(result.citations_bound or 0)
    if proposed <= 0:
        return None
    return max(0.0, min(1.0, bound / proposed))


def include_in_session_average(metrics: dict) -> bool:
    """Session means: grounded answers only — skip refusal and Case C / extractive."""
    if not isinstance(metrics, dict):
        return False
    if metrics.get("outcome") == "refusal":
        return False
    if metrics.get("insufficient_evidence"):
        return False
    if metrics.get("exclude_from_session"):
        return False
    return True

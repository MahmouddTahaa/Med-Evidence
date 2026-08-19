from __future__ import annotations

from clinical_rag.errors import IngestError
from clinical_rag.schemas import SmokeHit


def weighted_rrf(
    ranked_lists: list[list[SmokeHit]],
    weights: list[float],
    *,
    k: int = 60,
) -> list[SmokeHit]:
    """Fuse ranked lists with weighted Reciprocal Rank Fusion.

    score(d) = sum_i  w_i / (k + rank_i(d))
    Ranks are 1-based. Chunks missing from a list contribute nothing from that list.
    Citation fields are taken from the first occurrence of each chunk_id.
    """
    if len(ranked_lists) != len(weights):
        raise IngestError("weighted_rrf: ranked_lists and weights must have the same length")
    if k < 1:
        raise IngestError("rrf_k must be >= 1")
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        raise IngestError(f"RRF weights must sum to 1.0 (got {weight_sum})")

    scores: dict[str, float] = {}
    first: dict[str, SmokeHit] = {}
    for hits, weight in zip(ranked_lists, weights, strict=True):
        if weight <= 0:
            continue
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk_id
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            if cid not in first:
                first[cid] = hit

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[SmokeHit] = []
    for cid, score in ordered:
        base = first[cid]
        out.append(
            SmokeHit(
                score=round(score, 6),
                text=base.text,
                document_name=base.document_name,
                section_title=base.section_title,
                page_number=base.page_number,
                chunk_id=cid,
                extraction_method=base.extraction_method,
                source_url=base.source_url,
                token_count=base.token_count,
            )
        )
    return out

import pytest
from pydantic import ValidationError

from clinical_rag.errors import IngestError
from clinical_rag.retrieval.rrf import weighted_rrf
from clinical_rag.schemas import RetrievalConfig, SmokeHit


def _hit(cid: str, score: float = 1.0) -> SmokeHit:
    return SmokeHit(
        score=score,
        text=f"text-{cid}",
        document_name="doc",
        section_title="sec",
        page_number=1,
        chunk_id=cid,
    )


def test_weighted_rrf_prefers_heavier_list():
    sem = [_hit("a"), _hit("b"), _hit("c")]
    kw = [_hit("c"), _hit("a"), _hit("d")]
    # Heavy semantic: a should beat c
    fused = weighted_rrf([sem, kw], [0.9, 0.1], k=60)
    ids = [h.chunk_id for h in fused]
    assert ids[0] == "a"
    assert "d" in ids  # present only in keyword list


def test_weighted_rrf_missing_list_docs_appear():
    sem = [_hit("only-sem")]
    kw = [_hit("only-kw")]
    fused = weighted_rrf([sem, kw], [0.5, 0.5], k=60)
    ids = {h.chunk_id for h in fused}
    assert ids == {"only-sem", "only-kw"}


def test_weighted_rrf_weights_must_sum():
    with pytest.raises(IngestError, match="sum to 1"):
        weighted_rrf([[_hit("a")]], [0.5])


def test_retrieval_config_weights_validator():
    with pytest.raises(ValidationError):
        RetrievalConfig(semantic_weight=0.8, keyword_weight=0.5)

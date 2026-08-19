from clinical_rag.eval.grid import retrieval_eval_grid, retrieval_label
from clinical_rag.retrieval.siblings import fill_section_siblings
from clinical_rag.schemas import RetrievalConfig, RetrievalMode, SmokeHit


def _hit(chunk_id: str, *, doc: str, section: str, score: float = 1.0) -> SmokeHit:
    return SmokeHit(
        score=score,
        text=f"text-{chunk_id}",
        document_name=doc,
        section_title=section,
        page_number=1,
        chunk_id=chunk_id,
    )


def _row(chunk_id: str, *, doc: str, section: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"body {chunk_id}",
        "document_name": doc,
        "section_title": section,
        "page_number": 1,
        "extraction_method": "n/a",
        "token_count": 10,
    }


def test_freezes_rank_one_and_packs_same_section():
    hits = [
        _hit("a-0002", doc="adalimumab", section="Administration", score=9.0),
        _hit("other-ci", doc="ampicillin", section="Contraindications", score=4.0),
        _hit("a-0001", doc="adalimumab", section="Indications", score=3.0),
    ]
    chunks = [
        _row("a-0001", doc="adalimumab", section="Indications"),
        _row("a-0002", doc="adalimumab", section="Administration"),
        _row("a-0003", doc="adalimumab", section="Administration"),
        _row("a-0004", doc="adalimumab", section="Administration"),
        _row("other-ci", doc="ampicillin", section="Contraindications"),
    ]
    out = fill_section_siblings(hits, chunks, top_k=4)
    assert [h.chunk_id for h in out] == ["a-0002", "a-0003", "a-0004", "other-ci"]
    assert out[0].score == 9.0


def test_injects_siblings_absent_from_the_ranked_list():
    hits = [_hit("a-0002", doc="adalimumab", section="Administration")]
    chunks = [
        _row("a-0002", doc="adalimumab", section="Administration"),
        _row("a-0003", doc="adalimumab", section="Administration"),
    ]
    out = fill_section_siblings(hits, chunks, top_k=2)
    assert [h.chunk_id for h in out] == ["a-0002", "a-0003"]
    assert out[1].text == "body a-0003"


def test_section_match_is_case_insensitive():
    hits = [_hit("x-1", doc="doc", section="Mechanism of Action")]
    chunks = [
        _row("x-1", doc="doc", section="Mechanism of Action"),
        _row("x-2", doc="doc", section="mechanism of action"),
    ]
    out = fill_section_siblings(hits, chunks, top_k=2)
    assert [h.chunk_id for h in out] == ["x-1", "x-2"]


def test_skips_fill_when_section_title_missing():
    hits = [
        _hit("x-1", doc="doc", section=""),
        _hit("y-1", doc="other", section="Indications"),
    ]
    chunks = [_row("x-2", doc="doc", section="")]
    out = fill_section_siblings(hits, chunks, top_k=2)
    assert [h.chunk_id for h in out] == ["x-1", "y-1"]


def test_empty_hits():
    assert fill_section_siblings([], [], top_k=5) == []


def test_default_config_and_grid_leave_sibling_fill_off():
    assert RetrievalConfig().sibling_fill is False
    grid = retrieval_eval_grid()
    assert len(grid) == 7
    assert all(not cfg.sibling_fill for cfg in grid)
    assert all(not cfg.parent_child for cfg in grid)
    assert retrieval_label(grid[-1]) == "hybrid/tfidf+rerank"


def test_retrieval_label_suffix():
    cfg = RetrievalConfig(
        mode=RetrievalMode.hybrid,
        rerank=True,
        sibling_fill=True,
    )
    assert retrieval_label(cfg) == "hybrid/bm25+rerank+siblings"

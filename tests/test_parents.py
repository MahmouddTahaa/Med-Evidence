from clinical_rag.eval.grid import retrieval_eval_grid, retrieval_label
from clinical_rag.retrieval.parents import expand_parent_children
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


def _row(
    chunk_id: str,
    *,
    doc: str,
    section: str,
    parent: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"body {chunk_id}",
        "document_name": doc,
        "section_title": section,
        "parent_chunk_id": parent or "",
        "page_number": 1,
        "extraction_method": "n/a",
        "token_count": 10,
    }


def test_expands_section_siblings_into_pool_without_freezing_order():
    hits = [
        _hit("a-0003", doc="adalimumab", section="Administration", score=5.0),
        _hit("other", doc="other", section="Indications", score=4.0),
    ]
    chunks = [
        _row("a-0002", doc="adalimumab", section="Administration"),
        _row("a-0003", doc="adalimumab", section="Administration"),
        _row("a-0004", doc="adalimumab", section="Administration"),
        _row("other", doc="other", section="Indications"),
    ]
    out = expand_parent_children(hits, chunks, limit=5)
    assert [h.chunk_id for h in out] == ["a-0002", "a-0003", "a-0004", "other"]


def test_expands_hierarchical_parent_chunk_id():
    hits = [_hit("c-2", doc="doc", section="Treatment", score=3.0)]
    chunks = [
        _row("c-1", doc="doc", section="Treatment", parent="doc-parent-0001"),
        _row("c-2", doc="doc", section="Treatment", parent="doc-parent-0001"),
        _row("c-3", doc="doc", section="Follow-up", parent="doc-parent-0002"),
    ]
    out = expand_parent_children(hits, chunks, limit=5)
    assert [h.chunk_id for h in out] == ["c-1", "c-2"]


def test_limit_caps_expansion():
    hits = [_hit("a-0001", doc="d", section="S")]
    chunks = [_row(f"a-000{i}", doc="d", section="S") for i in range(1, 6)]
    out = expand_parent_children(hits, chunks, limit=3)
    assert len(out) == 3


def test_grid_leaves_parent_child_off():
    assert RetrievalConfig().parent_child is False
    grid = retrieval_eval_grid()
    assert len(grid) == 7
    assert all(not cfg.parent_child for cfg in grid)
    cfg = RetrievalConfig(mode=RetrievalMode.hybrid, rerank=True, parent_child=True)
    assert retrieval_label(cfg) == "hybrid/bm25+rerank+parent_child"

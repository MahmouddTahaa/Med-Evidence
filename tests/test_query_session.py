from pathlib import Path

from clinical_rag.query.context import citation_blocks
from clinical_rag.query.session import RetrievalSession
from clinical_rag.retrieval.sparse import SparseIndex
from clinical_rag.schemas import KeywordMethod, RetrievalConfig, RetrievalMode, SmokeHit
from clinical_rag.stack import FrozenStack
from tests.helpers import HashEmbedder
from tests.test_retrieve import StubReranker, _ingest_sample


def test_citation_blocks_include_metadata():
    hits = [
        SmokeHit(
            score=0.9,
            text="Do not give in pregnancy.",
            document_name="article-1",
            section_title="Contraindications",
            page_number=3,
            chunk_id="c1",
        )
    ]
    block = citation_blocks(hits)
    assert block.startswith("[1] article-1 · Contraindications · page 3 · c1")
    assert "Do not give in pregnancy." in block


def test_citation_blocks_empty():
    assert citation_blocks([]) == ""


def test_retrieval_session_hybrid(tmp_path: Path):
    report, _chunks, _persist, jobs = _ingest_sample(tmp_path)
    stack = FrozenStack(
        chunk={"strategy_id": "passthrough"},
        embed={"model_id": "hash-test"},
        vector_store="chroma",
        retrieval=RetrievalConfig(
            mode=RetrievalMode.hybrid,
            top_k=3,
            fetch_k=5,
            semantic_weight=0.7,
            keyword_weight=0.3,
            rerank=True,
            rerank_top_n=5,
        ),
    )
    session = RetrievalSession.open(
        job_id=report.job_id,
        stack=stack,
        jobs_dir=jobs,
        embedder=HashEmbedder(),
        reranker=StubReranker(),
    )
    hits = session.retrieve("asthma rescue")
    assert hits
    assert hits[0].chunk_id
    assert session.collection == report.collection_name
    assert session._sparse is not None
    assert isinstance(session._sparse, SparseIndex)
    assert session._sparse.method is KeywordMethod.bm25


def test_retrieval_session_top_k_override(tmp_path: Path):
    report, chunks, _persist, jobs = _ingest_sample(tmp_path)
    stack = FrozenStack(
        chunk={"strategy_id": "passthrough"},
        embed={"model_id": "hash-test"},
        vector_store="chroma",
        retrieval=RetrievalConfig(mode=RetrievalMode.dense, top_k=5),
    )
    session = RetrievalSession.open(
        job_id=report.job_id,
        stack=stack,
        jobs_dir=jobs,
        embedder=HashEmbedder(),
    )
    hits = session.retrieve(chunks[0].text, top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == chunks[0].chunk_id

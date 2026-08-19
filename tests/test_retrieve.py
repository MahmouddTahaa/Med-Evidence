import json
from pathlib import Path

from clinical_rag.pipeline.ingest import run_ingest
from clinical_rag.retrieval.pipeline import run_retrieve
from clinical_rag.retrieval.sparse import SparseIndex
from clinical_rag.schemas import (
    ChunkConfig,
    EmbedConfig,
    KeywordMethod,
    IngestJobConfig,
    ParserConfig,
    RetrievalConfig,
    RetrievalMode,
    SmokeHit,
    StrategyId,
)
from tests.helpers import HashEmbedder, legal_ok, raw_doc
from tests.test_ingest_smoke import cfg_chroma


class CountingEmbedder(HashEmbedder):
    def __init__(self):
        self.encode_calls = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.encode_calls += 1
        return super().encode(texts)


class StubReranker:
    model_id = "stub-reranker"

    def rerank(self, query: str, hits: list[SmokeHit]) -> list[SmokeHit]:
        # Reverse order to prove rerank applied
        flipped = list(reversed(hits))
        out = []
        for i, h in enumerate(flipped):
            out.append(
                SmokeHit(
                    score=float(len(flipped) - i),
                    text=h.text,
                    document_name=h.document_name,
                    section_title=h.section_title,
                    page_number=h.page_number,
                    chunk_id=h.chunk_id,
                    extraction_method=h.extraction_method,
                    source_url=h.source_url,
                    token_count=h.token_count,
                )
            )
        return out


def _ingest_sample(tmp_path: Path):
    src = Path("data/guidelines/sample_asthma_prechunked.json")
    payload = json.loads(src.read_text(encoding="utf-8"))
    path = tmp_path / "asthma.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    persist = tmp_path / "chroma"
    jobs = tmp_path / "jobs"
    raw = raw_doc(path, doc_id="asthma")
    raw.legal = legal_ok()
    cfg = IngestJobConfig(
        corpus_id="demo",
        job_id="job-ret",
        files=[raw],
        parser=ParserConfig(),
        chunk=ChunkConfig(strategy_id=StrategyId.passthrough),
        embed=EmbedConfig(model_id="hash-test"),
        chroma=cfg_chroma(persist),
    )
    report, chunks = run_ingest(cfg, embedder=HashEmbedder(), jobs_dir=jobs)
    return report, chunks, persist, jobs


def test_keyword_only_skips_embedder(tmp_path: Path):
    report, _chunks, persist, jobs = _ingest_sample(tmp_path)
    embedder = CountingEmbedder()
    chunks_path = jobs / report.job_id / "chunks.json"
    sparse = SparseIndex.from_chunks(
        json.loads(chunks_path.read_text(encoding="utf-8")),
        method=KeywordMethod.bm25,
    )
    hits = run_retrieve(
        query="rescue inhaler",
        retrieval=RetrievalConfig(mode=RetrievalMode.keyword, top_k=3),
        collection=report.collection_name,
        sparse_index=sparse,
        embedder=embedder,
        persist_dir=str(persist),
    )
    assert embedder.encode_calls == 0
    assert hits
    assert hits[0].chunk_id


def test_semantic_retrieve(tmp_path: Path):
    report, chunks, persist, jobs = _ingest_sample(tmp_path)
    hits = run_retrieve(
        query=chunks[0].text,
        retrieval=RetrievalConfig(mode=RetrievalMode.dense, top_k=3),
        collection=report.collection_name,
        embedder=HashEmbedder(),
        index_model_id=report.embed_model_id,
        persist_dir=str(persist),
    )
    assert hits[0].chunk_id == chunks[0].chunk_id


def test_hybrid_with_stub_reranker(tmp_path: Path):
    report, _chunks, persist, jobs = _ingest_sample(tmp_path)
    chunks_path = jobs / report.job_id / "chunks.json"
    rows = json.loads(chunks_path.read_text(encoding="utf-8"))
    sparse = SparseIndex.from_chunks(rows, method=KeywordMethod.bm25)
    cfg = RetrievalConfig(
        mode=RetrievalMode.hybrid,
        top_k=3,
        fetch_k=5,
        semantic_weight=0.7,
        keyword_weight=0.3,
        rerank=True,
        rerank_top_n=5,
    )
    without = run_retrieve(
        query="asthma rescue",
        retrieval=cfg.model_copy(update={"rerank": False}),
        collection=report.collection_name,
        sparse_index=sparse,
        embedder=HashEmbedder(),
        index_model_id=report.embed_model_id,
        persist_dir=str(persist),
    )
    with_rr = run_retrieve(
        query="asthma rescue",
        retrieval=cfg,
        collection=report.collection_name,
        sparse_index=sparse,
        embedder=HashEmbedder(),
        index_model_id=report.embed_model_id,
        persist_dir=str(persist),
        reranker=StubReranker(),
    )
    assert len(with_rr) == min(3, len(without) or 3)
    if len(without) >= 2:
        # Stub reverses fused list; first without should become last among reranked window
        assert with_rr[0].chunk_id != without[0].chunk_id or len(without) == 1

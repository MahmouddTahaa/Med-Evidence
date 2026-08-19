import json
from pathlib import Path

import pytest

from clinical_rag.errors import IngestError
from clinical_rag.pipeline.ingest import run_ingest
from clinical_rag.pipeline.smoke_query import run_smoke_query
from clinical_rag.schemas import (
    ChunkConfig,
    EmbedConfig,
    IngestJobConfig,
    LegalFlags,
    ParserConfig,
    StrategyId,
)
from tests.helpers import HashEmbedder, legal_ok, raw_doc


def test_legal_gate_blocks(tmp_path: Path):
    path = tmp_path / "a.md"
    path.write_text("# A\n\nhello\n", encoding="utf-8")
    raw = raw_doc(path)
    raw.legal = LegalFlags()
    cfg = IngestJobConfig(corpus_id="t", job_id="j", files=[raw])
    with pytest.raises(IngestError, match="Legal checklist"):
        run_ingest(cfg, embedder=HashEmbedder(), jobs_dir=tmp_path / "jobs")


def test_prechunked_passthrough_smoke(tmp_path: Path):
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
        job_id="job-pc",
        files=[raw],
        parser=ParserConfig(),
        chunk=ChunkConfig(strategy_id=StrategyId.passthrough),
        embed=EmbedConfig(model_id="hash-test"),
        chroma=cfg_chroma(persist),
    )
    report, chunks = run_ingest(cfg, embedder=HashEmbedder(), jobs_dir=jobs)
    assert report.chunk_count == 5
    assert report.combo["vector_store"] == "chroma"
    assert "persist_dir" in report.combo["store"]
    assert all(c.strategy_id is StrategyId.passthrough for c in chunks)
    hits = run_smoke_query(
        persist_dir=str(persist),
        collection=report.collection_name,
        embedder=HashEmbedder(),
        query=chunks[0].text,
        top_k=5,
        index_model_id=report.embed_model_id,
    )
    assert len(hits) == 5
    assert hits[0].chunk_id == chunks[0].chunk_id
    assert hits[0].document_name
    assert hits[0].section_title
    assert hits[0].page_number is not None


def cfg_chroma(persist: Path):
    from clinical_rag.schemas import ChromaConfig

    return ChromaConfig(persist_dir=str(persist))

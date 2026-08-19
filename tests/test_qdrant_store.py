from pathlib import Path

import pytest

from clinical_rag.indexing.qdrant_store import QdrantStore
from clinical_rag.schemas import Chunk, ExtractionMethod, StrategyId


def test_qdrant_store_roundtrip(tmp_path: Path):
    pytest.importorskip("qdrant_client")
    chunk = Chunk(
        chunk_id="c1",
        text="ampicillin contraindications include hypersensitivity",
        document_name="Ampicillin",
        section_title="Contraindications",
        page_number=2,
        strategy_id=StrategyId.section_aware,
        corpus_id="demo",
        job_id="job1",
        extraction_method=ExtractionMethod.na,
        token_count=6,
        embed_model_id="hash-test",
        filename="ampicillin.nxml",
        doc_id="ampicillin",
    )
    store = QdrantStore(tmp_path)
    vector = [0.05] * 32
    store.replace("col", [chunk], [vector])
    assert store.index_embed_model_id("col") == "hash-test"
    hits = store.query("col", vector, 1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"
    assert hits[0].section_title == "Contraindications"
    assert "hypersensitivity" in hits[0].text

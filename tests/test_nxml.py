from __future__ import annotations

from pathlib import Path

import pytest

from clinical_rag.chunking import chunk_parsed
from clinical_rag.errors import IngestError
from clinical_rag.parsing.router import media_type_for, parse_document
from clinical_rag.pipeline.ingest import run_ingest
from clinical_rag.pipeline.smoke_query import run_smoke_query
from clinical_rag.schemas import (
    ChunkConfig,
    ChromaConfig,
    EmbedConfig,
    IngestJobConfig,
    MediaType,
    ParserConfig,
    StrategyId,
)
from tests.helpers import HashEmbedder, legal_ok, raw_doc

FIXTURE = Path("data/guidelines/sample_fd_guideline.nxml")


def _write_nxml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_media_type_nxml():
    assert media_type_for("guideline.nxml") is MediaType.nxml


def test_committed_fixture_parses_jats_sections():
    outcome = parse_document(raw_doc(FIXTURE, doc_id="fd"), ParserConfig(), None)
    assert outcome.parsed is not None
    parsed = outcome.parsed
    assert parsed.media_type is MediaType.nxml
    assert len(parsed.pages) >= 5

    titles = [p.blocks[0].text for p in parsed.pages]
    assert titles[0] == "Abstract > Summary"
    assert "Introduction" in titles
    assert "Treatment" in titles
    assert "Treatment > First-line" in titles
    assert "Treatment > Second-line" in titles

    intro = next(p for p in parsed.pages if p.blocks[0].text == "Introduction")
    assert "epigastric pain" in intro.text
    assert "postprandial fullness" in intro.text

    first_line = next(p for p in parsed.pages if p.blocks[0].text == "Treatment > First-line")
    assert "Omeprazole 20 mg" in first_line.text

    # ref-list-only sections have no paragraph text and are omitted
    assert "References" not in titles


def test_jats_default_namespace(tmp_path: Path):
    path = _write_nxml(
        tmp_path / "namespaced.nxml",
        """<?xml version="1.0"?>
        <article xmlns="http://jats.nlm.nih.gov/publishing/1.2">
          <body>
            <sec><title>Care</title><p>Namespace-safe paragraph.</p></sec>
          </body>
        </article>
        """,
    )
    outcome = parse_document(raw_doc(path), ParserConfig(), None)
    assert outcome.parsed is not None
    assert outcome.parsed.pages[0].blocks[0].text == "Care"
    assert "Namespace-safe" in outcome.parsed.pages[0].text


def test_jats_detected_on_xml_article_root(tmp_path: Path):
    path = _write_nxml(
        tmp_path / "article.xml",
        """<?xml version="1.0"?>
        <article>
          <body>
            <sec><title>From XML</title><p>JATS path on .xml extension.</p></sec>
          </body>
        </article>
        """,
    )
    raw = raw_doc(path, media=MediaType.xml)
    outcome = parse_document(raw, ParserConfig(), None)
    assert outcome.parsed is not None
    assert outcome.parsed.pages[0].blocks[0].text == "From XML"


def test_nxml_body_paragraph_without_sec(tmp_path: Path):
    path = _write_nxml(
        tmp_path / "body-p.nxml",
        """<?xml version="1.0"?>
        <article><body><p>Loose body paragraph.</p></body></article>
        """,
    )
    outcome = parse_document(raw_doc(path), ParserConfig(), None)
    assert outcome.parsed is not None
    assert outcome.parsed.pages[0].blocks[0].text == "Body"
    assert "Loose body paragraph" in outcome.parsed.pages[0].text


def test_nxml_skips_figures_and_ref_lists(tmp_path: Path):
    path = _write_nxml(
        tmp_path / "skip-meta.nxml",
        """<?xml version="1.0"?>
        <article>
          <body>
            <sec>
              <title>Clinical</title>
              <p>Keep this clinical sentence.</p>
              <fig><caption>Figure 1 should not appear.</caption></fig>
              <table-wrap><table><tr><td>Table cell noise</td></tr></table></table-wrap>
            </sec>
          </body>
        </article>
        """,
    )
    outcome = parse_document(raw_doc(path), ParserConfig(), None)
    page = outcome.parsed.pages[0]
    assert "Keep this clinical sentence" in page.text
    assert "Figure 1" not in page.text
    assert "Table cell" not in page.text


def test_nxml_empty_body_fails_closed(tmp_path: Path):
    path = _write_nxml(
        tmp_path / "empty.nxml",
        """<?xml version="1.0"?><article><body></body></article>""",
    )
    with pytest.raises(IngestError, match="no text content"):
        parse_document(raw_doc(path), ParserConfig(), None)


def test_nxml_invalid_xml_fails_closed(tmp_path: Path):
    path = _write_nxml(tmp_path / "bad.nxml", "<article><unclosed>")
    with pytest.raises(IngestError, match="invalid XML"):
        parse_document(raw_doc(path), ParserConfig(), None)


def test_nxml_section_aware_chunking_preserves_titles(tmp_path: Path):
    dest = tmp_path / "fd.nxml"
    dest.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    raw = raw_doc(dest, doc_id="fd")
    outcome = parse_document(raw, ParserConfig(), None)
    assert outcome.parsed is not None

    cfg = IngestJobConfig(
        corpus_id="demo",
        job_id="job-nxml-chunk",
        files=[raw],
        chunk=ChunkConfig(
            strategy_id=StrategyId.section_aware,
            target_tokens=40,
            max_tokens=60,
        ),
    )
    chunks = chunk_parsed(outcome.parsed, raw, cfg)
    assert len(chunks) >= 4
    titles = {c.section_title for c in chunks}
    assert "Introduction" in titles
    assert "Treatment > First-line" in titles
    assert all(c.chunk_id.startswith("fd-section_aware-") for c in chunks)
    assert all(c.page_number is not None for c in chunks)


def test_nxml_small_guideline_packs_into_one_chunk_by_default(tmp_path: Path):
    dest = tmp_path / "fd.nxml"
    dest.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    raw = raw_doc(dest, doc_id="fd")
    outcome = parse_document(raw, ParserConfig(), None)
    cfg = IngestJobConfig(
        corpus_id="demo",
        job_id="job-pack",
        files=[raw],
        chunk=ChunkConfig(strategy_id=StrategyId.section_aware),
    )
    chunks = chunk_parsed(outcome.parsed, raw, cfg)
    assert len(chunks) == 1
    blob = chunks[0].text
    assert "PPI therapy" in blob
    assert "Omeprazole 20 mg" in blob


def test_nxml_ingest_and_smoke_query(tmp_path: Path):
    dest = tmp_path / "fd.nxml"
    dest.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    persist = tmp_path / "chroma"
    jobs = tmp_path / "jobs"
    raw = raw_doc(dest, doc_id="fd")
    raw.legal = legal_ok()
    cfg = IngestJobConfig(
        corpus_id="demo",
        job_id="job-nxml",
        files=[raw],
        parser=ParserConfig(),
        chunk=ChunkConfig(
            strategy_id=StrategyId.section_aware,
            target_tokens=40,
            max_tokens=60,
        ),
        embed=EmbedConfig(model_id="hash-test"),
        chroma=ChromaConfig(persist_dir=str(persist)),
    )
    report, chunks = run_ingest(cfg, embedder=HashEmbedder(), jobs_dir=jobs)
    assert report.chunk_count >= 4
    assert report.page_count == 5
    assert all(c.filename.endswith(".nxml") for c in chunks)
    assert any("PPI therapy" in c.text for c in chunks)
    assert any(c.section_title == "Treatment > First-line" for c in chunks)

    hits = run_smoke_query(
        persist_dir=str(persist),
        collection=report.collection_name,
        embedder=HashEmbedder(),
        query="first line PPI functional dyspepsia",
        top_k=3,
        index_model_id=report.embed_model_id,
    )
    assert hits
    assert hits[0].document_name
    assert hits[0].section_title
    assert hits[0].chunk_id in {c.chunk_id for c in chunks}

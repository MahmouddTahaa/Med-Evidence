from pathlib import Path

from clinical_rag.chunking import chunk_parsed
from clinical_rag.parsing.pdf_pymupdf import parse_pdf
from clinical_rag.schemas import ChunkConfig, IngestJobConfig, ParserConfig, ParserProfile, StrategyId
from tests.helpers import raw_doc


def test_pdf_chunks_have_page_numbers(tmp_path: Path):
    import pymupdf

    pdf_path = tmp_path / "bp.pdf"
    doc = pymupdf.open()
    for i, body in enumerate(("Measurement protocol on page one.", "Treatment notes on page two."), start=1):
        page = doc.new_page()
        page.insert_text((72, 72), f"Heading {i}\n\n{body} " * 20)
    doc.save(pdf_path)
    doc.close()

    raw = raw_doc(pdf_path, doc_id="bp")
    parsed = parse_pdf(raw, ParserConfig(profile=ParserProfile.text_only), None)
    assert [p.page_number for p in parsed.pages] == [1, 2]
    assert all(p.text.strip() for p in parsed.pages)

    cfg = IngestJobConfig(
        corpus_id="t",
        job_id="job-pdf",
        files=[raw],
        chunk=ChunkConfig(strategy_id=StrategyId.fixed, target_tokens=80),
    )
    chunks = chunk_parsed(parsed, raw, cfg)
    assert chunks
    assert all(c.page_number in (1, 2) for c in chunks)

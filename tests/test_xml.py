from pathlib import Path

import pytest

from clinical_rag.errors import IngestError
from clinical_rag.parsing.router import media_type_for, parse_document
from clinical_rag.schemas import MediaType, ParserConfig
from tests.helpers import raw_doc


def test_media_type_xml():
    assert media_type_for("guide.xml") is MediaType.xml


def test_xml_happy_path(tmp_path: Path):
    path = tmp_path / "guide.xml"
    path.write_text(
        """<?xml version="1.0"?>
        <document title="Demo Guide">
          <section title="Treatment" number="2">
            <p>Start PPI therapy for functional dyspepsia.</p>
          </section>
        </document>
        """,
        encoding="utf-8",
    )
    outcome = parse_document(raw_doc(path, doc_id="xml"), ParserConfig(), None)
    assert outcome.parsed is not None
    assert outcome.parsed.pages[0].page_number == 2
    assert "PPI" in outcome.parsed.pages[0].text
    assert outcome.parsed.pages[0].blocks[0].kind == "heading"


def test_xml_invalid_fails_closed(tmp_path: Path):
    path = tmp_path / "bad.xml"
    path.write_text("<doc><unclosed>", encoding="utf-8")
    with pytest.raises(IngestError, match="invalid XML"):
        parse_document(raw_doc(path), ParserConfig(), None)

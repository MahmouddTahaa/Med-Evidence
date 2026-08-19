import json
from pathlib import Path

import pytest

from clinical_rag.errors import IngestError
from clinical_rag.parsing.json_doc import parse_raw_json
from clinical_rag.parsing.json_prechunked import is_prechunked_payload, load_prechunked
from clinical_rag.parsing.router import parse_document
from clinical_rag.schemas import ParserConfig
from tests.helpers import raw_doc


def test_detects_prechunked_vs_raw():
    assert is_prechunked_payload({"document_name": "A", "chunks": [{"text": "x", "chunk_id": "1"}]})
    assert not is_prechunked_payload({"document_name": "A", "sections": []})


def test_prechunked_rejects_missing_text(tmp_path: Path):
    payload = {"document_name": "Guide", "chunks": [{"chunk_id": "a"}]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(IngestError, match="missing text"):
        load_prechunked(payload, raw_doc(path))


def test_prechunked_warns_on_missing_section(tmp_path: Path):
    payload = {
        "document_name": "Guide",
        "chunks": [{"chunk_id": "a", "text": "hello", "page_number": 1}],
    }
    chunks, warnings = load_prechunked(payload, raw_doc(tmp_path / "g.json"))
    assert chunks[0].section_title == "(unknown)"
    assert any("section_title" in w for w in warnings)


def test_raw_json_sections_become_pages(tmp_path: Path):
    payload = {
        "document_name": "Guide",
        "sections": [{"title": "Rx", "text": "Start ACEI", "page_number": 3}],
    }
    parsed = parse_raw_json(payload, raw_doc(tmp_path / "g.json"))
    assert parsed.pages[0].page_number == 3
    assert parsed.pages[0].blocks[0].kind == "heading"
    assert parsed.pages[0].blocks[0].text == "Rx"


def test_router_prechunked_file(tmp_path: Path):
    path = tmp_path / "pc.json"
    path.write_text(
        json.dumps(
            {
                "document_name": "Guide",
                "chunks": [
                    {
                        "chunk_id": "g-1",
                        "text": "body",
                        "section_title": "S",
                        "page_number": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    outcome = parse_document(raw_doc(path), ParserConfig(), None)
    assert outcome.prechunked is not None
    assert outcome.prechunked[0].chunk_id == "g-1"
    assert outcome.parsed is None

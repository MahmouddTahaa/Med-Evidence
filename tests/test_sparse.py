from clinical_rag.retrieval.sparse import SparseIndex, tokenize
from clinical_rag.schemas import KeywordMethod


def _rows():
    return [
        {
            "chunk_id": "c-noise",
            "text": "Blood pressure measurement technique for adults in clinic.",
            "document_name": "Demo",
            "section_title": "Measurement",
            "page_number": 1,
            "extraction_method": "text",
            "source_url": "",
            "token_count": 10,
        },
        {
            "chunk_id": "c-target",
            "text": "First-line hypertension treatment uses ACE inhibitors or ARBs.",
            "document_name": "Demo",
            "section_title": "Treatment",
            "page_number": 2,
            "extraction_method": "text",
            "source_url": "",
            "token_count": 12,
        },
        {
            "chunk_id": "c-other",
            "text": "Lifestyle advice includes sodium reduction and exercise.",
            "document_name": "Demo",
            "section_title": "Lifestyle",
            "page_number": 3,
            "extraction_method": "text",
            "source_url": "",
            "token_count": 8,
        },
    ]


def test_tokenize_lowercase_words():
    assert tokenize("ACE-Inhibitors & ARBs!") == ["ace", "inhibitors", "arbs"]


def test_bm25_ranks_distinctive_term():
    idx = SparseIndex.from_chunks(_rows(), method=KeywordMethod.bm25)
    hits = idx.query("ACE inhibitors hypertension", top_k=3)
    assert hits
    assert hits[0].chunk_id == "c-target"
    assert all(h.chunk_id for h in hits)


def test_tfidf_ranks_distinctive_term():
    idx = SparseIndex.from_chunks(_rows(), method=KeywordMethod.tfidf)
    hits = idx.query("ACE inhibitors", top_k=3)
    assert hits
    assert hits[0].chunk_id == "c-target"


def test_sparse_empty_query():
    idx = SparseIndex.from_chunks(_rows(), method=KeywordMethod.bm25)
    assert idx.query("   ", top_k=5) == []

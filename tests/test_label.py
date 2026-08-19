from clinical_rag.eval.label import label_from_chunks


def test_section_gold_includes_sibling_chunks():
    templates = [
        {
            "id": "q1",
            "query": "How is Adalimumab administered and dosed?",
            "anchor": "syringes, pens, and single-use glass vials",
            "filename": "article-93108.nxml",
            "section_title": "Administration",
        }
    ]
    chunks = [
        {
            "chunk_id": "admin-1",
            "filename": "article-93108.nxml",
            "section_title": "Administration",
            "text": "syringes, pens, and single-use glass vials, with strengths of 10 mg",
        },
        {
            "chunk_id": "admin-2",
            "filename": "article-93108.nxml",
            "section_title": "Administration",
            "text": "Dose is 40 mg subcutaneously every other week.",
        },
        {
            "chunk_id": "ae-1",
            "filename": "article-93108.nxml",
            "section_title": "Adverse Effects",
            "text": "Injection site reactions are common.",
        },
    ]
    labeled = label_from_chunks(templates, chunks)
    assert labeled[0].relevant_chunk_ids == ["admin-1", "admin-2"]


def test_anchor_only_when_no_section():
    templates = [
        {
            "id": "q1",
            "query": "x",
            "anchor": "unique-anchor-phrase",
        }
    ]
    chunks = [
        {"chunk_id": "a", "filename": "a.nxml", "section_title": "A", "text": "unique-anchor-phrase here"},
        {"chunk_id": "b", "filename": "a.nxml", "section_title": "A", "text": "other text"},
    ]
    labeled = label_from_chunks(templates, chunks)
    assert labeled[0].relevant_chunk_ids == ["a"]

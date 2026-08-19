from clinical_rag.parsing.quality import needs_ocr


def test_empty_and_low_text_need_ocr():
    assert needs_ocr("") is True
    assert needs_ocr("...") is True
    assert needs_ocr("/// ###") is True


def test_normal_prose_skips_ocr():
    text = "Confirm hypertension with repeated seated measurements using a sized cuff. " * 3
    assert needs_ocr(text) is False

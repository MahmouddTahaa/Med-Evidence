from clinical_rag.parsing.cleanup import (
    clean_section_title,
    normalize_line,
    strip_repeating_headers_footers,
)
from clinical_rag.schemas import (
    ExtractionMethod,
    MediaType,
    ParsedDocument,
    ParsedPage,
    TextBlock,
)


def test_clean_section_title_noise():
    assert clean_section_title("Guidelines") == "(unknown)"
    assert clean_section_title("Copyright © 2020 ACME") == "(unknown)"
    assert clean_section_title("First-line treatment") == "First-line treatment"


def test_clean_section_title_publisher_and_authors():
    assert clean_section_title("Gut: first published as") == "(unknown)"
    assert clean_section_title("by guest") == "(unknown)"
    assert clean_section_title("Alexander C Ford \u200d \u200d 1,2") == "(unknown)"
    assert (
        clean_section_title(
            "Christopher J Black ,1,2 Peter A Paine,3,4 Anurag Agrawal,5"
        )
        == "(unknown)"
    )
    assert (
        clean_section_title("British Society of Gastroenterology guidelines on the")
        == "British Society of Gastroenterology guidelines on the"
    )


def test_normalize_line_strips_zwj_and_nbsp():
    assert normalize_line("Alexander C Ford\u2002 \u200d \u200d 1,2") == "Alexander C Ford 1,2"
    assert normalize_line("Black\xa0CJ, et\xa0al.") == "Black CJ, et al."


def test_strip_repeating_headers():
    pages = []
    for i in range(3):
        pages.append(
            ParsedPage(
                page_number=i + 1,
                text=f"Confidential watermark\nBody page {i + 1}\nConfidential watermark",
                extraction_method=ExtractionMethod.text,
                blocks=[
                    TextBlock(kind="heading", text="Guidelines", heading_level=1),
                    TextBlock(kind="paragraph", text=f"Body page {i + 1}"),
                ],
            )
        )
    doc = ParsedDocument(
        doc_id="d",
        document_name="d",
        media_type=MediaType.pdf,
        pages=pages,
    )
    cleaned = strip_repeating_headers_footers(doc)
    assert all("Confidential" not in p.text for p in cleaned.pages)
    assert all(
        not any(b.kind == "heading" and b.text == "Guidelines" for b in p.blocks)
        for p in cleaned.pages
    )


def test_strip_bmj_guest_watermark_and_author_headings():
    """BSG/Gut-style PDFs: guest watermark + author lines as false headings."""
    watermark = (
        "Protected by copyright, including for uses related to text and data mining.\n"
        "by guest\n"
        "on July 9, 2026\n"
        "http://gut.bmj.com/\n"
        "Downloaded from\n"
        "7 July 2022.\n"
        "10.1136/gutjnl-2022-327737 on\n"
        "Gut: first published as"
    )
    running = "Black CJ, et al. Gut 2022;71:1697–1723. doi:10.1136/gutjnl-2022-327737"
    pages = []
    for i in range(5):
        pages.append(
            ParsedPage(
                page_number=i + 1,
                text=(
                    f"{i + 1697}\n{running}\nGuidelines\n"
                    f"We recommend acid suppression on page {i + 1}.\n"
                    f"{watermark}"
                ),
                extraction_method=ExtractionMethod.text,
                blocks=[
                    TextBlock(kind="heading", text="Guidelines", heading_level=1),
                    TextBlock(
                        kind="heading",
                        text="Alexander C Ford \u200d \u200d 1,2",
                        heading_level=1,
                    ),
                    TextBlock(
                        kind="heading",
                        text="First-line treatment of FD",
                        heading_level=1,
                    ),
                    TextBlock(
                        kind="paragraph",
                        text=f"We recommend acid suppression on page {i + 1}.",
                    ),
                    TextBlock(kind="paragraph", text=watermark),
                ],
            )
        )
    doc = ParsedDocument(
        doc_id="fd",
        document_name="fd",
        media_type=MediaType.pdf,
        pages=pages,
    )
    cleaned = strip_repeating_headers_footers(doc)

    for page in cleaned.pages:
        low = page.text.lower()
        assert "by guest" not in low
        assert "downloaded from" not in low
        assert "first published as" not in low
        assert "gut.bmj.com" not in low
        assert "doi:10.1136" not in low
        assert "We recommend acid suppression" in page.text

        headings = [b.text for b in page.blocks if b.kind == "heading"]
        assert headings == ["First-line treatment of FD"]
        assert not any("Alexander" in b.text for b in page.blocks if b.kind == "heading")
        # Author line demoted to paragraph, not dropped entirely.
        assert any(
            b.kind == "paragraph" and "Alexander C Ford" in b.text for b in page.blocks
        )

from __future__ import annotations

import re
from collections import Counter

from clinical_rag.schemas import ParsedDocument, TextBlock

# Soft hyphen / zero-width / BOM noise common in publisher PDFs.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\xad]")

_PUBLISHER_NOISE = re.compile(
    r"(?i)("
    r"\bby\s+guest\b|"
    r"\bdownloaded\s+from\b|"
    r"first\s+published\s+as|"
    r"protected\s+by\s+copyright|"
    r"bmj\.com|"
    r"bmj\s+publishing|"
    r"disclaims\s+all\s+liability|"
    r"placed\s+on\s+this\s+supplemental|"
    r"supplemental\s+material\s+which\s+has\s+been\s+supplied|"
    r"all\s+rights\s+reserved|"
    r"\bconfidential\b|"
    r"do\s+not\s+copy|"
    r"\bwatermark\b|"
    r"©|\(c\)|"
    r"\bcopyright\b"
    r")"
)
_RUNNING_CITE = re.compile(
    r".{0,80}\bet\s*al\b.{0,60}\bdoi:\s*10\.\d|^doi:\s*10\.\d{4,}/",
    re.IGNORECASE,
)
_DOI_FRAGMENT = re.compile(
    r"^(doi:\s*)?10\.\d{4,}/[\w.\-]+(?:\s+on)?\.?$",
    re.IGNORECASE,
)
_DATE_ONLY = re.compile(
    r"^(?:on\s+)?[A-Za-z]+\s+\d{1,2},\s+\d{4}\.?$|^\d{1,2}\s+[A-Za-z]+\s+\d{4}\.?$",
    re.IGNORECASE,
)
_PAGE_NUM = re.compile(r"^\d{1,4}$")
_NOISE_TITLE = re.compile(
    r"(?i)^(guidelines?|table of contents|contents|page \d+|draft|references?)$"
)
# Author / affiliation lines mis-detected as headings (font-size heuristics).
_AUTHOR_AFFIL = re.compile(
    r"(?:"
    r"[A-Za-z]\s+\d{1,2}(?:,\d{1,2})+\s*$|"  # Ford 1,2
    r",\s*\d{1,2}(?:,\d{1,2})*(?:\s|,|$)"  # Aziz ,6,7
    r")"
)


def scrub_line(text: str) -> str:
    """Remove invisible PDF glyphs but preserve ordinary spacing."""
    return _INVISIBLE.sub("", text.replace("\xa0", " "))


def normalize_line(text: str) -> str:
    """Collapse publisher PDF artifacts so the same watermark matches across pages."""
    return " ".join(scrub_line(text).split()).strip()


def _looks_like_author_line(text: str) -> bool:
    cleaned = normalize_line(text)
    if not cleaned or len(cleaned) > 220:
        return False
    if not _AUTHOR_AFFIL.search(cleaned):
        return False
    # Real recommendations rarely pack many comma-separated names + affil numbers.
    return cleaned.count(",") >= 1 and bool(re.search(r"[A-Za-z]{2,}", cleaned))


def _is_noise_line(line: str) -> bool:
    text = normalize_line(line)
    if not text:
        return True
    if len(text) <= 3:
        return True
    if _PAGE_NUM.match(text):
        return True
    if _PUBLISHER_NOISE.search(text):
        return True
    if _RUNNING_CITE.search(text):
        return True
    if _DOI_FRAGMENT.match(text):
        return True
    if _DATE_ONLY.match(text):
        return True
    if _NOISE_TITLE.match(text):
        return True
    return False


def clean_section_title(title: str) -> str:
    text = normalize_line(title)
    if not text or _is_noise_line(text) or _looks_like_author_line(text):
        return "(unknown)"
    # Drop leading boilerplate like "Guidelines — " when the rest is usable.
    text = re.sub(r"(?i)^guidelines?\s*[-–:]\s*", "", text).strip() or text
    if _is_noise_line(text) or _looks_like_author_line(text):
        return "(unknown)"
    return text


def _repeating_lines(parsed: ParsedDocument) -> set[str]:
    """Normalized lines that repeat on enough pages (headers/footers)."""
    if len(parsed.pages) < 2:
        return set()

    line_counts: Counter[str] = Counter()
    for page in parsed.pages:
        lines = {normalize_line(ln) for ln in page.text.splitlines() if normalize_line(ln)}
        line_counts.update(lines)

    # 40% catches guest/download watermarks that miss a strict majority (e.g. 27/48).
    threshold = max(2, int(0.4 * len(parsed.pages)))
    repeating = {
        line
        for line, count in line_counts.items()
        if count >= threshold and len(line) < 160
    }
    return repeating


def _keep_line(line: str, repeating: set[str]) -> bool:
    text = normalize_line(line)
    if not text:
        return False
    if text in repeating:
        return False
    if _is_noise_line(text):
        return False
    return True


def strip_repeating_headers_footers(parsed: ParsedDocument) -> ParsedDocument:
    """Drop headers/footers/watermarks and demote author-list pseudo-headings."""
    repeating = _repeating_lines(parsed)

    for page in parsed.pages:
        page_changed = False
        kept_lines = [
            scrub_line(ln) for ln in page.text.splitlines() if _keep_line(ln, repeating)
        ]
        new_text = "\n".join(kept_lines).strip()
        if new_text != page.text.strip():
            page_changed = True
        page.text = new_text

        cleaned_blocks: list[TextBlock] = []
        for block in page.blocks:
            if block.kind == "heading":
                if _looks_like_author_line(block.text) or _is_noise_line(block.text):
                    # Keep authorship as body text; never let it become section_title.
                    body = normalize_line(block.text)
                    if body and not _is_noise_line(body):
                        cleaned_blocks.append(
                            TextBlock(kind="paragraph", text=body, heading_level=None)
                        )
                    page_changed = True
                    continue
                title = clean_section_title(block.text)
                if title == "(unknown)":
                    page_changed = True
                    continue
                if title != block.text:
                    page_changed = True
                cleaned_blocks.append(
                    TextBlock(kind="heading", text=title, heading_level=block.heading_level)
                )
            else:
                text = "\n".join(
                    scrub_line(ln)
                    for ln in block.text.splitlines()
                    if _keep_line(ln, repeating)
                ).strip()
                if text:
                    cleaned_blocks.append(
                        TextBlock(kind=block.kind, text=text, heading_level=block.heading_level)
                    )
                    if text != block.text.strip():
                        page_changed = True
                elif block.text.strip():
                    page_changed = True
        page.blocks = cleaned_blocks
        if page_changed:
            page.warnings.append("stripped repeating headers/footers/watermarks")
    return parsed

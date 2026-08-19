from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from clinical_rag.errors import IngestError
from clinical_rag.schemas import (
    ExtractionMethod,
    ParsedDocument,
    ParsedPage,
    RawDocument,
    TextBlock,
)

_SKIP_TAGS = frozenset(
    {
        "sec",
        "title",
        "ref-list",
        "fig",
        "table-wrap",
        "graphic",
        "media",
        "supplementary-material",
    }
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _text_join(elem: ET.Element) -> str:
    parts = [t.strip() for t in elem.itertext() if t and t.strip()]
    return " ".join(parts)


def _is_nxml_jats(root: ET.Element, filename: str) -> bool:
    if Path(filename).suffix.lower() == ".nxml":
        return True
    if _local(root.tag) == "article":
        return True
    return any(_local(elem.tag) == "body" for elem in root.iter())


def _direct_title(elem: ET.Element) -> str:
    for child in elem:
        if _local(child.tag) == "title":
            return _text_join(child)
    return ""


def _direct_paragraphs(elem: ET.Element) -> str:
    parts: list[str] = []
    for child in elem:
        tag = _local(child.tag)
        if tag in _SKIP_TAGS:
            continue
        if tag == "p":
            text = _text_join(child)
            if text:
                parts.append(text)
        else:
            text = _text_join(child)
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def _walk_jats_secs(sec: ET.Element, prefix: str = "") -> list[tuple[str, str]]:
    title = _direct_title(sec) or _local(sec.tag)
    full_title = f"{prefix} > {title}" if prefix else title
    out: list[tuple[str, str]] = []
    text = _direct_paragraphs(sec)
    if text.strip():
        out.append((full_title, text))
    for child in sec:
        if _local(child.tag) == "sec":
            out.extend(_walk_jats_secs(child, full_title))
    return out


def _parse_jats_abstract(root: ET.Element) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for elem in root.iter():
        if _local(elem.tag) != "abstract":
            continue
        for child in elem:
            tag = _local(child.tag)
            if tag == "sec":
                out.extend(_walk_jats_secs(child, "Abstract"))
            elif tag == "p":
                text = _text_join(child)
                if text:
                    out.append(("Abstract", text))
        break
    return out


def _parse_jats_body(root: ET.Element) -> list[tuple[str, str]]:
    body = next((elem for elem in root.iter() if _local(elem.tag) == "body"), None)
    container = body if body is not None else root
    out: list[tuple[str, str]] = []
    for child in container:
        tag = _local(child.tag)
        if tag == "sec":
            out.extend(_walk_jats_secs(child))
        elif tag == "p":
            text = _text_join(child)
            if text:
                out.append(("Body", text))
    return out


def _parse_jats_pages(root: ET.Element, filename: str) -> list[ParsedPage]:
    sections = _parse_jats_abstract(root) + _parse_jats_body(root)
    if not sections:
        body = next((elem for elem in root.iter() if _local(elem.tag) == "body"), None)
        fallback = _text_join(body) if body is not None else _text_join(root)
        if not fallback.strip():
            raise IngestError(f"{filename}: NXML/JATS has no text content")
        sections = [("Document", fallback)]

    pages: list[ParsedPage] = []
    for i, (title, text) in enumerate(sections, start=1):
        section_title = title.strip() or f"Section {i}"
        blocks = [TextBlock(kind="heading", text=section_title, heading_level=1)]
        if text.strip():
            blocks.append(TextBlock(kind="paragraph", text=text))
        pages.append(
            ParsedPage(
                page_number=i,
                text=text or section_title,
                extraction_method=ExtractionMethod.na,
                blocks=blocks,
            )
        )
    return pages


def _parse_generic_pages(root: ET.Element, filename: str) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    page_nodes = [e for e in root.iter() if _local(e.tag) in {"page", "section", "chapter"}]
    if not page_nodes:
        body = _text_join(root)
        if not body.strip():
            raise IngestError(f"{filename}: XML has no text content")
        title = root.attrib.get("title") or _local(root.tag)
        pages.append(
            ParsedPage(
                page_number=1,
                text=body,
                extraction_method=ExtractionMethod.na,
                blocks=[
                    TextBlock(kind="heading", text=title, heading_level=1),
                    TextBlock(kind="paragraph", text=body),
                ],
            )
        )
        return pages

    for i, node in enumerate(page_nodes, start=1):
        title = (
            node.attrib.get("title")
            or node.attrib.get("name")
            or next(
                (
                    (c.text or "").strip()
                    for c in list(node)
                    if _local(c.tag) in {"title", "heading", "head"} and (c.text or "").strip()
                ),
                _local(node.tag),
            )
        )
        text = _text_join(node)
        page_number = int(node.attrib.get("number") or node.attrib.get("page") or i)
        blocks = [TextBlock(kind="heading", text=title, heading_level=1)]
        if text.strip():
            blocks.append(TextBlock(kind="paragraph", text=text))
        pages.append(
            ParsedPage(
                page_number=page_number,
                text=text or title,
                extraction_method=ExtractionMethod.na,
                blocks=blocks,
            )
        )
    return pages


def parse_xml(raw: RawDocument) -> ParsedDocument:
    try:
        root = ET.parse(raw.path).getroot()
    except ET.ParseError as exc:
        raise IngestError(f"{raw.filename}: invalid XML ({exc})") from exc
    except OSError as exc:
        raise IngestError(f"{raw.filename}: cannot read XML ({exc})") from exc

    if _is_nxml_jats(root, raw.filename):
        pages = _parse_jats_pages(root, raw.filename)
    else:
        pages = _parse_generic_pages(root, raw.filename)

    return ParsedDocument(
        doc_id=raw.doc_id,
        document_name=raw.document_name,
        source_url=raw.source_url,
        media_type=raw.media_type,
        filename=raw.filename,
        pages=pages,
    )

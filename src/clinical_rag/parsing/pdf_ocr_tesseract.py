from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from clinical_rag.errors import IngestError


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_pixmap_png(png_path: str, lang: str = "eng") -> str:
    import pytesseract

    return pytesseract.image_to_string(png_path, lang=lang)


def ocr_page(page, *, lang: str, dpi: int, cache_path: Path | None) -> str:
    if cache_path and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    if not tesseract_available():
        raise IngestError(
            "tesseract is not installed. Install tesseract-ocr (and eng language data) "
            "or use parser profile text_only."
        )
    import pymupdf

    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        pix.save(tmp.name)
        text = ocr_pixmap_png(tmp.name, lang=lang)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    return text

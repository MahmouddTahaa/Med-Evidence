from __future__ import annotations

import re

_ALNUM = re.compile(r"[0-9A-Za-z\u0600-\u06FF]")


def alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for _ in _ALNUM.finditer(text)) / max(len(text), 1)


def needs_ocr(text: str, min_chars: int = 50, min_alnum_ratio: float = 0.3) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    return alnum_ratio(stripped) < min_alnum_ratio

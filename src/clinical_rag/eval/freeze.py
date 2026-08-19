from __future__ import annotations

from pathlib import Path

import yaml

from clinical_rag.errors import IngestError


def freeze_combo(combo: dict, path: Path | str = "configs/winning.yaml") -> Path:
    """Write the winning lab combination for a future product UI to load.

    Pass through chunk.prefix_section_title and retrieval sibling/parent flags
    when present. Schema defaults stay off so a sequential-freeze re-run does
    not silently enable the 2026-08-19 product prefix.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    required = ("parser_engine", "chunk", "embed", "vector_store")
    missing = [k for k in required if k not in combo]
    if missing:
        raise IngestError(f"Cannot freeze combo; missing keys: {', '.join(missing)}")
    payload = {
        "parser_engine": combo.get("parser_engine"),
        "parser_profile": combo.get("parser_profile", "ocr_fallback"),
        "chunk": combo.get("chunk") or {},
        "embed": combo.get("embed") or {},
        "vector_store": combo.get("vector_store", "chroma"),
        "retrieval": combo.get("retrieval")
        or {
            "mode": "dense",
            "top_k": 5,
            "keyword_method": "bm25",
            "semantic_weight": 0.70,
            "keyword_weight": 0.30,
            "rrf_k": 60,
            "fetch_k": 20,
            "rerank": False,
            "sibling_fill": False,
            "parent_child": False,
        },
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def load_selected(path: Path | str = "configs/winning.yaml") -> dict:
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"No frozen config at {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

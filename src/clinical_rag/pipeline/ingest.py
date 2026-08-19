from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from clinical_rag.adapters.embedders import build_embedder
from clinical_rag.adapters.parsers import parse_with_engine
from clinical_rag.adapters.stores import build_store
from clinical_rag.chunking import chunk_parsed
from clinical_rag.chunking.base import stamp_prechunked
from clinical_rag.config import collection_name
from clinical_rag.errors import IngestError
from clinical_rag.indexing.embed import Embedder
from clinical_rag.schemas import (
    Chunk,
    ExtractionMethod,
    IngestJobConfig,
    IngestReport,
    StrategyId,
    VectorStoreKind,
)

ProgressFn = Callable[[str, str], None]


def _progress(fn: ProgressFn | None, stage: str, message: str) -> None:
    if fn:
        fn(stage, message)


def _store_connection(config: IngestJobConfig) -> dict:
    kind = config.vector_store
    if kind is VectorStoreKind.chroma:
        return {"persist_dir": config.chroma.persist_dir}
    if kind is VectorStoreKind.qdrant:
        return {"persist_dir": config.qdrant.persist_dir}
    raise IngestError(f"unsupported store: {kind}")



def combo_from_config(config: IngestJobConfig, *, embed_model_id: str, embed_device: str) -> dict:
    return {
        "parser_engine": config.parser.engine.value,
        "parser_profile": config.parser.profile.value,
        "chunk": {
            "strategy_id": config.chunk.strategy_id.value,
            "target_tokens": config.chunk.target_tokens,
            "overlap_ratio": config.chunk.overlap_ratio,
            "parent_tokens": config.chunk.parent_tokens,
            "child_tokens": config.chunk.child_tokens,
            "max_tokens": config.chunk.max_tokens,
            "prefix_section_title": config.chunk.prefix_section_title,
        },
        "embed": {
            "provider": config.embed.provider.value,
            "model_id": embed_model_id,
            "device": embed_device,
        },
        "vector_store": config.vector_store.value,
        "store": _store_connection(config),
        "retrieval": config.retrieval.to_combo_dict(),
    }


def run_ingest(
    config: IngestJobConfig,
    *,
    progress: ProgressFn | None = None,
    embedder: Embedder | None = None,
    jobs_dir: Path | None = None,
) -> tuple[IngestReport, list[Chunk]]:
    incomplete = [d.filename for d in config.files if not d.legal.complete()]
    if incomplete:
        raise IngestError(f"Legal checklist incomplete for: {', '.join(incomplete)}")
    if not config.files:
        raise IngestError("No files to ingest")

    jobs_dir = Path(jobs_dir or "artifacts/jobs")
    job_dir = jobs_dir / config.job_id
    cache_dir = job_dir / "parsed"
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Chunk] = []
    warnings: list[str] = []
    page_count = 0
    ocr_page_count = 0

    for raw in config.files:
        _progress(progress, "parse", f"Parsing {raw.filename} ({config.parser.engine.value})")
        outcome = parse_with_engine(raw, config.parser, cache_dir)
        warnings.extend(outcome.warnings)
        if outcome.prechunked is not None:
            if config.chunk.strategy_id is not StrategyId.passthrough:
                warnings.append(
                    f"{raw.filename}: pre-chunked JSON used passthrough (not re-chunked)"
                )
            chunks.extend(stamp_prechunked(c, config) for c in outcome.prechunked)
            continue
        if outcome.parsed is None:
            raise IngestError(f"{raw.filename}: parser returned nothing")
        parsed = outcome.parsed
        warnings.extend(parsed.warnings)
        page_count += len(parsed.pages)
        ocr_page_count += sum(
            1
            for p in parsed.pages
            if p.extraction_method in (ExtractionMethod.ocr, ExtractionMethod.hybrid)
        )
        if config.chunk.strategy_id is StrategyId.passthrough:
            raise IngestError("passthrough is only valid for pre-chunked JSON")
        chunks.extend(chunk_parsed(parsed, raw, config))

    if not chunks:
        raise IngestError("No chunks produced")

    _progress(progress, "embed", f"Embedding {len(chunks)} chunks ({config.embed.model_id})")
    embedder = embedder or build_embedder(config.embed)
    warnings.extend(getattr(embedder, "warnings", []))
    vectors = embedder.encode([c.text for c in chunks])
    for chunk in chunks:
        chunk.embed_model_id = embedder.model_id

    name = collection_name(config.corpus_id, config.chunk.strategy_id.value, embedder.model_id)
    _progress(progress, "store", f"Writing collection {name} ({config.vector_store.value})")
    store = build_store(
        config.vector_store,
        chroma=config.chroma,
        qdrant=config.qdrant,
    )
    store.replace(name, chunks, vectors)

    embed_device = getattr(embedder, "device", "n/a")
    combo = combo_from_config(config, embed_model_id=embedder.model_id, embed_device=embed_device)
    report = IngestReport(
        job_id=config.job_id,
        corpus_id=config.corpus_id,
        collection_name=name,
        strategy_id=config.chunk.strategy_id.value,
        embed_model_id=embedder.model_id,
        embed_device=embed_device,
        embed_provider=config.embed.provider.value,
        parser_engine=config.parser.engine.value,
        parser_profile=config.parser.profile.value,
        vector_store=config.vector_store.value,
        retrieval_mode=config.retrieval.mode.value,
        page_count=page_count,
        ocr_page_count=ocr_page_count,
        chunk_count=len(chunks),
        warnings=warnings,
        combo=combo,
    )
    (job_dir / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    payload = [c.model_dump(mode="json") for c in chunks]
    (job_dir / "chunks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report, chunks

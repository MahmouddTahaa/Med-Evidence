"""CLI twin of the Streamlit ingest tab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from clinical_rag.adapters.embedders import embedder_for_index
from clinical_rag.config import get_settings, make_doc_id, make_job_id, store_kwargs
from clinical_rag.errors import IngestError
from clinical_rag.eval.protocol import (
    HOLD_EMBED_MODEL_ID,
    OPENAI_EMBED_MODEL_ID,
    STAGE3_CHUNKS,
)
from clinical_rag.parsing.router import media_type_for
from clinical_rag.pipeline.ingest import run_ingest
from clinical_rag.pipeline.smoke_query import run_smoke_query
from clinical_rag.schemas import (
    ChunkConfig,
    EmbedConfig,
    EmbedProvider,
    IngestJobConfig,
    LegalFlags,
    ParserConfig,
    ParserProfile,
    RawDocument,
    StrategyId,
    VectorStoreKind,
)


def _embed_provider(model_id: str) -> EmbedProvider:
    if model_id == OPENAI_EMBED_MODEL_ID:
        return EmbedProvider.openai
    return EmbedProvider.sentence_transformers


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a vector index from local files")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--corpus-id", default="demo")
    p.add_argument(
        "--strategy",
        default=StrategyId.section_aware.value,
        choices=[s.value for s in STAGE3_CHUNKS],
    )
    p.add_argument("--parser-profile", default="ocr_fallback", choices=[p.value for p in ParserProfile])
    p.add_argument("--embed-model", default=HOLD_EMBED_MODEL_ID)
    p.add_argument(
        "--vector-store",
        choices=[v.value for v in VectorStoreKind],
        default=None,
        help="Override VECTOR_STORE from .env (chroma | qdrant)",
    )
    p.add_argument("--source-url", default="")
    p.add_argument("--confirm-legal", action="store_true", help="Attest all four legal flags")
    p.add_argument("--smoke-query", default="")
    p.add_argument("--top-k", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _args()
    if not args.confirm_legal:
        raise SystemExit("Refusing to ingest without --confirm-legal")
    settings = get_settings()
    legal = LegalFlags(
        open_access_or_reusable=True,
        redistribution_ok_for_indexing=True,
        edition_current=True,
        attribution_documented=True,
    )
    files: list[RawDocument] = []
    used: dict[str, int] = {}
    for path in args.files:
        if not path.is_file():
            raise SystemExit(f"Not a file: {path}")
        doc_id = make_doc_id(path.name)
        used[doc_id] = used.get(doc_id, 0) + 1
        if used[doc_id] > 1:
            doc_id = f"{doc_id}-{used[doc_id]}"
        files.append(
            RawDocument(
                doc_id=doc_id,
                filename=path.name,
                media_type=media_type_for(path.name),
                document_name=path.stem.replace("_", " "),
                source_url=args.source_url,
                path=str(path.resolve()),
                legal=legal,
            )
        )
    vector_store = (
        VectorStoreKind(args.vector_store) if args.vector_store else settings.vector_store
    )
    provider = _embed_provider(args.embed_model)
    config = IngestJobConfig(
        corpus_id=args.corpus_id,
        job_id=make_job_id(),
        files=files,
        parser=ParserConfig(profile=ParserProfile(args.parser_profile)),
        chunk=ChunkConfig(strategy_id=StrategyId(args.strategy)),
        embed=EmbedConfig(
            model_id=args.embed_model,
            provider=provider,
            device=settings.embed.device,
        ),
        chroma=settings.chroma,
        qdrant=settings.qdrant,
        vector_store=vector_store,
        smoke_query=settings.smoke_query,
    )
    try:
        report, _chunks = run_ingest(
            config,
            progress=lambda stage, msg: print(f"[{stage}] {msg}"),
            jobs_dir=settings.jobs_dir,
        )
    except IngestError as exc:
        raise SystemExit(str(exc)) from exc
    print(report.model_dump_json(indent=2))
    query = args.smoke_query.strip()
    if query:
        persist_dir = (
            config.qdrant.persist_dir
            if vector_store is VectorStoreKind.qdrant
            else config.chroma.persist_dir
        )
        embedder = embedder_for_index(
            model_id=report.embed_model_id,
            provider=report.embed_provider,
            device=report.embed_device,
            batch_size=settings.embed.batch_size,
            purpose="query",
        )
        hits = run_smoke_query(
            persist_dir=persist_dir,
            collection=report.collection_name,
            embedder=embedder,
            query=query,
            top_k=args.top_k or settings.smoke_query.top_k,
            index_model_id=report.embed_model_id,
            vector_store=report.vector_store,
            **store_kwargs(settings),
        )
        for hit in hits:
            print(f"{hit.score:.3f}\t{hit.chunk_id}\t{hit.document_name}\t{hit.section_title}\tp{hit.page_number}")


if __name__ == "__main__":
    main()

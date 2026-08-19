"""Typed product stack: `configs/winning.yaml` plus the local serving pointer.

Eval freeze still writes untyped combo dicts. Query and ingest for the clinician
path load this module — not raw YAML — so knobs stay validated at the boundary.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from clinical_rag.errors import IngestError
from clinical_rag.schemas import (
    ChromaConfig,
    ChunkConfig,
    EmbedConfig,
    IngestJobConfig,
    KeywordMethod,
    ParserConfig,
    ParserEngine,
    ParserProfile,
    QdrantConfig,
    RawDocument,
    RetrievalConfig,
    RetrievalMode,
    VectorStoreKind,
)

WINNING_PATH = Path("configs/winning.yaml")
SERVING_PATH = Path("configs/serving.yaml")
SERVING_EXAMPLE_PATH = Path("configs/serving.example.yaml")


class FrozenStack(BaseModel):
    """Locked ingest + retrieval recipe. Later product code reads this, not the eval grid."""

    model_config = ConfigDict(extra="ignore")

    parser_engine: ParserEngine = ParserEngine.pymupdf
    parser_profile: ParserProfile = ParserProfile.ocr_fallback
    chunk: ChunkConfig
    embed: EmbedConfig
    vector_store: VectorStoreKind
    retrieval: RetrievalConfig

    def parser(self) -> ParserConfig:
        return ParserConfig(engine=self.parser_engine, profile=self.parser_profile)

    def ingest_config(
        self,
        *,
        corpus_id: str,
        job_id: str,
        files: list[RawDocument],
        chroma: ChromaConfig | None = None,
        qdrant: QdrantConfig | None = None,
    ) -> IngestJobConfig:
        """Build an ingest job that matches the locked stack (document-agnostic files)."""
        return IngestJobConfig(
            corpus_id=corpus_id,
            job_id=job_id,
            files=files,
            parser=self.parser(),
            chunk=self.chunk.model_copy(),
            embed=self.embed.model_copy(),
            vector_store=self.vector_store,
            retrieval=self.retrieval.model_copy(),
            chroma=chroma or ChromaConfig(),
            qdrant=qdrant or QdrantConfig(),
        )


class ServingPointer(BaseModel):
    """Machine-local index handle. Job ids live under artifacts/ and are not the stack."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    jobs_dir: Path = Field(default_factory=lambda: Path("artifacts/jobs"))


def load_frozen_stack(path: Path | str = WINNING_PATH) -> FrozenStack:
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"No frozen stack at {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise IngestError(f"{p}: expected a mapping")
    return FrozenStack.model_validate(raw)


def try_load_frozen_stack(path: Path | str = WINNING_PATH) -> FrozenStack | None:
    try:
        return load_frozen_stack(path)
    except IngestError:
        return None


def load_serving_pointer(path: Path | str = SERVING_PATH) -> ServingPointer:
    p = Path(path)
    if not p.is_file():
        raise IngestError(
            f"No serving pointer at {p}. Copy {SERVING_EXAMPLE_PATH} and set job_id "
            "to a local ingest job built with the frozen stack."
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise IngestError(f"{p}: expected a mapping")
    pointer = ServingPointer.model_validate(raw)
    if not str(pointer.job_id).strip():
        raise IngestError(f"{p}: job_id must be non-empty")
    return pointer


def retrieval_config_from_mapping(
    raw: dict | None,
    *,
    fallback: RetrievalConfig | None = None,
) -> RetrievalConfig:
    """Build RetrievalConfig from a combo/eval dict. Missing keys take fallback defaults."""
    base = fallback or RetrievalConfig()
    data = raw or {}

    def _get(name: str, default):
        if name not in data or data[name] is None:
            return default
        return data[name]

    return RetrievalConfig(
        mode=RetrievalMode(_get("mode", base.mode.value)),
        top_k=int(_get("top_k", base.top_k)),
        keyword_method=KeywordMethod(_get("keyword_method", base.keyword_method.value)),
        semantic_weight=float(_get("semantic_weight", base.semantic_weight)),
        keyword_weight=float(_get("keyword_weight", base.keyword_weight)),
        rrf_k=int(_get("rrf_k", base.rrf_k)),
        fetch_k=int(_get("fetch_k", base.fetch_k)),
        rerank=bool(_get("rerank", base.rerank)),
        rerank_model=str(_get("rerank_model", base.rerank_model)),
        rerank_top_n=int(_get("rerank_top_n", base.rerank_top_n)),
        sibling_fill=bool(_get("sibling_fill", base.sibling_fill)),
        parent_child=bool(_get("parent_child", base.parent_child)),
    )

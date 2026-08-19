from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class MediaType(str, Enum):
    pdf = "pdf"
    md = "md"
    txt = "txt"
    json = "json"
    xml = "xml"
    nxml = "nxml"


class ExtractionMethod(str, Enum):
    text = "text"
    ocr = "ocr"
    hybrid = "hybrid"
    na = "n/a"


class StrategyId(str, Enum):
    fixed = "fixed"
    section_aware = "section_aware"
    hierarchical = "hierarchical"
    passthrough = "passthrough"
    langchain_recursive = "langchain_recursive"
    langchain_token = "langchain_token"
    langchain_markdown = "langchain_markdown"
    semantic = "semantic"


class ParserProfile(str, Enum):
    text_only = "text_only"
    ocr_fallback = "ocr_fallback"
    ocr_all = "ocr_all"


class ParserEngine(str, Enum):
    pymupdf = "pymupdf"


class EmbedProvider(str, Enum):
    sentence_transformers = "sentence_transformers"
    openai = "openai"


class VectorStoreKind(str, Enum):
    chroma = "chroma"
    qdrant = "qdrant"


class RetrievalMode(str, Enum):
    dense = "dense"  # semantic / dense vector (kept for existing reports)
    keyword = "keyword"
    hybrid = "hybrid"


class KeywordMethod(str, Enum):
    bm25 = "bm25"
    tfidf = "tfidf"


DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LegalFlags(BaseModel):
    open_access_or_reusable: bool = False
    redistribution_ok_for_indexing: bool = False
    edition_current: bool = False
    attribution_documented: bool = False

    def complete(self) -> bool:
        return all(
            (
                self.open_access_or_reusable,
                self.redistribution_ok_for_indexing,
                self.edition_current,
                self.attribution_documented,
            )
        )


class RawDocument(BaseModel):
    doc_id: str
    filename: str
    media_type: MediaType
    document_name: str
    source_url: str = ""
    path: str
    legal: LegalFlags


class TextBlock(BaseModel):
    kind: str  # heading | paragraph | table
    text: str
    heading_level: int | None = None


class ParsedPage(BaseModel):
    page_number: int
    text: str
    extraction_method: ExtractionMethod
    blocks: list[TextBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    doc_id: str
    document_name: str
    source_url: str = ""
    media_type: MediaType
    filename: str = ""
    pages: list[ParsedPage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    text: str
    document_name: str
    section_title: str
    page_number: int | None = None
    source_url: str = ""
    strategy_id: StrategyId
    corpus_id: str
    job_id: str
    extraction_method: ExtractionMethod
    token_count: int
    embed_model_id: str
    parent_chunk_id: str | None = None
    filename: str = ""
    doc_id: str = ""

    @model_validator(mode="after")
    def pdf_pages_required(self) -> Self:
        if self.extraction_method in (ExtractionMethod.text, ExtractionMethod.ocr, ExtractionMethod.hybrid):
            if self.filename.lower().endswith(".pdf") and self.page_number is None:
                raise ValueError("PDF-derived chunks must include page_number")
        return self


class SmokeQueryConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=50)


class ChunkConfig(BaseModel):
    strategy_id: StrategyId = StrategyId.section_aware
    target_tokens: int = 400
    overlap_ratio: float = Field(default=0.12, ge=0.0, lt=1.0)
    min_tokens: int = 120
    max_tokens: int = 520
    child_tokens: int = 350
    parent_tokens: int = 800
    prefix_section_title: bool = False


class EmbedConfig(BaseModel):
    model_id: str = "BAAI/bge-m3"
    fallback_model_id: str = "BAAI/bge-small-en-v1.5"
    device: str = "auto"
    batch_size: int = Field(default=16, ge=1)
    provider: EmbedProvider = EmbedProvider.sentence_transformers


class ChromaConfig(BaseModel):
    persist_dir: str = "artifacts/indexes/chroma"


class QdrantConfig(BaseModel):
    persist_dir: str = "artifacts/indexes/qdrant"


class ParserConfig(BaseModel):
    engine: ParserEngine = ParserEngine.pymupdf
    profile: ParserProfile = ParserProfile.ocr_fallback
    ocr_lang: str = "eng"
    ocr_dpi: int = 250
    min_chars: int = 50
    min_alnum_ratio: float = 0.3


class RetrievalConfig(BaseModel):
    mode: RetrievalMode = RetrievalMode.dense
    top_k: int = Field(default=5, ge=1, le=50)
    keyword_method: KeywordMethod = KeywordMethod.bm25
    semantic_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    fetch_k: int = Field(default=20, ge=1, le=200)
    rerank: bool = False
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_top_n: int = Field(default=20, ge=1, le=200)
    sibling_fill: bool = False
    parent_child: bool = False

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> Self:
        total = self.semantic_weight + self.keyword_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"semantic_weight + keyword_weight must equal 1.0 (got {total})"
            )
        return self

    def to_combo_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "top_k": self.top_k,
            "keyword_method": self.keyword_method.value,
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
            "rrf_k": self.rrf_k,
            "fetch_k": self.fetch_k,
            "rerank": self.rerank,
            "rerank_model": self.rerank_model,
            "rerank_top_n": self.rerank_top_n,
            "sibling_fill": self.sibling_fill,
            "parent_child": self.parent_child,
        }


class IngestJobConfig(BaseModel):
    corpus_id: str
    job_id: str
    files: list[RawDocument]
    parser: ParserConfig = Field(default_factory=ParserConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    vector_store: VectorStoreKind = VectorStoreKind.chroma
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    smoke_query: SmokeQueryConfig = Field(default_factory=SmokeQueryConfig)


class SmokeHit(BaseModel):
    score: float
    text: str
    document_name: str
    section_title: str
    page_number: int | None
    chunk_id: str
    extraction_method: str = ""
    source_url: str = ""
    token_count: int = 0


class IngestReport(BaseModel):
    job_id: str
    corpus_id: str
    collection_name: str
    strategy_id: str
    embed_model_id: str
    embed_device: str = "auto"
    embed_provider: str = EmbedProvider.sentence_transformers.value
    parser_engine: str = ParserEngine.pymupdf.value
    parser_profile: str
    vector_store: str = VectorStoreKind.chroma.value
    retrieval_mode: str = RetrievalMode.dense.value
    page_count: int
    ocr_page_count: int
    chunk_count: int
    warnings: list[str] = Field(default_factory=list)
    combo: dict = Field(default_factory=dict)
    rebuild_policy: str = (
        "Replace the target collection on each job so reruns are idempotent "
        "(same corpus/strategy/model name does not duplicate chunks)."
    )

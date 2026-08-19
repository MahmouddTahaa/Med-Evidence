from __future__ import annotations

from clinical_rag.chunking.base import stamp_chunk
from clinical_rag.chunking.fixed import chunk_fixed
from clinical_rag.chunking.hierarchical import chunk_hierarchical
from clinical_rag.chunking.langchain_split import (
    chunk_langchain_markdown,
    chunk_langchain_recursive,
    chunk_langchain_token,
)
from clinical_rag.chunking.passthrough import chunk_passthrough
from clinical_rag.chunking.section_aware import chunk_section_aware
from clinical_rag.chunking.semantic import chunk_semantic
from clinical_rag.errors import IngestError
from clinical_rag.schemas import Chunk, IngestJobConfig, ParsedDocument, RawDocument, StrategyId


def chunk_parsed(parsed: ParsedDocument, raw: RawDocument, config: IngestJobConfig) -> list[Chunk]:
    strategy = config.chunk.strategy_id
    if strategy is StrategyId.passthrough:
        chunk_passthrough(parsed)
    if strategy is StrategyId.fixed:
        drafts = chunk_fixed(parsed, config.chunk)
    elif strategy is StrategyId.section_aware:
        drafts = chunk_section_aware(parsed, config.chunk)
    elif strategy is StrategyId.hierarchical:
        drafts = chunk_hierarchical(parsed, config.chunk, raw.doc_id)
    elif strategy is StrategyId.langchain_recursive:
        drafts = chunk_langchain_recursive(parsed, config.chunk)
    elif strategy is StrategyId.langchain_token:
        drafts = chunk_langchain_token(parsed, config.chunk)
    elif strategy is StrategyId.langchain_markdown:
        drafts = chunk_langchain_markdown(parsed, config.chunk)
    elif strategy is StrategyId.semantic:
        drafts = chunk_semantic(parsed, config.chunk)
    else:
        raise IngestError(f"Unknown strategy: {strategy}")
    return [
        stamp_chunk(d, raw=raw, config=config, strategy_id=strategy, index=i)
        for i, d in enumerate(drafts, start=1)
        if d.text.strip()
    ]

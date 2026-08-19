from __future__ import annotations

from clinical_rag.schemas import KeywordMethod, RetrievalConfig, RetrievalMode


def retrieval_eval_grid(*, top_k: int = 10, fetch_k: int = 20) -> list[RetrievalConfig]:
    """Locked sequential retrieval bakeoff: 7 runs, TF-IDF vs BM25 as a first-class axis."""
    configs = [
        RetrievalConfig(mode=RetrievalMode.dense, top_k=top_k),
    ]
    for method in (KeywordMethod.bm25, KeywordMethod.tfidf):
        configs.append(
            RetrievalConfig(
                mode=RetrievalMode.keyword,
                keyword_method=method,
                top_k=top_k,
            )
        )
        configs.append(
            RetrievalConfig(
                mode=RetrievalMode.hybrid,
                keyword_method=method,
                top_k=top_k,
                fetch_k=fetch_k,
                rerank=False,
            )
        )
        configs.append(
            RetrievalConfig(
                mode=RetrievalMode.hybrid,
                keyword_method=method,
                top_k=top_k,
                fetch_k=fetch_k,
                rerank=True,
            )
        )
    return configs


def retrieval_label(cfg: RetrievalConfig) -> str:
    if cfg.mode is RetrievalMode.dense:
        label = "dense"
    elif cfg.mode is RetrievalMode.keyword:
        label = f"sparse/{cfg.keyword_method.value}"
    else:
        suffix = "+rerank" if cfg.rerank else ""
        label = f"hybrid/{cfg.keyword_method.value}{suffix}"
    if cfg.sibling_fill:
        label = f"{label}+siblings"
    if cfg.parent_child:
        label = f"{label}+parent_child"
    return label

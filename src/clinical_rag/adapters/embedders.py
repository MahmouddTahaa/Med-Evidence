from __future__ import annotations

import os
from pathlib import Path

from clinical_rag.errors import IngestError
from clinical_rag.indexing.embed import Embedder, SentenceTransformerEmbedder
from clinical_rag.schemas import EmbedConfig, EmbedProvider


def _load_dotenv_keys() -> None:
    """Load OPENAI_ keys from .env into os.environ if not already set."""
    env_path = Path(".env")
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def build_embedder(cfg: EmbedConfig, *, purpose: str = "document") -> Embedder:
    """Build an embedder. purpose is 'document' (index) or 'query' (retrieve)."""
    _load_dotenv_keys()
    if cfg.provider is EmbedProvider.sentence_transformers:
        return SentenceTransformerEmbedder(cfg)
    if cfg.provider is EmbedProvider.openai:
        return OpenAIEmbedder(cfg)
    raise IngestError(f"unsupported embed provider: {cfg.provider}")


def embedder_for_index(
    *,
    model_id: str,
    provider: str | EmbedProvider = EmbedProvider.sentence_transformers,
    device: str = "auto",
    batch_size: int = 16,
    purpose: str = "query",
) -> Embedder:
    """Rebuild the embedder that matches an existing index (for smoke/eval)."""
    prov = provider if isinstance(provider, EmbedProvider) else EmbedProvider(provider)
    cfg = EmbedConfig(model_id=model_id, device=device, batch_size=batch_size, provider=prov)
    if prov is EmbedProvider.sentence_transformers:
        return SentenceTransformerEmbedder.for_index(model_id, device=device, batch_size=batch_size)
    return build_embedder(cfg, purpose=purpose)


class OpenAIEmbedder:
    def __init__(self, cfg: EmbedConfig):
        self.model_id = cfg.model_id or "text-embedding-3-small"
        self.device = "api"
        self.warnings: list[str] = []
        self._batch_size = cfg.batch_size
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise IngestError("Embed provider 'openai' requires OPENAI_API_KEY (API). Fail closed.")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise IngestError(
                "Embed provider 'openai' requires optional package. uv sync --extra openai"
            ) from exc
        self._client = OpenAI(api_key=key)

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.embeddings.create(model=self.model_id, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            out.extend([list(d.embedding) for d in ordered])
        return out

from __future__ import annotations

from typing import Protocol

from clinical_rag.config import detect_device, resolve_embed_config
from clinical_rag.errors import IngestError
from clinical_rag.schemas import EmbedConfig


class Embedder(Protocol):
    model_id: str
    device: str
    warnings: list[str]

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbedConfig, *, resolve: bool = True):
        self._requested_model = cfg.model_id
        self._fallback = cfg.fallback_model_id
        self.warnings: list[str] = []

        if resolve:
            resolved, auto_warnings = resolve_embed_config(cfg)
            self.warnings.extend(auto_warnings)
            self.model_id = resolved.model_id
            self.device = resolved.device if resolved.device != "auto" else detect_device("auto")
            self._batch_size = resolved.batch_size
        else:
            self.model_id = cfg.model_id
            self.device = cfg.device if cfg.device != "auto" else detect_device("auto")
            self._batch_size = cfg.batch_size

        self._model = None

    @classmethod
    def for_index(
        cls,
        model_id: str,
        *,
        device: str = "auto",
        batch_size: int = 16,
    ) -> SentenceTransformerEmbedder:
        """Load the exact model used at index time (no VRAM-based model swap)."""
        return cls(
            EmbedConfig(model_id=model_id, device=device, batch_size=batch_size),
            resolve=False,
        )

    def _clear_cuda(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_model(self, model_id: str, device: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_id, device=device)

    def _load(self):
        if self._model is not None:
            return
        try:
            self._model = self._load_model(self.model_id, self.device)
        except Exception as exc:
            if self._try_smaller_or_cpu_after_failure(exc):
                return
            raise IngestError(f"Failed to load embed model {self.model_id}: {exc}") from exc

    def _try_smaller_or_cpu_after_failure(self, exc: Exception) -> bool:
        err = str(exc).lower()
        if "out of memory" not in err and "cuda" not in err:
            return False

        if self.model_id != self._fallback and self._fallback:
            self.warnings.append(
                f"CUDA OOM loading {self.model_id}; retrying with {self._fallback}."
            )
            self.model_id = self._fallback
            self._model = None
            self._clear_cuda()
            try:
                self._model = self._load_model(self.model_id, self.device)
                return True
            except Exception:
                self._model = None

        if self.device == "cuda":
            self.warnings.append(f"CUDA OOM; retrying {self.model_id} on CPU.")
            self.device = "cpu"
            self._batch_size = min(self._batch_size, 8)
            self._model = None
            self._clear_cuda()
            self._model = self._load_model(self.model_id, self.device)
            return True
        return False

    def _encode_once(self, texts: list[str], batch_size: int) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        batch_size = self._batch_size
        while True:
            try:
                return self._encode_once(texts, batch_size)
            except RuntimeError as exc:
                err = str(exc).lower()
                if "out of memory" not in err:
                    raise
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    self.warnings.append(f"CUDA OOM during embed; batch_size -> {batch_size}.")
                    self._clear_cuda()
                    continue
                if self._try_smaller_or_cpu_after_failure(exc):
                    batch_size = self._batch_size
                    continue
                raise IngestError(
                    f"Embedding failed after OOM retries (model={self.model_id}, device={self.device}). "
                    "Set EMBED__MODEL_ID=BAAI/bge-small-en-v1.5 or EMBED__DEVICE=cpu in .env."
                ) from exc


def assert_query_embedder_matches_index(embedder: Embedder, index_model_id: str) -> None:
    if embedder.model_id != index_model_id:
        raise IngestError(
            f"Query embed model {embedder.model_id!r} does not match index model "
            f"{index_model_id!r}. Rebuild the index or query with the same model."
        )
